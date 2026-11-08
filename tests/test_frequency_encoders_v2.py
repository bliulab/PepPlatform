import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pepplatform.encoders import BoWEncoder, TfidfEncoder
from pepplatform.featurizers import FeaturizerFactory
from pepplatform.frequency import (
    FREQUENCY_ENCODER_CONTRACT_VERSION,
    count_feature_ids,
    save_idf_state,
    validate_idf_vector,
)
from pepplatform.lda import validate_lda_upstream_contract
from pepplatform.plsa import validate_plsa_upstream_contract
from pepplatform.web.inference import InferenceUnsupportedError, ModelService
from scripts.train_pair_pipeline import build_classic_featurizer
from scripts.train_sequence_pipeline import Pipeline


def _factory():
    return FeaturizerFactory(bpe_path="")


def test_kmer_bow_v2_uses_full_stream_raw_counts_and_counts_unk():
    featurizer = _factory().create(
        "kmer",
        "bow",
        max_length=1,
        k=1,
    )

    encoded = featurizer.encode("AAAA*")

    assert encoded[featurizer.tokenizer.kmer_vocab["A"]].item() == 4.0
    assert encoded[featurizer.tokenizer.kmer_vocab["<UNK>"]].item() == 1.0
    assert encoded.sum().item() == 5.0
    assert featurizer.frequency_encoder_contract["stream_policy"] == ("full_token_stream")


def test_tfidf_v2_formula_includes_zero_token_training_documents():
    featurizer = _factory().create(
        "kmer",
        "tfidf",
        max_length=1,
        k=2,
        sequences=["AA", "CC", ""],
    )
    aa = featurizer.tokenizer.kmer_vocab["AA"]
    unseen = featurizer.tokenizer.kmer_vocab["AC"]

    assert featurizer.encoder.idf[aa].item() == pytest.approx(
        torch.log(torch.tensor(2.0)).item() + 1.0
    )
    assert featurizer.encoder.idf[unseen].item() == pytest.approx(
        torch.log(torch.tensor(4.0)).item() + 1.0
    )
    assert featurizer.encode("AAAA")[aa].item() == pytest.approx(featurizer.encoder.idf[aa].item())
    assert torch.equal(
        featurizer.encode(""),
        torch.zeros(featurizer.vocab_size),
    )
    assert featurizer.frequency_encoder_contract["training_document_count"] == 3


def test_tfidf_requires_exactly_one_nonempty_state_source():
    factory = _factory()
    with pytest.raises(ValueError, match="training corpus is empty"):
        factory.create(
            "kmer",
            "tfidf",
            max_length=2,
            k=1,
            sequences=[],
        )

    fitted = factory.create(
        "kmer",
        "tfidf",
        max_length=2,
        k=1,
        sequences=["AA"],
    )
    with pytest.raises(ValueError, match="not both"):
        factory.create(
            "kmer",
            "tfidf",
            max_length=2,
            k=1,
            sequences=["AA"],
            persisted_idf=fitted.encoder.idf,
        )


@pytest.mark.parametrize(
    "idf",
    [
        [1.0],
        [float("nan"), 1.0],
        [float("inf"), 1.0],
        [0.0, 1.0],
        [-1.0, 1.0],
    ],
)
def test_idf_validation_is_strict(idf):
    with pytest.raises(ValueError):
        validate_idf_vector(idf, feature_dim=2, context="test")


@pytest.mark.parametrize("ids", [[1.0], [True], [-1], [3]])
def test_feature_id_validation_rejects_noninteger_and_out_of_range(ids):
    with pytest.raises(ValueError):
        count_feature_ids(ids, feature_dim=3)


def test_frequency_raw_length_limit_is_independent_of_max_length():
    featurizer = _factory().create(
        "kmer",
        "bow",
        max_length=1,
        k=1,
        frequency_max_raw_length=3,
    )
    with pytest.raises(
        ValueError,
        match="exceeds frequency_max_raw_length=3",
    ):
        featurizer.encode("AAAA")


@pytest.mark.parametrize(
    ("tokenizer", "kwargs"),
    [
        ("kmer", {}),
        ("dr", {"dr_max_distance": 1}),
    ],
)
def test_nonprofile_tfidf_persisted_state_round_trip(tokenizer, kwargs):
    factory = _factory()
    fitted = factory.create(
        tokenizer,
        "tfidf",
        max_length=1,
        k=1,
        sequences=["AAAA", "CCCC"],
        **kwargs,
    )
    restored = factory.create(
        tokenizer,
        "tfidf",
        max_length=1,
        k=1,
        persisted_idf=fitted.encoder.idf,
        **kwargs,
    )

    assert torch.equal(fitted.encoder.idf, restored.encoder.idf)
    assert torch.equal(fitted.encode("AAC"), restored.encode("AAC"))


def test_v1_generic_bow_and_tfidf_preserve_legacy_truncation():
    factory = _factory()
    bow = factory.create(
        "kmer",
        "bow",
        max_length=2,
        k=1,
        frequency_contract_version=1,
    )
    tfidf = factory.create(
        "kmer",
        "tfidf",
        max_length=2,
        k=1,
        sequences=["AAAA"],
        frequency_contract_version=1,
    )
    a_id = bow.tokenizer.kmer_vocab["A"]

    assert bow.encode("AAAA")[a_id].item() == 1.0
    assert bow.encode("AAAA").sum().item() == 1.0
    assert tfidf.encode("AAAA")[a_id].item() == 1.0
    assert bow.frequency_encoder_contract["contract_version"] == 1


def _v2_bundle(tmp_path):
    fitted = _factory().create(
        "kmer",
        "tfidf",
        max_length=1,
        k=1,
        sequences=["AAAA", "CCCC"],
    )
    descriptor = save_idf_state(
        fitted.encoder.idf,
        artifact_dir=tmp_path,
        filename="idf.pt",
    )
    contract = dict(fitted.frequency_encoder_contract)
    contract["training_corpus_sha256"] = hashlib.sha256(b"train-corpus").hexdigest()
    contract["fit_scope"] = "sequence_train_split"
    metadata = {
        "classic": {
            "tokenizer": "kmer",
            "encoder": "tfidf",
            "max_length": 1,
            "k": 1,
            "frequency_max_raw_length": 50000,
            "frequency_encoder_contract": contract,
            "tfidf_idf": descriptor,
            "tfidf_idf_path": descriptor["path"],
        },
        "args": {},
    }
    return fitted, metadata


def test_v2_bundle_round_trip_verifies_descriptor(tmp_path):
    fitted, metadata = _v2_bundle(tmp_path)

    restored, encoder = ModelService._build_classic_featurizer(
        metadata,
        bundle=tmp_path,
    )

    assert encoder == "tfidf"
    assert torch.equal(restored.encoder.idf, fitted.encoder.idf)
    assert torch.equal(restored.encode("AAC"), fitted.encode("AAC"))
    assert (
        restored.frequency_encoder_contract["contract_version"]
        == FREQUENCY_ENCODER_CONTRACT_VERSION
    )


def test_v2_bundle_rejects_missing_and_tampered_idf(tmp_path):
    _fitted, metadata = _v2_bundle(tmp_path)
    Path(tmp_path, "idf.pt").unlink()
    with pytest.raises(InferenceUnsupportedError, match="missing IDF"):
        ModelService._build_classic_featurizer(metadata, bundle=tmp_path)

    _fitted, metadata = _v2_bundle(tmp_path)
    with Path(tmp_path, "idf.pt").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(InferenceUnsupportedError, match="digest mismatch"):
        ModelService._build_classic_featurizer(metadata, bundle=tmp_path)


def test_v2_bundle_rejects_descriptor_shape_and_vocab_mismatch(tmp_path):
    _fitted, metadata = _v2_bundle(tmp_path)
    metadata["classic"]["tfidf_idf"]["shape"] = [27]
    with pytest.raises(
        InferenceUnsupportedError,
        match="does not match its descriptor",
    ):
        ModelService._build_classic_featurizer(metadata, bundle=tmp_path)

    _fitted, metadata = _v2_bundle(tmp_path)
    metadata["classic"]["frequency_encoder_contract"]["vocabulary_sha256"] = "0" * 64
    with pytest.raises(
        InferenceUnsupportedError,
        match="does not match the tokenizer state",
    ):
        ModelService._build_classic_featurizer(metadata, bundle=tmp_path)


def test_v2_bundle_rejects_idf_path_escape(tmp_path):
    outside = tmp_path.parent / "outside-idf.pt"
    torch.save(torch.ones(28), outside)
    _fitted, metadata = _v2_bundle(tmp_path)
    metadata["classic"]["tfidf_idf"] = {
        "path": "../outside-idf.pt",
        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
        "dtype": "float32",
        "shape": [28],
    }

    with pytest.raises(InferenceUnsupportedError, match="escapes"):
        ModelService._build_classic_featurizer(metadata, bundle=tmp_path)


def test_legacy_tfidf_without_complete_state_is_rejected():
    metadata = {
        "classic": {
            "tokenizer": "kmer",
            "encoder": "tfidf",
            "max_length": 2,
            "k": 1,
        },
        "args": {},
    }
    with pytest.raises(
        InferenceUnsupportedError,
        match="missing its train-only IDF",
    ):
        ModelService._build_classic_featurizer(metadata)


def test_pair_tfidf_fit_scope_is_combined_or_role_specific(tmp_path):
    rows = [
        ("first", "AA", "CC", 1),
        ("second", "AA", "DD", 0),
    ]

    def args():
        return SimpleNamespace(
            tokenizer="kmer",
            encoder="tfidf",
            max_length=1,
            k=1,
            frequency_max_raw_length=50_000,
        )

    combined_args = args()
    combined = build_classic_featurizer(
        combined_args,
        str(tmp_path),
        rows,
    )
    peptide_args = args()
    peptide = build_classic_featurizer(
        peptide_args,
        str(tmp_path),
        rows,
        sequence_role="peptide",
    )
    a_id = combined.tokenizer.kmer_vocab["A"]

    assert combined.encoder.idf[a_id].item() == pytest.approx(
        torch.log(torch.tensor(5.0 / 3.0)).item() + 1.0
    )
    assert peptide.encoder.idf[a_id].item() == pytest.approx(1.0)
    assert combined_args._frequency_encoder_contract["fit_scope"] == "combined_pair_roles"
    assert peptide_args._frequency_encoder_contract["fit_scope"] == "peptide_role"


def test_sequence_artifact_export_includes_v2_contract_and_signed_idf(
    tmp_path,
):
    fitted = _factory().create(
        "kmer",
        "tfidf",
        max_length=2,
        k=1,
        sequences=["AA", "CC"],
    )
    contract = dict(fitted.frequency_encoder_contract)
    contract["training_corpus_sha256"] = hashlib.sha256(b"train").hexdigest()
    contract["fit_scope"] = "sequence_train_split"
    pipeline = object.__new__(Pipeline)
    pipeline.args = SimpleNamespace(
        tokenizer="kmer",
        encoder="tfidf",
        max_length=2,
        k=1,
        artifact_dir=str(tmp_path),
        _classic_tfidf_idf=fitted.encoder.idf,
        _frequency_encoder_contract=contract,
    )

    info = pipeline._sequence_feature_artifact_info(SimpleNamespace(label_names=["positive"]))

    classic = info["classic"]
    assert classic["frequency_encoder_contract"] == contract
    assert classic["tfidf_idf"]["path"] == "classic_tfidf_idf.pt"
    assert classic["tfidf_idf"]["shape"] == [fitted.vocab_size]
    assert classic["tfidf_idf"]["dtype"] == "float32"
    assert len(classic["tfidf_idf"]["sha256"]) == 64


@pytest.mark.parametrize(
    ("tokenizer", "validator"),
    [
        ("dr", validate_lda_upstream_contract),
        ("top-n-gram", validate_plsa_upstream_contract),
    ],
)
def test_sequence_bow_contract_is_bound_to_train_split(
    tmp_path,
    tokenizer,
    validator,
):
    data_dir = tmp_path / "data"
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "seqs.fasta").write_text(
        ">Train1\nAR\n>Train2\nRA\n",
        encoding="ascii",
    )
    profile_index = tmp_path / "target-frequency-index.json"
    profile_index.write_text("{}", encoding="utf-8")
    captured = {}

    class FakeFeaturizerFactory:
        def create(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                encoder=SimpleNamespace(),
                tokenizer=SimpleNamespace(profile_store=SimpleNamespace(manifest_sha256="a" * 64)),
                frequency_encoder_contract={
                    "contract_version": 2,
                    "stream_policy": "full_token_stream",
                    "bow_formula": "x_j = c_j",
                    "tf_formula": "c_j / sum_k(c_k)",
                    "idf_formula": "ln((N + 1) / (df_j + 1)) + 1",
                    "post_normalization": "none",
                    "frequency_max_raw_length": 50_000,
                    "feature_dimension": 20,
                    "vocabulary_sha256": "b" * 64,
                    "encoder": "bow",
                    "training_document_count": 0,
                    "training_corpus_sha256": None,
                    "fit_scope": "not_applicable",
                },
            )

    pipeline = object.__new__(Pipeline)
    pipeline.args = SimpleNamespace(
        tokenizer=tokenizer,
        encoder="bow",
        data_dir=str(data_dir),
        max_length=208,
        k=1,
        dr_max_distance=5,
        top_ngram_n=2,
        dt_max_distance=5,
        frequency_max_raw_length=50_000,
        sequence_model="",
        model="logreg",
        profile_on_missing="blosum62_target_frequency",
        frequency_profile_index_path=str(profile_index),
        artifact_dir=str(tmp_path / "artifact"),
    )
    pipeline.feat_factory = FakeFeaturizerFactory()
    pipeline.frequency_profile_index_path = str(profile_index)
    pipeline.pssm_index_path = ""
    pipeline.psfm_index_path = ""
    pipeline.word2vec_path = ""
    pipeline.glove_path = ""
    pipeline.fasttext_path = ""
    pipeline.ensure_static_embedding_ready = lambda: None
    pipeline.ensure_pssm_cache_ready = lambda: None
    pipeline.ensure_bpe_tokenizer_ready = lambda: None
    pipeline.ensure_wordpiece_tokenizer_ready = lambda: None
    pipeline.ensure_unigram_tokenizer_ready = lambda: None

    pipeline.build_featurizer()
    info = pipeline._sequence_feature_artifact_info(SimpleNamespace(label_names=["positive"]))
    contract = info["classic"]["frequency_encoder_contract"]

    assert captured["kwargs"]["sequences"] == ["AR", "RA"]
    if tokenizer == "top-n-gram":
        assert captured["kwargs"]["profile_samples"] == [
            ("Train1", "AR"),
            ("Train2", "RA"),
        ]
        assert info["classic"]["frequency_profile_index_path"] == str(profile_index)
    else:
        assert captured["kwargs"]["profile_samples"] is None
    assert contract["fit_scope"] == "sequence_train_split"
    assert len(contract["training_corpus_sha256"]) == 64
    validator(info["classic"], input_feature_count=20)
