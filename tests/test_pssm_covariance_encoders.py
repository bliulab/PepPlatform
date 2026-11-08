from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import pepplatform.encoders as encoders_module
from pepplatform.encoders import (
    ACCPSSMBlosum62HybridEncoder,
    ACCPSSMEncoder,
    ACPSSMBlosum62HybridEncoder,
    ACPSSMEncoder,
    CCPSSMBlosum62HybridEncoder,
    CCPSSMEncoder,
)
from pepplatform.featurizers import FeaturizerFactory
from pepplatform.pssm_covariance import (
    build_pssm_covariance_encoder_contract,
    compute_pssm_covariance_features,
    pssm_covariance_feature_names,
)
from pepplatform.web.inference import InferenceUnsupportedError, ModelService
from pepplatform.web.preflight import preflight_experiment
from pepplatform.sequence_profiles import (
    FEATURE_ACC_PSSM_BLOSUM62_HYBRID,
    FEATURE_AC_PSSM_BLOSUM62_HYBRID,
    FEATURE_CC_PSSM_BLOSUM62_HYBRID,
)
from scripts.train_pair_pipeline import (
    _classic_extra_artifact_info,
    _feature_cache_meta,
    _stream_artifact_metadata,
    classic_feature_cache_tag,
    ensure_pair_profile_cache_ready,
)
from scripts.train_sequence_pipeline import SequencePipeline
from pepplatform.stream_spec import namespace_for_stream


def _oracle(matrix, encoder, lag):
    base_encoder = {
        "ac_pssm_blosum62_hybrid": "ac_pssm",
        "cc_pssm_blosum62_hybrid": "cc_pssm",
        "acc_pssm_blosum62_hybrid": "acc_pssm",
    }.get(encoder, encoder)
    value = np.asarray(matrix, dtype=np.float64)
    centered = value - value.mean(axis=0)
    result = []
    for distance in range(1, lag + 1):
        denominator = len(value) - distance
        if base_encoder in {"ac_pssm", "acc_pssm"}:
            result.extend(
                np.sum(
                    centered[:-distance] * centered[distance:],
                    axis=0,
                    dtype=np.float64,
                )
                / denominator
            )
        if base_encoder in {"cc_pssm", "acc_pssm"}:
            for source in range(20):
                for target in range(20):
                    if source == target:
                        continue
                    result.append(
                        np.sum(
                            centered[:-distance, source] * centered[distance:, target],
                            dtype=np.float64,
                        )
                        / denominator
                    )
    return np.asarray(result, dtype=np.float64)


@pytest.mark.parametrize(
    ("encoder", "dimension"),
    [
        ("ac_pssm", 40),
        ("cc_pssm", 760),
        ("acc_pssm", 800),
        ("ac_pssm_blosum62_hybrid", 40),
        ("cc_pssm_blosum62_hybrid", 760),
        ("acc_pssm_blosum62_hybrid", 800),
    ],
)
def test_pssm_covariance_matches_independent_oracle(encoder, dimension):
    matrix = np.arange(6 * 20, dtype=np.int32).reshape(6, 20) + (
        np.arange(6, dtype=np.int32)[:, None] ** 2
    )
    actual = compute_pssm_covariance_features(
        matrix,
        encoder=encoder,
        lag=2,
    )

    assert actual.dtype == np.float64
    assert actual.shape == (dimension,)
    np.testing.assert_allclose(
        actual,
        _oracle(matrix, encoder, 2),
        rtol=0,
        atol=1e-12,
    )


def test_cc_is_directed_excludes_diagonal_and_acc_is_lag_major():
    matrix = np.arange(7 * 20, dtype=np.int32).reshape(7, 20)
    matrix[:, 1] += np.asarray([0, 3, 1, 8, 2, 13, 5])
    ac = compute_pssm_covariance_features(matrix, encoder="ac_pssm", lag=2)
    cc = compute_pssm_covariance_features(matrix, encoder="cc_pssm", lag=2)
    acc = compute_pssm_covariance_features(matrix, encoder="acc_pssm", lag=2)
    names = pssm_covariance_feature_names("cc_pssm", 2)

    forward = names.index("cc_pssm_lag1_A_to_R")
    reverse = names.index("cc_pssm_lag1_R_to_A")
    assert cc[forward] != pytest.approx(cc[reverse])
    assert not any("A_to_A" in name for name in names)
    np.testing.assert_array_equal(acc[:20], ac[:20])
    np.testing.assert_array_equal(acc[20:400], cc[:380])
    np.testing.assert_array_equal(acc[400:420], ac[20:40])
    np.testing.assert_array_equal(acc[420:800], cc[380:760])


class _FakeProfileStore:
    matrix = torch.arange(8 * 20, dtype=torch.float32).reshape(8, 20)

    def __init__(self, *_args, **kwargs):
        self.feature_type = kwargs.get("feature_type")

    def load(self, _sample_id, _sequence):
        return SimpleNamespace(matrix=self.matrix)


@pytest.mark.parametrize(
    ("encoder_class", "dimension"),
    [(ACPSSMEncoder, 40), (CCPSSMEncoder, 760), (ACCPSSMEncoder, 800)],
)
def test_encoder_uses_complete_profile_and_returns_float32(
    monkeypatch,
    encoder_class,
    dimension,
):
    monkeypatch.setattr(encoders_module, "ProfileStore", _FakeProfileStore)
    short_limit = encoder_class(1, "ignored", lag=2)
    long_limit = encoder_class(100, "ignored", lag=2)

    short_output = short_limit.encode_sequence("A" * 8, sample_id="sample")
    long_output = long_limit.encode_sequence("A" * 8, sample_id="sample")
    assert short_output.shape == (dimension,)
    assert short_output.dtype == torch.float32
    torch.testing.assert_close(short_output, long_output, rtol=0, atol=0)
    with pytest.raises(ValueError, match="do not permit column z-score"):
        encoder_class(10, "ignored", lag=2, zscore=True)


@pytest.mark.parametrize(
    ("encoder_class", "feature_type", "first_name"),
    [
        (
            ACPSSMBlosum62HybridEncoder,
            FEATURE_AC_PSSM_BLOSUM62_HYBRID,
            "ac_pssm_blosum62_hybrid_lag1_A",
        ),
        (
            CCPSSMBlosum62HybridEncoder,
            FEATURE_CC_PSSM_BLOSUM62_HYBRID,
            "cc_pssm_blosum62_hybrid_lag1_A_to_R",
        ),
        (
            ACCPSSMBlosum62HybridEncoder,
            FEATURE_ACC_PSSM_BLOSUM62_HYBRID,
            "ac_pssm_blosum62_hybrid_lag1_A",
        ),
    ],
)
def test_hybrid_encoders_have_independent_profile_and_contract(
    monkeypatch,
    encoder_class,
    feature_type,
    first_name,
):
    monkeypatch.setattr(encoders_module, "ProfileStore", _FakeProfileStore)
    encoder = encoder_class(2, "independent-index", lag=1)

    assert encoder.store.feature_type == feature_type
    assert encoder.feature_names[0] == first_name
    assert encoder.pssm_covariance_encoder_contract["profile_feature_type"] == feature_type
    assert encoder.pssm_covariance_encoder_contract["fallback_policy"] == "blosum62_hybrid"
    assert encoder.pssm_covariance_encoder_contract["fallback_condition"] == (
        "strict_profile_unavailable_after_successful_search"
    )


@pytest.mark.parametrize("lag", [True, False, "1", 0, -1, 1.5])
def test_pssm_covariance_rejects_invalid_lag(lag):
    with pytest.raises(ValueError, match="positive integer"):
        compute_pssm_covariance_features(
            np.ones((4, 20), dtype=np.int32),
            encoder="ac_pssm",
            lag=lag,
        )


def test_pssm_covariance_rejects_short_and_noninteger_profiles():
    with pytest.raises(ValueError, match="profile length > lag"):
        compute_pssm_covariance_features(
            np.ones((2, 20), dtype=np.int32),
            encoder="ac_pssm",
            lag=2,
        )
    with pytest.raises(ValueError, match="integer PSI-BLAST scores"):
        compute_pssm_covariance_features(
            np.full((3, 20), 0.5),
            encoder="ac_pssm",
            lag=1,
        )


def test_factory_rejects_incompatible_tokenizer_and_k():
    factory = FeaturizerFactory(bpe_path="")
    with pytest.raises(ValueError, match="tokenizer=kmer and k=1"):
        factory.create("wordpiece", "ac_pssm", max_length=10, k=1)
    with pytest.raises(ValueError, match="requires k=1"):
        factory.create("kmer", "acc_pssm", max_length=10, k=2)


def test_factory_requires_and_uses_independent_hybrid_index(monkeypatch):
    monkeypatch.setattr(encoders_module, "ProfileStore", _FakeProfileStore)
    missing = FeaturizerFactory(bpe_path="", pssm_index_path="strict-index")
    with pytest.raises(ValueError, match="ac_pssm_blosum62_hybrid_index_path"):
        missing.create(
            "kmer",
            "ac_pssm_blosum62_hybrid",
            max_length=10,
            k=1,
        )

    factory = FeaturizerFactory(
        bpe_path="",
        pssm_index_path="strict-index",
        ac_pssm_blosum62_hybrid_index_path="hybrid-index",
        ac_pssm_blosum62_hybrid_lag=2,
    )
    featurizer = factory.create(
        "kmer",
        "ac_pssm_blosum62_hybrid",
        max_length=10,
        k=1,
    )
    assert featurizer.encoder.index_path == "hybrid-index"
    assert featurizer.encoder.lag == 2


@pytest.mark.parametrize(
    ("encoder", "index_attribute"),
    [
        (
            "ac_pssm_blosum62_hybrid",
            "ac_pssm_blosum62_hybrid_index_path",
        ),
        (
            "cc_pssm_blosum62_hybrid",
            "cc_pssm_blosum62_hybrid_index_path",
        ),
        (
            "acc_pssm_blosum62_hybrid",
            "acc_pssm_blosum62_hybrid_index_path",
        ),
        ("ac_pssm", "pssm_index_path"),
        ("cc_pssm", "pssm_index_path"),
        ("acc_pssm", "pssm_index_path"),
    ],
)
def test_sequence_build_featurizer_checks_active_profile_index(
    tmp_path,
    encoder,
    index_attribute,
):
    active_index = tmp_path / f"{encoder}-index.json"
    active_index.write_text("{}", encoding="ascii")
    pipeline = object.__new__(SequencePipeline)
    pipeline.args = SimpleNamespace(
        tokenizer="kmer",
        encoder=encoder,
        max_length=10,
        k=1,
    )
    pipeline.pssm_index_path = str(tmp_path / "missing-pssm-index.json")
    pipeline.psfm_index_path = ""
    pipeline.frequency_profile_index_path = ""
    setattr(pipeline, index_attribute, str(active_index))
    pipeline.ensure_static_embedding_ready = lambda: None
    pipeline.ensure_pssm_cache_ready = lambda: None
    pipeline.ensure_bpe_tokenizer_ready = lambda: None
    pipeline.ensure_wordpiece_tokenizer_ready = lambda: None
    pipeline.ensure_unigram_tokenizer_ready = lambda: None

    class FakeFactory:
        def create(self, *args, **kwargs):
            return SimpleNamespace()

    pipeline.feat_factory = FakeFactory()

    assert pipeline.build_featurizer() is not None


@pytest.mark.parametrize(
    ("encoder", "index_attribute"),
    [
        (
            "ac_pssm_blosum62_hybrid",
            "ac_pssm_blosum62_hybrid_index_path",
        ),
        (
            "cc_pssm_blosum62_hybrid",
            "cc_pssm_blosum62_hybrid_index_path",
        ),
        (
            "acc_pssm_blosum62_hybrid",
            "acc_pssm_blosum62_hybrid_index_path",
        ),
    ],
)
def test_sequence_build_featurizer_reports_missing_hybrid_profile_index(
    tmp_path,
    encoder,
    index_attribute,
):
    pipeline = object.__new__(SequencePipeline)
    pipeline.args = SimpleNamespace(
        tokenizer="kmer",
        encoder=encoder,
        max_length=10,
        k=1,
    )
    pipeline.pssm_index_path = str(tmp_path / "generic-pssm-index.json")
    Path(pipeline.pssm_index_path).write_text("{}", encoding="ascii")
    setattr(pipeline, index_attribute, str(tmp_path / "missing-hybrid-index.json"))
    pipeline.ensure_static_embedding_ready = lambda: None
    pipeline.ensure_pssm_cache_ready = lambda: None
    pipeline.ensure_bpe_tokenizer_ready = lambda: None
    pipeline.ensure_wordpiece_tokenizer_ready = lambda: None
    pipeline.ensure_unigram_tokenizer_ready = lambda: None

    class FakeFactory:
        def create(self, *args, **kwargs):
            return SimpleNamespace()

    pipeline.feat_factory = FakeFactory()

    with pytest.raises(
        SystemExit,
        match=rf"{encoder} profile index not found:.*--{index_attribute}",
    ):
        pipeline.build_featurizer()


def test_training_metadata_and_cache_identity_include_contract_and_manifest():
    contract = build_pssm_covariance_encoder_contract(
        encoder="cc_pssm",
        lag=3,
    )
    pipeline = object.__new__(SequencePipeline)
    pipeline.args = SimpleNamespace(
        tokenizer="kmer",
        encoder="cc_pssm",
        max_length=4,
        k=1,
        ac_lag=10,
        cc_lag=10,
        acc_lag=10,
        ac_pssm_lag=10,
        cc_pssm_lag=3,
        acc_pssm_lag=10,
        covariance_max_raw_length=100000,
        artifact_dir="",
        _pssm_covariance_encoder_contract=contract,
    )
    classic = pipeline._sequence_feature_artifact_info(SimpleNamespace(label_names=["positive"]))[
        "classic"
    ]
    pair = _classic_extra_artifact_info(
        Namespace(
            tokenizer="kmer",
            encoder="cc_pssm",
            _pssm_covariance_encoder_contract=contract,
        )
    )
    cache_args = SimpleNamespace(
        tokenizer="kmer",
        encoder="cc_pssm",
        max_length=4,
        k=1,
        cc_pssm_lag=3,
        pssm_index_path="profiles/index.json",
        _feature_train_sha256="train-digest",
        _profile_manifest_sha256="a" * 64,
        _pssm_covariance_encoder_contract=contract,
    )

    assert classic["cc_pssm_lag"] == 3
    assert classic["pssm_covariance_encoder_contract"] == contract
    assert pair["pssm_covariance_encoder_contract"] == contract
    first = _feature_cache_meta(
        cache_args,
        "train",
        "sequence",
        [("sample", "ACDE")],
    )
    first_tag = classic_feature_cache_tag(cache_args, "cc_pssm")
    cache_args._profile_manifest_sha256 = "b" * 64
    second = _feature_cache_meta(
        cache_args,
        "train",
        "sequence",
        [("sample", "ACDE")],
    )
    second_tag = classic_feature_cache_tag(cache_args, "cc_pssm")
    assert first["pssm_covariance_encoder_contract"] == contract
    assert first["profile_manifest_sha256"] == "a" * 64
    assert first != second
    assert first_tag != second_tag


def test_hybrid_stream_metadata_preserves_independent_path_lag_and_contract(
    monkeypatch,
):
    monkeypatch.setattr(encoders_module, "ProfileStore", _FakeProfileStore)
    encoder_name = "cc_pssm_blosum62_hybrid"
    path_key = "cc_pssm_blosum62_hybrid_index_path"
    lag_key = "cc_pssm_blosum62_hybrid_lag"
    contract = build_pssm_covariance_encoder_contract(
        encoder=encoder_name,
        lag=3,
    )
    stream_spec = {
        "feature_route": "classic",
        "tokenizer": "kmer",
        "encoder": encoder_name,
        "stream_model": "",
        "model_source": "",
        "max_length": 60,
        "params": {
            "k": 1,
            path_key: "independent-index",
            lag_key: 3,
        },
    }
    args = Namespace(
        streams={"peptide": stream_spec},
        pssm_index_path="generic-index-must-not-be-used",
        _stream_runtime={
            "peptide": {
                "pssm_covariance_encoder_contract": contract,
            }
        },
        artifact_dir="",
    )

    stream_args = namespace_for_stream(args, "peptide")
    assert getattr(stream_args, path_key) == "independent-index"
    assert getattr(stream_args, lag_key) == 3

    stream = _stream_artifact_metadata(
        args,
        "peptide",
        torch.ones((2, 1140)),
        "vector",
    )
    prepared = ModelService._stream_legacy_metadata(
        {"streams": {"peptide": stream}, "args": {}},
        "peptide",
    )
    restored, encoder = ModelService._build_classic_featurizer(prepared)

    assert stream["spec"]["params"][path_key] == "independent-index"
    assert stream["spec"]["params"][lag_key] == 3
    assert stream["pssm_covariance_encoder_contract"] == contract
    assert prepared["classic"][path_key] == "independent-index"
    assert prepared["classic"][lag_key] == 3
    assert prepared["classic"]["pssm_covariance_encoder_contract"] == contract
    assert encoder == encoder_name
    assert restored.feature_dim == 1140
    assert Path(restored.encoder.index_path) == Path.cwd() / "independent-index"


def test_inference_requires_exact_pssm_covariance_contract(monkeypatch):
    monkeypatch.setattr(encoders_module, "ProfileStore", _FakeProfileStore)
    contract = build_pssm_covariance_encoder_contract(
        encoder="ac_pssm",
        lag=2,
    )
    metadata = {
        "classic": {
            "tokenizer": "kmer",
            "encoder": "ac_pssm",
            "max_length": 3,
            "k": 1,
            "ac_pssm_lag": 2,
            "pssm_index_path": "ignored",
            "pssm_covariance_encoder_contract": contract,
        }
    }
    restored, encoder = ModelService._build_classic_featurizer(metadata)
    assert encoder == "ac_pssm"
    assert restored.feature_dim == 40

    missing = deepcopy(metadata)
    del missing["classic"]["pssm_covariance_encoder_contract"]
    with pytest.raises(InferenceUnsupportedError) as missing_error:
        ModelService._build_classic_featurizer(missing)
    assert missing_error.value.code == "pssm_covariance_contract_unsupported"

    tampered = deepcopy(metadata)
    tampered["classic"]["pssm_covariance_encoder_contract"]["output_dimension"] = 999
    with pytest.raises(InferenceUnsupportedError) as tampered_error:
        ModelService._build_classic_featurizer(tampered)
    assert tampered_error.value.code == "pssm_covariance_contract_unsupported"


def test_inference_restores_independent_hybrid_contract(monkeypatch):
    monkeypatch.setattr(encoders_module, "ProfileStore", _FakeProfileStore)
    encoder_name = "cc_pssm_blosum62_hybrid"
    contract = build_pssm_covariance_encoder_contract(
        encoder=encoder_name,
        lag=2,
    )
    metadata = {
        "classic": {
            "tokenizer": "kmer",
            "encoder": encoder_name,
            "max_length": 3,
            "k": 1,
            "cc_pssm_blosum62_hybrid_lag": 2,
            "cc_pssm_blosum62_hybrid_index_path": "independent-index",
            "pssm_covariance_encoder_contract": contract,
        }
    }

    restored, encoder = ModelService._build_classic_featurizer(metadata)
    assert encoder == encoder_name
    assert restored.feature_dim == 760
    assert Path(restored.encoder.index_path) == Path.cwd() / "independent-index"

    tampered = deepcopy(metadata)
    tampered["classic"]["pssm_covariance_encoder_contract"]["profile_feature_type"] = (
        "pssm_blosum62_hybrid"
    )
    with pytest.raises(InferenceUnsupportedError) as error:
        ModelService._build_classic_featurizer(tampered)
    assert error.value.code == "pssm_covariance_contract_unsupported"


def _dataset(task_type, *, sequence_min=8, peptide_min=8, partner_min=10):
    roles = {
        "sequence": {"count": 12, "min": sequence_min, "max": 60},
        "peptide": {"count": 12, "min": peptide_min, "max": 50},
        "partner": {"count": 12, "min": partner_min, "max": 512},
    }
    return {
        "id": "dataset",
        "name": "dataset",
        "task_type": task_type,
        "format": "sequence_dir" if task_type == "pep_function" else "pair_split_dir",
        "metadata": {
            "splits": {"train": 10, "val": 2},
            "length_stats": {
                "version": 1,
                "by_split": {},
                "overall": roles,
            },
        },
    }


def test_preflight_checks_single_and_pair_pssm_covariance_minima():
    single = preflight_experiment(
        _dataset("pep_function", sequence_min=8),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "ac_pssm",
            "model": "mlp",
            "k": 1,
            "ac_pssm_lag": 8,
        },
    )
    assert "pssm_covariance_lag_too_large" in single["error_codes"]

    homogeneous = preflight_experiment(
        _dataset("pep_protein_interaction", peptide_min=8, partner_min=10),
        "pep_protein_interaction",
        {
            "tokenizer": "kmer",
            "encoder": "cc_pssm",
            "model": "mlp",
            "classifier": "mlp",
            "pair_strategy": "two_stream",
            "interaction_level": "vector",
            "stream_mode": "homogeneous",
            "k": 1,
            "cc_pssm_lag": 8,
        },
    )
    assert "pssm_covariance_lag_too_large" in homogeneous["error_codes"]

    heterogeneous = preflight_experiment(
        _dataset("pep_protein_interaction", peptide_min=8, partner_min=10),
        "pep_protein_interaction",
        {
            "tokenizer": "kmer",
            "encoder": "index",
            "model": "mlp",
            "classifier": "mlp",
            "pair_strategy": "two_stream",
            "interaction_level": "vector",
            "stream_mode": "heterogeneous",
            "streams": {
                "peptide": {
                    "feature_route": "classic",
                    "tokenizer": "kmer",
                    "encoder": "index",
                    "max_length": 60,
                    "params": {"k": 1},
                },
                "partner": {
                    "feature_route": "classic",
                    "tokenizer": "kmer",
                    "encoder": "acc_pssm",
                    "max_length": 512,
                    "params": {"k": 1, "acc_pssm_lag": 10},
                },
            },
        },
    )
    assert "pssm_covariance_lag_too_large" in heterogeneous["error_codes"]


def test_preflight_requires_current_length_statistics():
    dataset = _dataset("pep_function")
    del dataset["metadata"]["length_stats"]
    report = preflight_experiment(
        dataset,
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "ac_pssm",
            "model": "mlp",
            "k": 1,
            "ac_pssm_lag": 2,
        },
    )
    assert "pssm_covariance_length_stats_unavailable" in report["error_codes"]


def test_training_entrypoints_reject_short_sequences_before_profile_build(
    tmp_path,
):
    sequence_root = tmp_path / "sequence"
    train_root = sequence_root / "train"
    train_root.mkdir(parents=True)
    (train_root / "seqs.fasta").write_text(
        ">short\nACDE\n",
        encoding="ascii",
    )
    pipeline = object.__new__(SequencePipeline)
    pipeline.args = SimpleNamespace(
        tokenizer="kmer",
        encoder="ac_pssm",
        ac_pssm_lag=4,
        data_dir=str(sequence_root),
    )
    with pytest.raises(SystemExit, match="minimum sequence length is 4"):
        pipeline.ensure_pssm_cache_ready()

    pair_csv = tmp_path / "pairs.csv"
    pair_csv.write_text(
        "sample_id,peptide,protein,label\nshort,ACDE,ACDEFG,1\n",
        encoding="ascii",
    )
    pair_args = SimpleNamespace(
        tokenizer="kmer",
        encoder="cc_pssm",
        cc_pssm_lag=4,
        task_type="pep_protein_interaction",
    )
    with pytest.raises(SystemExit, match="minimum sequence length is 4"):
        ensure_pair_profile_cache_ready(
            pair_args,
            str(tmp_path),
            str(pair_csv),
            "",
            "",
        )
