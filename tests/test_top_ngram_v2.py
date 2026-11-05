import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pepplatform.encoders import (
    TopNGramCountEncoder,
    TopNGramTextRankEncoder,
    TopNGramTfidfEncoder,
)
from pepplatform.featurizers import FeaturizerFactory
from pepplatform.tokenizers import TopNGramTokenizer
from pepplatform.top_ngram import (
    NCBISTDAA_CODES,
    TOP_N_GRAM_AA_ORDER,
    TOP_N_GRAM_CONTRACT_VERSION,
    TOP_N_GRAM_FALLBACK_POLICY,
    TOP_N_GRAM_FALLBACK_POLICY_VERSION,
    TOP_N_GRAM_PROFILE_FEATURE_TYPE,
    TOP_N_GRAM_PROFILE_SOURCE_FALLBACK,
    TOP_N_GRAM_PROFILE_SOURCE_PSI_BLAST,
    TOP_N_GRAM_STRICT_PROFILE_POLICY,
    TOP_N_GRAM_TIE_POLICY,
    TopNGramContractError,
    TargetFrequencyProfileStore,
    blosum62_target_frequency_profile,
    configured_top_ngram_max_residues,
    normalize_top_ngram_sequence,
    parse_psiblast_target_frequencies,
    resolve_top_ngram_max_length,
    save_target_frequency_matrix,
    sequence_sha256,
    sha256_file,
    top_ngram_feature_dim,
    validate_top_ngram_n,
)
from pepplatform.web.inference import InferenceUnsupportedError, ModelService
from pepplatform.sequence_profiles import (
    PROFILE_CONTRACT_VERSION,
    PROFILE_SCHEMA,
)
from scripts.train_pair_pipeline import (
    _feature_cache_meta,
    classic_feature_cache_tag,
    resolve_pair_top_ngram_lengths,
)


def _write_profile_manifest(tmp_path, samples):
    profile_dir = tmp_path / "target_frequency"
    profile_dir.mkdir(parents=True)
    entries = {}
    for item in samples:
        sample_id, sequence, matrix = item[:3]
        metadata = item[3] if len(item) > 3 else {}
        fallback = bool(metadata.get("fallback", False))
        path = profile_dir / f"{sample_id}.npz"
        save_target_frequency_matrix(path, matrix)
        entries[sample_id] = {
            "path": path.name,
            "matrix_sha256": sha256_file(path),
            "sequence_length": len(sequence),
            "sequence_sha256": sequence_sha256(sequence),
            "zero_nonself_hits": metadata.get("zero_nonself_hits", False),
            "fallback": fallback,
            "profile_source": metadata.get(
                "profile_source",
                (
                    TOP_N_GRAM_PROFILE_SOURCE_FALLBACK
                    if fallback
                    else TOP_N_GRAM_PROFILE_SOURCE_PSI_BLAST
                ),
            ),
            "fallback_reason": metadata.get(
                "fallback_reason",
                "checkpoint_output_missing" if fallback else "",
            ),
        }
    manifest = {
        "contract_version": TOP_N_GRAM_CONTRACT_VERSION,
        "feature_type": TOP_N_GRAM_PROFILE_FEATURE_TYPE,
        "aa_order": TOP_N_GRAM_AA_ORDER,
        "matrix_layout": "residue_by_amino_acid",
        "tie_policy": TOP_N_GRAM_TIE_POLICY,
        "samples": entries,
        "provenance": {
            "feature_fit_scope": "train_only",
            "fallback_policy": (
                TOP_N_GRAM_FALLBACK_POLICY_VERSION
                if any(bool(item[3].get("fallback", False)) for item in samples if len(item) > 3)
                else TOP_N_GRAM_STRICT_PROFILE_POLICY
            ),
        },
    }
    index_path = profile_dir / "index.json"
    index_path.write_text(json.dumps(manifest), encoding="utf-8")
    return index_path


def test_top_ngram_max_length_uses_dataset_floor_and_platform_limit(monkeypatch):
    monkeypatch.setenv("PEP_TOP_NGRAM_MAX_RESIDUES", "10000")

    assert configured_top_ngram_max_residues() == 10000
    assert resolve_top_ngram_max_length(60, 207) == (207, True)
    assert resolve_top_ngram_max_length(300, 207) == (300, False)
    with pytest.raises(
        TopNGramContractError,
        match="platform",
    ) as exc_info:
        resolve_top_ngram_max_length(60, 10001)
    assert exc_info.value.code == "top_ngram_sequence_limit_exceeded"

    monkeypatch.setenv("PEP_TOP_NGRAM_MAX_RESIDUES", "10")
    with pytest.raises(TopNGramContractError) as normalize_error:
        normalize_top_ngram_sequence("A", max_length=11)
    assert normalize_error.value.code == "top_ngram_sequence_limit_exceeded"


def test_blosum62_target_frequency_prior_is_normalized_and_deterministic():
    matrix = blosum62_target_frequency_profile("ARX")

    assert matrix.shape == (3, 20)
    assert matrix.dtype == np.float32
    assert np.allclose(matrix.sum(axis=1), 1.0)
    # X uses the standard extended BLOSUM62 X row, not an arbitrary
    # leucine/background surrogate.
    assert matrix.argmax(axis=1).tolist() == [0, 1, 0]
    assert np.array_equal(matrix, blosum62_target_frequency_profile("ARX"))


@pytest.mark.skipif(
    shutil.which("makeblastdb") is None or shutil.which("psiblast") is None,
    reason="BLAST+ is required for the real zero-hit integration test",
)
def test_real_psiblast_zero_hit_build_uses_explicit_frequency_fallback(tmp_path):
    data_dir = tmp_path / "data"
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True)
    sequence = "KLLSTGLVQNFPNTIISK"
    (train_dir / "seqs.fasta").write_text(
        f">Train1\n{sequence}\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "profiles"

    subprocess.run(
        [
            sys.executable,
            "tools/build_sequence_profile_cache.py",
            "--data-dir",
            str(data_dir),
            "--out-root",
            str(output_dir),
            "--max-length",
            str(len(sequence)),
            "--iterations",
            "3",
            "--evalue",
            "0.001",
            "--threads",
            "1",
            "--feature",
            "target_frequency",
            "--on-missing",
            TOP_N_GRAM_FALLBACK_POLICY,
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    index_path = output_dir / "target_frequency" / "index.json"
    manifest = json.loads(index_path.read_text(encoding="utf-8"))
    entry = manifest["samples"]["Train1"]
    assert manifest["schema"] == PROFILE_SCHEMA
    assert manifest["contract_version"] == PROFILE_CONTRACT_VERSION
    assert manifest["provenance"]["fallback_policy"] == TOP_N_GRAM_FALLBACK_POLICY
    assert entry["fallback"] is True
    assert entry["profile_source"] == TOP_N_GRAM_PROFILE_SOURCE_FALLBACK
    assert entry["fallback_reason"] in {
        "profile_output_missing",
        "profile_frequency_unavailable",
    }
    store = TargetFrequencyProfileStore(
        index_path,
        max_length=len(sequence),
    )
    with pytest.warns(UserWarning, match="BLOSUM62"):
        loaded = store.load("Train1", sequence)
    assert loaded.shape == (len(sequence), 20)


def test_pair_profile_role_collection_is_role_aware():
    from tools.build_ppi_profile_cache import collect_role_sequences

    rows = [("sample", "PEPTIDE", "PARTNER")]

    assert collect_role_sequences(rows, query_role="peptide") == {"sample::pep": "PEPTIDE"}
    assert collect_role_sequences(rows, query_role="partner") == {"sample::pro": "PARTNER"}


def test_pair_top_ngram_lengths_cover_all_rows_for_each_stream():
    heterogeneous = SimpleNamespace(
        stream_mode="heterogeneous",
        max_length=60,
        streams={
            "peptide": {"tokenizer": "top-n-gram", "max_length": 8},
            "partner": {"tokenizer": "top-n-gram", "max_length": 60},
        },
    )
    rows = [
        ("train", "PEPTIDE", "PARTNER", 1),
        ("test", "P" * 12, "R" * 207, 0),
    ]

    resolve_pair_top_ngram_lengths(heterogeneous, rows)

    assert heterogeneous.peptide_max_length == 12
    assert heterogeneous.partner_max_length == 207
    assert heterogeneous.streams["peptide"]["max_length"] == 12
    assert heterogeneous.streams["partner"]["max_length"] == 207

    homogeneous = SimpleNamespace(
        stream_mode="homogeneous",
        max_length=60,
        streams={
            "peptide": {"tokenizer": "top-n-gram", "max_length": 60},
            "partner": {"tokenizer": "top-n-gram", "max_length": 60},
        },
    )
    resolve_pair_top_ngram_lengths(homogeneous, rows)

    assert homogeneous.max_length == 207
    assert homogeneous.peptide_max_length == 207
    assert homogeneous.partner_max_length == 207


def test_pair_feature_cache_identity_records_v3_fallback_contract():
    args = SimpleNamespace(
        tokenizer="top-n-gram",
        encoder="bow",
        max_length=207,
        top_ngram_n=2,
        profile_on_missing=TOP_N_GRAM_FALLBACK_POLICY,
        _feature_train_sha256="train-digest",
        _frequency_profile_manifest_sha256="a" * 64,
    )

    fallback_meta = _feature_cache_meta(
        args,
        "train",
        "sequence",
        [("sample", "AR")],
    )
    contract = fallback_meta["top_ngram_contract"]
    assert contract["contract_version"] == TOP_N_GRAM_CONTRACT_VERSION
    assert contract["profile_type"] == TOP_N_GRAM_PROFILE_FEATURE_TYPE
    assert contract["full_length"] is True
    assert contract["fallback_profiles_allowed"] is True
    assert contract["fallback_policy"] == TOP_N_GRAM_FALLBACK_POLICY_VERSION
    assert "v3" in classic_feature_cache_tag(args, "bow")

    args.profile_on_missing = TOP_N_GRAM_STRICT_PROFILE_POLICY
    strict_meta = _feature_cache_meta(
        args,
        "train",
        "sequence",
        [("sample", "AR")],
    )
    strict_contract = strict_meta["top_ngram_contract"]
    assert strict_contract["fallback_profiles_allowed"] is False
    assert strict_contract["fallback_policy"] == TOP_N_GRAM_STRICT_PROFILE_POLICY
    assert strict_meta != fallback_meta


def _ratio_tuple(value):
    return f"{{ {int(value)}, 10, 0 }}"


def _write_checkpoint(path: Path, matrix, *, by_row):
    matrix = np.asarray(matrix, dtype=np.int64)
    assert matrix.shape[1] == 28
    values = matrix.T.reshape(-1) if by_row else matrix.reshape(-1)
    ratios = ",\n        ".join(_ratio_tuple(value) for value in values)
    path.write_text(
        "\n".join(
            [
                "PssmWithParameters ::= {",
                "  pssm {",
                "    isProtein TRUE,",
                "    numRows 28,",
                f"    numColumns {matrix.shape[0]},",
                f"    byRow {'TRUE' if by_row else 'FALSE'},",
                "    intermediateData {",
                "      freqRatios {",
                f"        {ratios}",
                "      }",
                "    }",
                "  }",
                "}",
            ]
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("by_row", [False, True])
def test_checkpoint_parser_recovers_target_frequencies_for_both_layouts(
    tmp_path,
    by_row,
):
    ratios = np.ones((2, 28), dtype=np.int64)
    ratios[0, NCBISTDAA_CODES[1]] = 100
    ratios[1, NCBISTDAA_CODES[19]] = 100
    checkpoint = tmp_path / "profile.chk"
    _write_checkpoint(checkpoint, ratios, by_row=by_row)

    matrix = parse_psiblast_target_frequencies(checkpoint)

    assert matrix.shape == (2, 20)
    assert matrix.argmax(axis=1).tolist() == [1, 19]
    assert np.allclose(matrix.sum(axis=1), 1.0)


def test_checkpoint_parser_does_not_reapply_background_frequencies(tmp_path):
    target_values = np.ones((1, 28), dtype=np.int64)
    target_values[0, NCBISTDAA_CODES[0]] = 2
    target_values[0, NCBISTDAA_CODES[1]] = 3
    checkpoint = tmp_path / "profile.chk"
    _write_checkpoint(checkpoint, target_values, by_row=False)

    matrix = parse_psiblast_target_frequencies(checkpoint)

    assert matrix.argmax(axis=1).tolist() == [1]


def test_tokenizer_uses_full_true_length_stable_ties_and_exact_feature_ids(
    tmp_path,
):
    tied = np.full((2, 20), 0.05, dtype=np.float32)
    index_path = _write_profile_manifest(
        tmp_path,
        [("sample", "AR", tied)],
    )
    tokenizer = TopNGramTokenizer(
        index_path=index_path,
        max_length=2,
        top_n=3,
    )

    assert tokenizer.tokenize("AR", sample_id="sample") == ["ARN", "ARN"]
    assert tokenizer.feature_ids("AR", sample_id="sample").tolist() == [22, 22]
    assert tokenizer.feature_dim == 20**3
    assert tokenizer.vocab_size == 20**3
    assert tokenizer.to_indices(["AAA", "ARN"]) == [0, 22]


def test_count_encoder_counts_feature_zero_instead_of_treating_it_as_padding(
    tmp_path,
):
    matrix = np.full((2, 20), 0.01 / 19, dtype=np.float32)
    matrix[:, 0] = 0.99
    matrix /= matrix.sum(axis=1, keepdims=True)
    index_path = _write_profile_manifest(
        tmp_path,
        [("sample", "AA", matrix)],
    )
    tokenizer = TopNGramTokenizer(index_path, max_length=2, top_n=1)
    encoded = TopNGramCountEncoder(tokenizer).encode_sequence(
        "AA",
        sample_id="sample",
    )

    assert encoded.shape == (20,)
    assert encoded[0].item() == 2.0
    assert encoded.sum().item() == 2.0


def test_top_ngram_bow_v2_ignores_ordered_max_length(tmp_path):
    matrix = np.full((3, 20), 0.01 / 19, dtype=np.float32)
    matrix[:, 0] = 0.99
    matrix /= matrix.sum(axis=1, keepdims=True)
    index_path = _write_profile_manifest(
        tmp_path,
        [("sample", "AAA", matrix)],
    )
    featurizer = FeaturizerFactory(
        bpe_path="",
        frequency_profile_index_path=str(index_path),
    ).create(
        "top-n-gram",
        "bow",
        max_length=1,
        k=1,
        top_ngram_n=1,
    )

    encoded = featurizer.encode("AAA", sample_id="sample")

    assert encoded[0].item() == 3.0
    assert encoded.sum().item() == 3.0


def test_tfidf_factory_fits_train_only_profile_documents(tmp_path):
    first = np.full((2, 20), 0.01 / 19, dtype=np.float32)
    first[:, 0] = 0.99
    first /= first.sum(axis=1, keepdims=True)
    second = np.full((2, 20), 0.01 / 19, dtype=np.float32)
    second[:, 1] = 0.99
    second /= second.sum(axis=1, keepdims=True)
    index_path = _write_profile_manifest(
        tmp_path,
        [
            ("first", "AA", first),
            ("second", "RR", second),
        ],
    )
    factory = FeaturizerFactory(
        bpe_path="",
        frequency_profile_index_path=str(index_path),
    )

    featurizer = factory.create(
        "top-n-gram",
        "tfidf",
        2,
        1,
        top_ngram_n=1,
        profile_samples=[("first", "AA"), ("second", "RR")],
    )

    expected_seen_idf = torch.log(torch.tensor(3.0 / 2.0)) + 1.0
    assert torch.isclose(featurizer.encoder.idf[0], expected_seen_idf)
    assert torch.isclose(featurizer.encoder.idf[1], expected_seen_idf)
    assert torch.isclose(
        featurizer.encoder.idf[2],
        torch.log(torch.tensor(3.0)) + 1.0,
    )
    encoded = featurizer.encode("AA", sample_id="first")
    assert torch.isclose(encoded[0], expected_seen_idf)
    assert torch.count_nonzero(encoded).item() == 1


def test_textrank_uses_exact_feature_dimension_and_full_stream(tmp_path):
    matrix = np.full((3, 20), 0.01 / 19, dtype=np.float32)
    matrix[0, 0] = 0.99
    matrix[1, 1] = 0.99
    matrix[2, 0] = 0.99
    matrix /= matrix.sum(axis=1, keepdims=True)
    index_path = _write_profile_manifest(
        tmp_path,
        [("sample", "ARA", matrix)],
    )
    tokenizer = TopNGramTokenizer(index_path, max_length=3, top_n=1)
    encoded = TopNGramTextRankEncoder(tokenizer).encode_sequence(
        "ARA",
        sample_id="sample",
    )

    assert encoded.shape == (20,)
    assert torch.isclose(encoded.sum(), torch.tensor(1.0))
    assert torch.count_nonzero(encoded).item() == 2


def test_profile_store_rejects_legacy_index_and_sequence_mismatch(tmp_path):
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps({"sample": "sample.pt"}), encoding="utf-8")
    with pytest.raises(TopNGramContractError, match="legacy"):
        TargetFrequencyProfileStore(legacy_path, max_length=2)

    matrix = np.full((2, 20), 0.05, dtype=np.float32)
    index_path = _write_profile_manifest(
        tmp_path / "v2",
        [("sample", "AR", matrix)],
    )
    store = TargetFrequencyProfileStore(index_path, max_length=2)
    with pytest.raises(
        TopNGramContractError,
        match="digest does not match",
    ):
        store.load("sample", "RA")


def test_profile_store_accepts_authorized_fallback_and_rejects_invalid_values(
    tmp_path,
):
    matrix = np.full((1, 20), 0.05, dtype=np.float32)
    index_path = _write_profile_manifest(
        tmp_path,
        [("sample", "A", matrix, {"fallback": True})],
    )
    store = TargetFrequencyProfileStore(index_path, max_length=1)
    with pytest.warns(UserWarning, match="BLOSUM62"):
        assert store.load("sample", "A").shape == (1, 20)

    manifest = json.loads(index_path.read_text(encoding="utf-8"))
    manifest["provenance"]["fallback_policy"] = TOP_N_GRAM_STRICT_PROFILE_POLICY
    index_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TopNGramContractError, match="not authorized"):
        TargetFrequencyProfileStore(index_path, max_length=1)

    invalid = matrix.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(TopNGramContractError, match="non-finite"):
        save_target_frequency_matrix(tmp_path / "invalid.npz", invalid)


def test_profile_store_rejects_tie_policy_and_matrix_digest_tampering(tmp_path):
    matrix = np.full((1, 20), 0.05, dtype=np.float32)
    index_path = _write_profile_manifest(
        tmp_path,
        [("sample", "A", matrix)],
    )
    manifest = json.loads(index_path.read_text(encoding="utf-8"))
    manifest["tie_policy"] = "unstable"
    index_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TopNGramContractError, match="tie_policy"):
        TargetFrequencyProfileStore(index_path, max_length=1)

    index_path = _write_profile_manifest(
        tmp_path / "digest",
        [("sample", "A", matrix)],
    )
    matrix_path = index_path.parent / "sample.npz"
    replacement = matrix.copy()
    replacement[0, 0] = 0.1
    replacement[0, 1:] = 0.9 / 19
    save_target_frequency_matrix(matrix_path, replacement)
    store = TargetFrequencyProfileStore(index_path, max_length=1)
    with pytest.raises(TopNGramContractError, match="digest"):
        store.load("sample", "A")


def test_profile_store_rejects_non_float32_archive(tmp_path):
    matrix = np.full((1, 20), 0.05, dtype=np.float32)
    index_path = _write_profile_manifest(
        tmp_path,
        [("sample", "A", matrix)],
    )
    matrix_path = index_path.parent / "sample.npz"
    np.savez_compressed(matrix_path, matrix=matrix.astype(np.float64))
    manifest = json.loads(index_path.read_text(encoding="utf-8"))
    manifest["samples"]["sample"]["matrix_sha256"] = sha256_file(matrix_path)
    index_path.write_text(json.dumps(manifest), encoding="utf-8")

    store = TargetFrequencyProfileStore(index_path, max_length=1)
    with pytest.raises(TopNGramContractError, match="float32"):
        store.load("sample", "A")


@pytest.mark.parametrize("value", [True, 0, 4, 2.5, "2"])
def test_top_n_validation_rejects_values_outside_v3_contract(value):
    with pytest.raises(TopNGramContractError):
        validate_top_ngram_n(value)


def test_feature_dimensions_are_exact_without_special_slots():
    assert [top_ngram_feature_dim(value) for value in (1, 2, 3)] == [
        20,
        400,
        8000,
    ]


def test_web_inference_rejects_legacy_top_ngram_artifact():
    metadata = {
        "classic": {
            "tokenizer": "top-n-gram",
            "top_ngram_n": 2,
        }
    }
    with pytest.raises(
        InferenceUnsupportedError,
        match="must be retrained",
    ):
        ModelService._validate_top_ngram_bundle_contract(metadata)


def test_web_inference_rejects_profile_bundle_without_v2_provenance(
    tmp_path,
    monkeypatch,
):
    profile_dir = tmp_path / "profile_generation"
    profile_dir.mkdir()
    subject = profile_dir / "subject.fasta"
    subject.write_text(">SUBJ_0\nAR\n", encoding="utf-8")
    metadata = {
        "classic": {
            "tokenizer": "top-n-gram",
            "encoder": "bow",
            "max_length": 2,
            "top_ngram_n": 1,
            "top_ngram_contract": {
                "contract_version": TOP_N_GRAM_CONTRACT_VERSION,
                "profile_type": "target_frequency",
                "tie_policy": TOP_N_GRAM_TIE_POLICY,
                "full_length": True,
                "fallback_profiles_allowed": True,
                "fallback_policy": TOP_N_GRAM_FALLBACK_POLICY_VERSION,
                "feature_dim": 20,
                "profile_manifest_sha256": "a" * 64,
            },
        },
        "profile_generation": {
            "feature": "target_frequency",
            "subject_fasta": "profile_generation/subject.fasta",
            "iterations": 3,
            "evalue": 0.001,
            "threads": 1,
            "on_missing": TOP_N_GRAM_FALLBACK_POLICY,
        },
    }
    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._build_inference_profile_cache(
            tmp_path,
            metadata,
            {"inference_0": "AR"},
        )
    assert exc_info.value.code == "profile_contract_legacy"
