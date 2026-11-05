import json
from argparse import Namespace

import numpy as np
import pytest
import torch

from pepplatform.dt import (
    DT_ALPHABET,
    DT_CONTRACT_VERSION,
    DTContractError,
    dt_contract_metadata,
    dt_feature_dim,
    dt_feature_index,
    ensure_dt_dense_memory_budget,
    estimate_dt_dense_bytes,
    parse_dt_max_distance,
    validate_dt_max_distance,
)
from pepplatform.encoders import DTCountEncoder, DTTextRankEncoder
from pepplatform.featurizers import FeaturizerFactory
from pepplatform.tokenizers import DTTokenizer
from pepplatform.top_ngram import (
    TOP_N_GRAM_AA_ORDER,
    TOP_N_GRAM_CONTRACT_VERSION,
    TOP_N_GRAM_PROFILE_FEATURE_TYPE,
    TOP_N_GRAM_PROFILE_SOURCE_PSI_BLAST,
    TOP_N_GRAM_STRICT_PROFILE_POLICY,
    TOP_N_GRAM_TIE_POLICY,
    save_target_frequency_matrix,
    sequence_sha256,
    sha256_file,
)
from pepplatform.stream_spec import normalize_pair_streams
from pepplatform.web.inference import InferenceUnsupportedError, ModelService
from scripts.train_pair_pipeline import (
    _feature_cache_meta,
    _pair_profile_feature_kind,
    _pair_profile_generation_needed,
    _stream_artifact_metadata,
    classic_feature_cache_tag,
)


def _write_profile_manifest(tmp_path, samples):
    profile_dir = tmp_path / "target_frequency"
    profile_dir.mkdir(parents=True)
    entries = {}
    for sample_id, sequence, matrix in samples:
        path = profile_dir / f"{sample_id}.npz"
        save_target_frequency_matrix(path, matrix)
        entries[sample_id] = {
            "path": path.name,
            "matrix_sha256": sha256_file(path),
            "sequence_length": len(sequence),
            "sequence_sha256": sequence_sha256(sequence),
            "zero_nonself_hits": False,
            "fallback": False,
            "profile_source": TOP_N_GRAM_PROFILE_SOURCE_PSI_BLAST,
            "fallback_reason": "",
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
            "fallback_policy": TOP_N_GRAM_STRICT_PROFILE_POLICY,
        },
    }
    index_path = profile_dir / "index.json"
    index_path.write_text(json.dumps(manifest), encoding="utf-8")
    return index_path


def _profile_for_top1(sequence):
    matrix = np.full((len(sequence), 20), 0.01 / 19, dtype=np.float32)
    for position, residue in enumerate(sequence):
        matrix[position, TOP_N_GRAM_AA_ORDER.index(residue)] = 0.99
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix


def test_dt_v1_matches_paper_example_and_exact_feature_map(tmp_path):
    index_path = _write_profile_manifest(
        tmp_path,
        [("paper", "AGLP", _profile_for_top1("KFFK"))],
    )
    tokenizer = DTTokenizer(index_path, max_length=4, max_distance=2)

    assert "".join(DT_ALPHABET) == "ARDCQEHIGNLKMFPSTWYV"
    assert tokenizer.feature_dim == dt_feature_dim(2) == 820
    assert tokenizer.tokenize("AGLP", sample_id="paper") == [
        "K",
        "K_1_F",
        "K_2_F",
        "F",
        "F_1_F",
        "F_2_K",
        "F",
        "F_1_K",
        "K",
    ]

    encoded = DTCountEncoder(tokenizer).encode_sequence("AGLP", sample_id="paper")
    assert encoded.shape == (820,)
    assert encoded.sum().item() == 9.0
    assert encoded[dt_feature_index("K", max_distance=2)].item() == 2.0
    assert encoded[dt_feature_index("K_1_F", max_distance=2)].item() == 1.0
    assert encoded[dt_feature_index("F_1_F", max_distance=2)].item() == 1.0
    assert encoded[dt_feature_index("K_2_F", max_distance=2)].item() == 1.0
    assert encoded[dt_feature_index("F_2_K", max_distance=2)].item() == 1.0


def test_dt_distance_zero_is_top1_occurrence_vector(tmp_path):
    index_path = _write_profile_manifest(
        tmp_path,
        [("zero", "ACD", _profile_for_top1("KFK"))],
    )
    tokenizer = DTTokenizer(index_path, max_length=3, max_distance=0)
    encoded = DTCountEncoder(tokenizer).encode_sequence("ACD", sample_id="zero")

    assert tokenizer.feature_dim == 20
    assert tokenizer.tokenize("ACD", sample_id="zero") == ["K", "F", "K"]
    assert encoded.sum().item() == 3.0
    assert encoded[dt_feature_index("K", max_distance=0)].item() == 2.0


def test_dt_uses_full_profile_without_token_limit_or_padding(tmp_path):
    index_path = _write_profile_manifest(
        tmp_path,
        [("full", "AAAA", _profile_for_top1("ARDC"))],
    )
    tokenizer = DTTokenizer(index_path, max_length=4, max_distance=3)
    encoded = DTCountEncoder(tokenizer).encode_sequence("AAAA", sample_id="full")

    assert encoded.sum().item() == 4 + 3 + 2 + 1
    assert encoded[dt_feature_index("A_3_C", max_distance=3)].item() == 1.0
    with pytest.raises(DTContractError):
        dt_feature_index("A_4_C", max_distance=3)


def test_dt_bow_v2_ignores_ordered_max_length(tmp_path):
    index_path = _write_profile_manifest(
        tmp_path,
        [("full", "AAAA", _profile_for_top1("ARDC"))],
    )
    featurizer = FeaturizerFactory(
        bpe_path="",
        frequency_profile_index_path=str(index_path),
    ).create(
        "dt",
        "bow",
        max_length=1,
        k=1,
        dt_max_distance=3,
    )

    encoded = featurizer.encode("AAAA", sample_id="full")

    assert encoded.sum().item() == 4 + 3 + 2 + 1
    assert encoded[dt_feature_index("A_3_C", max_distance=3)].item() == 1.0


def test_dt_reuses_stable_target_frequency_ties_and_sequence_binding(tmp_path):
    tied = np.full((2, 20), 0.05, dtype=np.float32)
    index_path = _write_profile_manifest(tmp_path, [("tie", "AR", tied)])
    tokenizer = DTTokenizer(index_path, max_length=2, max_distance=1)

    assert tokenizer.tokenize("AR", sample_id="tie") == ["A", "A_1_A", "A"]
    with pytest.raises(Exception, match="digest does not match"):
        tokenizer.tokenize("RA", sample_id="tie")


def test_dt_tfidf_is_fitted_from_training_profiles_and_persistable(tmp_path):
    index_path = _write_profile_manifest(
        tmp_path,
        [
            ("first", "AA", _profile_for_top1("AA")),
            ("second", "RR", _profile_for_top1("RR")),
        ],
    )
    factory = FeaturizerFactory(
        bpe_path="",
        frequency_profile_index_path=str(index_path),
    )
    featurizer = factory.create(
        "dt",
        "tfidf",
        2,
        1,
        dt_max_distance=1,
        profile_samples=[("first", "AA"), ("second", "RR")],
    )

    expected = torch.log(torch.tensor(3.0 / 2.0)) + 1.0
    assert torch.isclose(
        featurizer.encoder.idf[dt_feature_index("A", max_distance=1)],
        expected,
    )
    assert torch.isclose(
        featurizer.encoder.idf[dt_feature_index("A_1_A", max_distance=1)],
        expected,
    )
    restored = factory.create(
        "dt",
        "tfidf",
        2,
        1,
        dt_max_distance=1,
        persisted_idf=featurizer.encoder.idf,
    )
    assert torch.equal(
        featurizer.encode("AA", sample_id="first"),
        restored.encode("AA", sample_id="first"),
    )


def test_dt_textrank_uses_complete_source_position_stream(tmp_path):
    index_path = _write_profile_manifest(
        tmp_path,
        [("rank", "AAA", _profile_for_top1("ARA"))],
    )
    tokenizer = DTTokenizer(index_path, max_length=3, max_distance=2)
    encoded = DTTextRankEncoder(tokenizer).encode_sequence("AAA", sample_id="rank")

    assert encoded.shape == (820,)
    assert torch.isclose(encoded.sum(), torch.tensor(1.0))
    assert len(tokenizer.tokenize("AAA", sample_id="rank")) == 6
    assert torch.count_nonzero(encoded).item() == 5


@pytest.mark.parametrize("value", [-1, 201, 1.5, "5", True, None])
def test_dt_distance_validation_is_strict(value):
    with pytest.raises(DTContractError):
        validate_dt_max_distance(value)


def test_dt_contract_and_memory_budget_are_exact():
    assert parse_dt_max_distance("5") == 5
    contract = dt_contract_metadata(max_distance=5)
    assert contract["contract_version"] == DT_CONTRACT_VERSION
    assert contract["feature_dim"] == 2020
    assert contract["bow_normalization"] == "none"
    estimated = estimate_dt_dense_bytes(2, max_distance=5)
    assert estimated == 2 * 2020 * 4
    assert ensure_dt_dense_memory_budget(estimated, limit_gib=1) == estimated
    with pytest.raises(DTContractError, match="dt_dense_memory_limit_gib"):
        ensure_dt_dense_memory_budget(estimated, limit_gib=1e-9)


def test_dt_bundle_contract_is_exact_and_rejects_legacy_hash_artifacts():
    contract = dt_contract_metadata(max_distance=5)
    contract.update(
        {
            "profile_manifest_sha256": "a" * 64,
            "fallback_profiles_allowed": False,
            "fallback_policy": TOP_N_GRAM_STRICT_PROFILE_POLICY,
        }
    )
    metadata = {
        "classic": {
            "tokenizer": "dt",
            "encoder": "bow",
            "dt_max_distance": 5,
            "feature_dim": 2020,
            "dt_contract": contract,
        },
        "profile_generation": {"on_missing": "error"},
    }
    ModelService._validate_dt_bundle_contract(metadata)

    with pytest.raises(InferenceUnsupportedError, match="legacy hashed"):
        ModelService._validate_dt_bundle_contract(
            {
                "classic": {
                    "tokenizer": "dt",
                    "dt_max_distance": 5,
                    "dt_hash_size": 200000,
                }
            }
        )


def test_dt_pair_cache_records_exact_contract_and_manifest_identity():
    args = Namespace(
        tokenizer="dt",
        encoder="bow",
        max_length=207,
        dt_max_distance=5,
        dt_dense_memory_limit_gib=4.0,
        profile_on_missing=TOP_N_GRAM_STRICT_PROFILE_POLICY,
        _feature_train_sha256="train-digest",
        _frequency_profile_manifest_sha256="a" * 64,
    )
    metadata = _feature_cache_meta(
        args,
        "train",
        "peptide",
        [("sample", "AR")],
    )
    contract = metadata["dt_contract"]
    assert contract["contract_version"] == DT_CONTRACT_VERSION
    assert contract["top_n"] == 1
    assert contract["feature_dim"] == 2020
    assert contract["profile_manifest_sha256"] == "a" * 64
    tag = classic_feature_cache_tag(args, "bow")
    assert "d5" in tag
    assert "h200000" not in tag


def test_dt_stream_uses_raw_residue_length_and_strict_distance_budget():
    streams = normalize_pair_streams(
        Namespace(
            task_type="pep_protein_interaction",
            tokenizer="dt",
            encoder="bow",
            max_length=60,
            dt_max_distance=0,
            dt_dense_memory_limit_gib=2.5,
        )
    )
    for role in ("peptide", "partner"):
        assert streams[role]["max_length_unit"] == "raw_residues_hard_limit"
        assert streams[role]["params"]["dt_max_distance"] == 0
        assert streams[role]["params"]["dt_dense_memory_limit_gib"] == 2.5


def test_heterogeneous_dt_stream_persists_role_profile_contract():
    args = Namespace(
        tokenizer="esm2",
        encoder="ppi_fusion",
        artifact_dir="",
        profile_on_missing="error",
        streams={
            "peptide": {
                "feature_route": "classic",
                "tokenizer": "dt",
                "encoder": "bow",
                "max_length": 20,
                "params": {
                    "dt_max_distance": 2,
                    "dt_dense_memory_limit_gib": 4.0,
                },
            },
            "partner": {
                "feature_route": "pretrained",
                "tokenizer": "esm2",
                "encoder": "esm2",
                "max_length": 20,
                "params": {},
            },
        },
        _stream_runtime={
            "peptide": {
                "frequency_profile_manifest_sha256": "a" * 64,
                "resolved_params": {"frequency_profile_index_path": "index.json"},
            }
        },
    )

    assert _pair_profile_generation_needed(args) is True
    assert _pair_profile_feature_kind(args) == "target_frequency"
    metadata = _stream_artifact_metadata(
        args,
        "peptide",
        torch.zeros((1, dt_feature_dim(2))),
        "vector",
    )
    params = metadata["spec"]["params"]
    assert params["feature_dim"] == dt_feature_dim(2)
    assert params["dt_contract"]["profile_manifest_sha256"] == "a" * 64
    ModelService._validate_dt_bundle_contract(
        {
            "classic": {
                "tokenizer": "dt",
                **params,
            },
            "profile_generation": {"on_missing": "error"},
        }
    )
