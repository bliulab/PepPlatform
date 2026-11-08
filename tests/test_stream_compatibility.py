"""Compatibility boundaries must preserve frozen inputs and numerical behavior."""

from argparse import Namespace
from copy import deepcopy
import json

import pytest
import torch

from pepplatform import pair_strategy as current
from pepplatform._legacy_stream import pair_strategy as historical
from pepplatform.pair_terminology import normalize_pair_strategy
from pepplatform.stream_compat import normalize_stream_fields
from pepplatform.web.config import normalize_experiment_config


@pytest.mark.parametrize("old,new", [
    ("single_tower", "single_stream"), ("single-tower", "single_stream"),
    ("dual_tower", "two_stream"), ("dual-tower", "two_stream"),
    ("single-stream", "single_stream"), ("two-stream", "two_stream"),
])
def test_old_and_new_inputs_resolve_to_canonical_strategy(old, new):
    assert normalize_pair_strategy(old) == new
    config = dict(tokenizer="kmer", encoder="index", model="cnn", pair_strategy=old)
    a = normalize_experiment_config("pep_hlai", config)
    b = normalize_experiment_config("pep_hlai", {**config, "pair_strategy": new})
    assert a == b
    assert a["pair_strategy"] == new
    assert "streams" in a and "towers" not in a


def test_configuration_aliases_preserve_originals_and_reject_conflicts():
    raw = {"pair_strategy": "dual_tower", "towers": {"peptide": {"tower_model": "identity"}},
           "candidate": "c008_single_dual_tower_prott5_esm2_b16",
           "route": "pep_hlai/c008_single_dual_tower_prott5_esm2_b16",
           "source_files": {"pepplatform/tower_spec.py": "abc"},
           "pair_strategy_contract": {"contract_version": 1, "strategy": "dual_tower"}}
    original = deepcopy(raw)
    view = normalize_stream_fields(raw)
    assert view["streams"]["peptide"]["stream_model"] == "identity"
    assert view["candidate"] == raw["candidate"]
    assert view["route"] == raw["route"]
    assert view["source_files"] == raw["source_files"]
    assert view["pair_strategy_contract"] == raw["pair_strategy_contract"]
    assert raw == original
    assert normalize_stream_fields({"tower_alignment_dim": 32, "stream_alignment_dim": 32}) == {"stream_alignment_dim": 32}
    with pytest.raises(ValueError, match="Conflicting"):
        normalize_stream_fields({"towers": {"peptide": 1}, "streams": {"peptide": 2}})


@pytest.mark.parametrize("strategy", ["single", "two"])
def test_frozen_strategy_protocol_is_validated_before_conversion(strategy):
    peptide = torch.tensor([[1, 2, 0], [3, 4, 5]])
    partner = torch.tensor([[2, 3, 0], [4, 5, 6]])
    if strategy == "single":
        joined = historical.join_single_tower_batches(peptide, partner, separator_id=9)
        contract = historical.build_single_tower_strategy_contract(
            joined, separator_id=9, source_feature_dim=None)
        validator = current.validate_single_stream_strategy_contract
    else:
        args = Namespace(pair_strategy="dual_tower", interaction_level="token",
                         predictor_type="classifier", interaction_model="cross_attn")
        contract = historical.build_dual_tower_strategy_contract(args, peptide, partner)
        validator = current.validate_two_stream_strategy_contract
    original = json.dumps(contract, sort_keys=True)
    normalized = validator(contract)
    assert normalized["contract_version"] == 2
    assert normalized["strategy"] == ("single_stream" if strategy == "single" else "two_stream")
    assert json.dumps(contract, sort_keys=True) == original
    damaged = deepcopy(contract)
    damaged["role_order"] = ["partner", "peptide"]
    with pytest.raises(current.PairStrategyContractError):
        validator(damaged)


@pytest.mark.parametrize("kind", ["index", "feature"])
def test_old_and_new_join_are_numerically_identical(kind):
    torch.manual_seed(3103)
    if kind == "index":
        peptide = torch.tensor([[1, 2, 0], [3, 4, 5]])
        partner = torch.tensor([[2, 0], [4, 5]])
        kwargs = {"separator_id": 9}
    else:
        peptide = torch.randn(2, 3, 4)
        partner = torch.randn(2, 2, 4)
        kwargs = {}
    before = torch.random.get_rng_state().clone()
    old = historical.join_single_tower_batches(peptide, partner, **kwargs)
    assert torch.equal(before, torch.random.get_rng_state())
    new = current.join_single_stream_batches(peptide, partner, **kwargs)
    torch.testing.assert_close(old, new, rtol=0, atol=0)
    assert torch.equal(before, torch.random.get_rng_state())


def test_old_cli_flags_work_without_leaking_old_fields():
    from scripts.train_pair_pipeline import build_argument_parser

    parser = build_argument_parser()
    args = parser.parse_args(["--pair_strategy", "dual-tower", "--tower_alignment_dim", "32",
                              "--peptide_tower_model", "identity"])
    assert args.pair_strategy == "two_stream"
    assert args.stream_alignment_dim == 32
    assert args.peptide_stream_model == "identity"
    assert not hasattr(args, "tower_alignment_dim")
    with pytest.raises(SystemExit):
        parser.parse_args(["--tower_alignment_dim", "32", "--stream_alignment_dim", "64"])
    args = parser.parse_args(["--tower_alignment_dim", "32", "--stream_alignment_dim", "32"])
    assert args.stream_alignment_dim == 32


def test_old_python_imports_remain_readable():
    from pepplatform.tower_spec import normalize_pair_towers
    from pepplatform.workflow import parse_tower_alignment_dim
    from scripts.train_pair_pipeline import CompactSingleTowerEncoding
    from scripts.train_pair_pipeline import CompactSingleStreamEncoding

    assert parse_tower_alignment_dim("32") == 32
    assert CompactSingleTowerEncoding is CompactSingleStreamEncoding
    spec = normalize_pair_towers({"task_type": "pep_hlai", "tokenizer": "kmer", "encoder": "onehot"})
    assert "tower_model" in spec["peptide"]
    assert "stream_model" not in spec["peptide"]


def test_legacy_token_contract_restores_each_role_dtype_without_losing_masks():
    from pepplatform.models import ResidueModelInput
    from pepplatform.web.inference import ModelService

    peptide = torch.ones(2, 3, 4, dtype=torch.float32)
    partner = ResidueModelInput(torch.ones(2, 5, 6, dtype=torch.float16),
                               torch.ones(2, 5, dtype=torch.bool))
    args = Namespace(pair_strategy="dual_tower", interaction_level="token",
                     predictor_type="classifier", interaction_model="bi_cross_attn")
    contract = historical.build_dual_tower_strategy_contract(args, peptide, partner)
    original = deepcopy(contract)
    restored_peptide, restored_partner = ModelService._cast_pair_tokens_to_contract_dtype(
        {"pair_strategy_contract": contract}, peptide, ResidueModelInput(partner.features.float(), partner.mask))
    assert restored_peptide.dtype == torch.float32
    assert restored_partner.features.dtype == torch.float16
    assert restored_partner.mask is partner.mask
    current.validate_two_stream_strategy_contract(
        contract, peptide=restored_peptide, partner=restored_partner)
    assert contract == original


def test_legacy_attention_description_keeps_model_identity_and_validation():
    from pepplatform.web.inference import (
        _validate_loaded_pair_attention_model, InferenceUnsupportedError,
    )
    from scripts.train_pair_pipeline import (
        HeterogeneousPairBiCrossAttentionClassifier, pair_attention_predictor_contract,
    )
    model = HeterogeneousPairBiCrossAttentionClassifier(
        peptide_encoder_name="onehot", partner_encoder_name="esm2",
        peptide_in_dim=4, partner_in_dim=6, hidden_dim=8, n_heads=2, dropout=0.0)
    contract = pair_attention_predictor_contract(model)
    contract["masking"] = "opposite-tower key padding masks and both query masked-mean readouts"
    original = deepcopy(contract)
    _validate_loaded_pair_attention_model(model, contract, "bi_cross_attn")
    assert contract == original
    contract["masking"] = "different masking"
    with pytest.raises(InferenceUnsupportedError):
        _validate_loaded_pair_attention_model(model, contract, "bi_cross_attn")



def test_new_cache_namespace_does_not_replace_historical_files(tmp_path):
    from pepplatform.stream_compat import stream_cache_root
    original = tmp_path / "train_peptide.pt"
    original.write_bytes(b"frozen historical cache")
    new = stream_cache_root(tmp_path)
    assert new == str(tmp_path / "stream_v2")
    assert stream_cache_root(new) == new
    assert original.read_bytes() == b"frozen historical cache"


def test_legacy_pair_bow_protocol_keeps_its_digest_and_rejects_tampering():
    from pepplatform._legacy_stream.pair_bow import build_pair_bow_joint_contract as old_build
    from pepplatform.pair_bow import validate_pair_bow_joint_contract, PairBoWContractError

    def role(width, scope):
        return {"spec": {"tokenizer": "kmer", "encoder": "bow"},
                "frequency_encoder_contract": {
                    "contract_version": 2, "encoder": "bow",
                    "stream_policy": "full_token_stream", "bow_formula": "x_j = c_j",
                    "post_normalization": "none", "frequency_max_raw_length": 50000,
                    "fit_scope": scope, "feature_dimension": width,
                    "training_document_count": 4,
                    "vocabulary_sha256": "a" * 64, "training_corpus_sha256": "b" * 64}}
    contract = old_build(role(3, "peptide_role"), role(2, "partner_role"),
                         train_sample_count=4, ordered_pair_train_sha256="c" * 64,
                         fused_feature_dimension=5)
    original = json.dumps(contract, sort_keys=True)
    assert validate_pair_bow_joint_contract(contract) == contract
    assert json.dumps(contract, sort_keys=True) == original
    damaged = deepcopy(contract)
    damaged["role_contract_sha256"]["peptide"] = "d" * 64
    with pytest.raises(PairBoWContractError):
        validate_pair_bow_joint_contract(damaged)


def test_new_database_config_is_canonical_and_historical_reads_are_not_rewrites(tmp_path):
    from sqlalchemy import text
    from pepplatform.web.store import MetadataStore

    store = MetadataStore(tmp_path / "metadata.sqlite3")
    dataset = store.create_dataset("pairs", "pep_hlai", "upload", "pair_csv_bundle", tmp_path)
    experiment = store.create_experiment("test", dataset["id"], "pep_hlai",
        {"pair_strategy": "dual_tower", "tower_alignment_dim": 32},
        tmp_path / "log.txt", tmp_path / "artifact")
    assert experiment["config"] == {"pair_strategy": "two_stream", "stream_alignment_dim": 32}
    original = json.dumps({"pair_strategy": "single_tower", "tower_alignment_dim": 32})
    with store.connect() as conn:
        conn.execute(text("UPDATE experiments SET config_json = :value WHERE id = :id"),
                     {"value": original, "id": experiment["id"]})
    assert store.get_experiment(experiment["id"])["config"]["pair_strategy"] == "single_stream"
    with store.connect() as conn:
        saved = conn.execute(text("SELECT config_json FROM experiments WHERE id = :id"),
                             {"id": experiment["id"]}).scalar_one()
    assert saved == original
