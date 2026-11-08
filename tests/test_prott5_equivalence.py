"""Validate cross-format comparisons without downloading or loading a big model."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

spec = importlib.util.spec_from_file_location(
    "verify_prott5", Path(__file__).resolve().parents[1] / "tools/verify_prott5_equivalence.py"
)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def checkpoints(tmp_path, *, modification=None):
    shared = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    state = {
        "shared.weight": shared,
        "encoder.embed_tokens.weight": shared,
        "decoder.embed_tokens.weight": shared,
        "lm_head.weight": shared,
        "encoder.block.0.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "decoder.block.0.weight": torch.arange(6, dtype=torch.float32).reshape(3, 2),
    }
    if modification == "bad_alias":
        state["decoder.embed_tokens.weight"] = shared + 1
    if modification == "bad_encoder_alias":
        state["encoder.embed_tokens.weight"] = shared + 1
    official = tmp_path / "pytorch_model.bin"
    torch.save(state, official)
    shards = tmp_path / "shards"
    shards.mkdir()
    saved = {k: v.clone() for k, v in state.items() if k not in verify.ALIASES}
    if modification == "decoder":
        saved["decoder.block.0.weight"][0, 0] += 1
    if modification == "dtype":
        saved["encoder.block.0.weight"] = saved["encoder.block.0.weight"].double()
    if modification == "missing_decoder":
        saved.pop("decoder.block.0.weight")
    save_file(saved, shards / "model-00001-of-00001.safetensors")
    (shards / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in saved}})
    )
    return official, shards


def test_shared_alias_deduplication_is_lossless_and_inputs_are_read_only(tmp_path):
    official, shards = checkpoints(tmp_path)
    before = {p: verify.file_sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    result = verify.compare_state(official, shards, progress=lambda _: None)
    assert result["full_state_bitwise_equal"]
    assert result["encoder_state_bitwise_equal"]
    assert result["official_raw_tensor_count"] == 6
    assert result["sharded_raw_tensor_count"] == 3
    assert result["canonical_tensor_count"] == 3
    assert len(result["aliases"]) == 3
    assert result["canonical_fingerprints"]["left"] == result["canonical_fingerprints"]["right"]
    assert before == {p: verify.file_sha(p) for p in before}


@pytest.mark.parametrize(
    "modification", ["decoder", "dtype", "missing_decoder", "bad_alias", "bad_encoder_alias"]
)
def test_parameter_or_alias_changes_cannot_be_accepted(tmp_path, modification):
    official, shards = checkpoints(tmp_path, modification=modification)
    result = verify.compare_state(official, shards, progress=lambda _: None)
    assert not result["full_state_bitwise_equal"]
    if modification in {"decoder", "missing_decoder", "bad_alias"}:
        assert result["encoder_state_bitwise_equal"]
    else:
        assert not result["encoder_state_bitwise_equal"]


def test_bit_comparison_detects_signed_zero_even_when_numerically_equal():
    result = verify.compare_tensor(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert not result["bitwise_equal"]
    assert result["different_values"] == 0
    assert result["max_absolute_error"] == 0


def test_output_masks_and_dtype_are_compared():
    left = {"pooled": torch.zeros((1, 3)), "mask": torch.tensor([[True, False]])}
    right = {"pooled": left["pooled"].clone(), "mask": left["mask"].clone()}
    assert verify.compare_outputs(left, right)["bitwise_equal"]
    right["mask"][0, 1] = True
    assert not verify.compare_outputs(left, right)["bitwise_equal"]


def test_dataset_identity_accepts_only_exact_registered_content(tmp_path):
    relocated = tmp_path / "pairs.csv"
    relocated.write_text("peptide,label\nACDE,1\n")
    expected = verify.file_sha(relocated)
    declared = {"/frozen/validation.csv": expected}
    assert verify.validate_dataset(relocated, declared)["sha256"] == expected
    relocated.write_text("peptide,label\nACDF,1\n")
    with pytest.raises(ValueError, match="not registered"):
        verify.validate_dataset(relocated, declared)


def test_configuration_metadata_does_not_mask_computational_changes(tmp_path):
    from transformers import T5Config

    official, sharded = tmp_path / "official", tmp_path / "sharded"
    official.mkdir()
    sharded.mkdir()
    config = T5Config(d_model=16, d_ff=32, num_layers=1, num_heads=2).to_dict()
    config["architectures"] = ["T5ForConditionalGeneration"]
    (official / "config.json").write_text(json.dumps(config))
    config["architectures"] = ["T5Model"]
    config["transformers_version"] = "different serialization version"
    (sharded / "config.json").write_text(json.dumps(config))
    comparison = verify.compare_config(official, sharded)
    assert comparison["effective_equal"]
    assert "architectures" in comparison["raw_differences"]
    config["layer_norm_epsilon"] = 1e-3
    (sharded / "config.json").write_text(json.dumps(config))
    assert not verify.compare_config(official, sharded)["effective_equal"]
