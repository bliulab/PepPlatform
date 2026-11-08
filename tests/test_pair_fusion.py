from argparse import Namespace

import pytest
import torch

from pepplatform.pair_fusion import (
    PairFusionError,
    fuse_pair_embeddings,
    normalize_pair_fusion_mode,
    pair_fusion_contract,
    validate_pair_fusion_contract,
)
from pepplatform.web.inference import InferenceUnsupportedError, ModelService
from scripts.train_pair_pipeline import (
    _pair_vector_artifact_info,
    fuse_embeddings as training_fuse_embeddings,
)


PEPTIDE = torch.tensor([[1, 2], [3, 4]])
PARTNER = torch.tensor([[5, 6], [7, 8]])
EXPECTED = {
    "concat": torch.tensor([[1, 2, 5, 6], [3, 4, 7, 8]], dtype=torch.float32),
    "concat_absdiff": torch.tensor([[1, 2, 5, 6, 4, 4], [3, 4, 7, 8, 4, 4]], dtype=torch.float32),
    "concat_hadamard": torch.tensor(
        [[1, 2, 5, 6, 5, 12], [3, 4, 7, 8, 21, 32]],
        dtype=torch.float32,
    ),
    "concat_all": torch.tensor(
        [
            [1, 2, 5, 6, 4, 4, 5, 12],
            [3, 4, 7, 8, 4, 4, 21, 32],
        ],
        dtype=torch.float32,
    ),
}
_GATED = torch.sigmoid((PEPTIDE * PARTNER).sum(dim=1, keepdim=True))
EXPECTED["gated"] = torch.cat(
    [
        _GATED * PEPTIDE + (1.0 - _GATED) * PARTNER,
        (PEPTIDE - PARTNER).abs(),
        PEPTIDE * PARTNER,
    ],
    dim=1,
)


@pytest.mark.parametrize("mode", tuple(EXPECTED))
def test_audited_pair_fusions_have_exact_training_and_inference_parity(mode):
    core = fuse_pair_embeddings(PEPTIDE, PARTNER, mode)
    training = training_fuse_embeddings(PEPTIDE, PARTNER, mode)
    inference = ModelService._fuse_pair_embeddings(PEPTIDE, PARTNER, mode)

    assert core.dtype == torch.float32
    assert torch.equal(core, EXPECTED[mode])
    assert torch.equal(training, core)
    assert torch.equal(inference, core)


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        (" CONCAT ", "concat"),
        ("concat+absdiff", "concat_absdiff"),
        ("concat+hadamard", "concat_hadamard"),
        ("concat+absdiff+hadamard", "concat_all"),
        ("gated(static)", "gated"),
        ("gate", "gated"),
    ],
)
def test_audited_pair_fusion_aliases_are_canonical(alias, canonical):
    assert normalize_pair_fusion_mode(alias) == canonical
    assert torch.equal(fuse_pair_embeddings(PEPTIDE, PARTNER, alias), EXPECTED[canonical])


def test_concat_preserves_role_order_and_allows_heterogeneous_widths():
    peptide = torch.tensor([[1.0, 2.0]], requires_grad=True)
    partner = torch.tensor([[3.0, 4.0, 5.0]], requires_grad=True)

    fused = fuse_pair_embeddings(peptide, partner, "concat")
    fused.sum().backward()

    assert fused.tolist() == [[1.0, 2.0, 3.0, 4.0, 5.0]]
    assert torch.equal(peptide.grad, torch.ones_like(peptide))
    assert torch.equal(partner.grad, torch.ones_like(partner))


@pytest.mark.parametrize("mode", ["concat_absdiff", "concat_hadamard", "concat_all", "gated"])
def test_elementwise_pair_fusions_reject_heterogeneous_widths(mode):
    with pytest.raises(PairFusionError) as exc_info:
        fuse_pair_embeddings(torch.ones(2, 2), torch.ones(2, 3), mode)
    assert exc_info.value.code == "pair_fusion_dimension_mismatch"


@pytest.mark.parametrize(
    ("peptide", "partner", "code"),
    [
        (torch.ones(2), torch.ones(2), "pair_fusion_input_rank_invalid"),
        (torch.ones(1, 2, 1), torch.ones(1, 2, 1), "pair_fusion_input_rank_invalid"),
        (torch.ones(2, 0), torch.ones(2, 0), "pair_fusion_feature_width_invalid"),
        (torch.ones(2, 2), torch.ones(3, 2), "pair_fusion_batch_mismatch"),
        (torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2), "pair_fusion_input_dtype_invalid"),
        (
            torch.ones(1, 2, dtype=torch.complex64),
            torch.ones(1, 2),
            "pair_fusion_input_dtype_invalid",
        ),
        (torch.tensor([[float("nan"), 1.0]]), torch.ones(1, 2), "pair_fusion_input_nonfinite"),
        (torch.tensor([[float("inf"), 1.0]]), torch.ones(1, 2), "pair_fusion_input_nonfinite"),
    ],
)
def test_pair_fusion_invalid_inputs_fail_closed(peptide, partner, code):
    with pytest.raises(PairFusionError) as exc_info:
        fuse_pair_embeddings(peptide, partner, "concat")
    assert exc_info.value.code == code


def test_pair_fusion_rejects_sparse_inputs_and_accepts_empty_batches():
    sparse = torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([1.0]), size=(1, 2))
    with pytest.raises(PairFusionError) as exc_info:
        fuse_pair_embeddings(sparse, torch.ones(1, 2), "concat")
    assert exc_info.value.code == "pair_fusion_input_layout_invalid"

    empty = fuse_pair_embeddings(torch.empty(0, 2), torch.empty(0, 2), "concat")
    assert empty.shape == (0, 4)


@pytest.mark.parametrize("mode", ["concat_all", "gated"])
def test_pair_fusion_rejects_nonfinite_outputs_from_finite_overflow(mode):
    huge = torch.full((1, 1), 1e20)
    with pytest.raises(PairFusionError) as exc_info:
        fuse_pair_embeddings(huge, huge, mode)
    assert exc_info.value.code == "pair_fusion_output_nonfinite"


def test_shared_pair_fusion_does_not_implement_aligned_gated_predictor():
    with pytest.raises(PairFusionError) as exc_info:
        fuse_pair_embeddings(PEPTIDE, PARTNER, "aligned_gated")
    assert exc_info.value.code == "pair_fusion_mode_unsupported"


@pytest.mark.parametrize(
    "removed_name",
    (
        "gated_" + "trainable",
        "gated(" + "trainable)",
        "aligned" + "-gated",
    ),
)
def test_removed_aligned_gated_aliases_are_rejected(removed_name):
    with pytest.raises(PairFusionError) as exc_info:
        normalize_pair_fusion_mode(removed_name)
    assert exc_info.value.code == "pair_fusion_mode_unsupported"


def test_training_and_web_translate_pair_fusion_errors_at_their_boundaries():
    invalid = torch.ones(1, 2, 1)
    with pytest.raises(SystemExit, match="shape "):
        training_fuse_embeddings(invalid, invalid, "concat")
    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._fuse_pair_embeddings(invalid, invalid, "concat")
    assert exc_info.value.code == "pair_fusion_input_rank_invalid"
    assert exc_info.value.stage == "input"


@pytest.mark.parametrize("mode", tuple(EXPECTED))
def test_pair_fusion_contract_binds_mode_roles_and_dimensions(mode):
    fused = fuse_pair_embeddings(PEPTIDE, PARTNER, mode)
    contract = pair_fusion_contract(mode, PEPTIDE, PARTNER, fused)

    assert contract["mode"] == mode
    assert contract["role_order"] == ["peptide", "partner"]
    assert contract["input_dimensions"] == {"peptide": 2, "partner": 2}
    assert contract["output_dimension"] == fused.shape[1]
    if mode == "concat_all":
        assert contract["output_blocks"] == [
            "peptide",
            "partner",
            "absolute_difference",
            "hadamard_product",
        ]
    if mode == "gated":
        assert contract["gate"] == {
            "kind": "static_scalar_similarity",
            "score": "sum(peptide * partner)",
            "activation": "sigmoid",
            "mix": "gate * peptide + (1 - gate) * partner",
            "output_blocks": [
                "gated_mix",
                "absolute_difference",
                "hadamard_product",
            ],
        }
    assert (
        validate_pair_fusion_contract(
            contract,
            mode=mode,
            peptide_embeddings=PEPTIDE,
            partner_embeddings=PARTNER,
            fused_embeddings=fused,
        )
        == contract
    )


def test_pair_fusion_artifact_rejects_tampered_dimensions():
    fused = fuse_pair_embeddings(PEPTIDE, PARTNER, "concat")
    contract = pair_fusion_contract("concat", PEPTIDE, PARTNER, fused)
    contract["output_dimension"] += 1

    with pytest.raises(PairFusionError) as exc_info:
        validate_pair_fusion_contract(
            contract,
            mode="concat",
            peptide_embeddings=PEPTIDE,
            partner_embeddings=PARTNER,
            fused_embeddings=fused,
        )
    assert exc_info.value.code == "pair_fusion_contract_invalid"


@pytest.mark.parametrize("mode", tuple(EXPECTED))
def test_pair_vector_artifact_writes_current_fusion_contract(mode):
    args = Namespace(
        tokenizer="esm2",
        esm2_model="facebook/esm2_t6_8M_UR50D",
        max_length=32,
        fusion_mode=mode,
        classifier="mlp",
        model="mlp",
        task_type="pep_protein_interaction",
    )
    fused = fuse_pair_embeddings(PEPTIDE, PARTNER, mode)

    metadata = _pair_vector_artifact_info(args, PEPTIDE, PARTNER, fused)

    assert metadata["pair_fusion_contract"]["mode"] == mode
    assert metadata["pair"]["fused_dim"] == fused.shape[1]


@pytest.mark.parametrize("mode", ["concat", "concat_all", "gated"])
def test_web_pair_fusion_contract_separates_current_and_legacy_artifacts(mode):
    fused = fuse_pair_embeddings(PEPTIDE, PARTNER, mode)
    current = {
        "pair": {"fusion_mode": mode},
        "pair_fusion_contract": pair_fusion_contract(mode, PEPTIDE, PARTNER, fused),
    }
    assert torch.equal(
        ModelService._fuse_pair_artifact_embeddings(current, PEPTIDE, PARTNER),
        fused,
    )

    with pytest.warns(RuntimeWarning, match="legacy_only"):
        legacy = ModelService._fuse_pair_artifact_embeddings(
            {"pair": {"fusion_mode": mode}}, PEPTIDE, PARTNER
        )
    assert torch.equal(legacy, fused)

    tampered = dict(current)
    tampered["pair_fusion_contract"] = dict(current["pair_fusion_contract"])
    tampered["pair_fusion_contract"]["role_order"] = ["partner", "peptide"]
    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._fuse_pair_artifact_embeddings(tampered, PEPTIDE, PARTNER)
    assert exc_info.value.code == "pair_fusion_contract_invalid"
    assert exc_info.value.stage == "artifact"
