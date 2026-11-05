import pytest
import torch

from pepplatform.textrank import (
    TEXTRANK_CONTRACT_VERSION,
    TEXTRANK_VARIANT,
    TextRankConvergenceError,
    build_textrank_contract,
    validate_textrank_parameters,
    weighted_textrank,
)


def test_weighted_textrank_converges_to_raw_weighted_fixed_point_before_l1():
    result = weighted_textrank(
        [1, 2, 1, 3],
        feature_dim=5,
        tolerance=1e-12,
    )

    # w(1, 2)=2 and w(1, 3)=1. Solving the weighted TextRank equations gives
    # these raw scores; their scale would be destroyed by iteration-time L1.
    expected_node_1 = 0.405 / 0.2775
    expected = torch.tensor(
        [
            0.0,
            expected_node_1,
            0.15 + 0.85 * (2.0 / 3.0) * expected_node_1,
            0.15 + 0.85 * (1.0 / 3.0) * expected_node_1,
            0.0,
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(result.raw_scores, expected, rtol=0.0, atol=1e-11)
    assert result.raw_scores.dtype == torch.float64
    assert result.raw_scores.sum().item() == pytest.approx(3.0, abs=1e-11)

    normalized = result.l1_normalized()
    assert normalized.dtype == torch.float32
    torch.testing.assert_close(
        normalized.double(),
        expected / expected.sum(),
        rtol=1e-7,
        atol=1e-8,
    )
    assert normalized.sum().item() == pytest.approx(1.0)


def test_window_size_is_sliding_window_width_not_neighbor_radius():
    adjacent_only = weighted_textrank(
        [1, 2, 3],
        feature_dim=4,
        window_size=2,
        tolerance=1e-12,
    ).raw_scores
    full_three_token_window = weighted_textrank(
        [1, 2, 3],
        feature_dim=4,
        window_size=3,
        tolerance=1e-12,
    ).raw_scores

    assert adjacent_only[2] > adjacent_only[1]
    assert adjacent_only[1] == pytest.approx(adjacent_only[3].item())
    assert full_three_token_window[1].item() == pytest.approx(full_three_token_window[2].item())
    assert full_three_token_window[2].item() == pytest.approx(full_three_token_window[3].item())


def test_repeated_cooccurrences_are_weights_and_self_loops_are_omitted():
    weighted = weighted_textrank(
        [1, 2, 1, 3],
        feature_dim=4,
        tolerance=1e-12,
    ).raw_scores
    with_self_loop_positions = weighted_textrank(
        [1, 1, 2, 1, 3],
        feature_dim=4,
        tolerance=1e-12,
    ).raw_scores

    torch.testing.assert_close(with_self_loop_positions, weighted)
    assert weighted[2] > weighted[3]


def test_empty_and_single_node_streams_have_defined_raw_and_final_scores():
    empty = weighted_textrank([], feature_dim=3)
    assert empty.iterations == 0
    assert empty.residual == 0.0
    assert empty.raw_scores.dtype == torch.float64
    assert torch.equal(empty.raw_scores, torch.zeros(3, dtype=torch.float64))
    assert torch.equal(empty.l1_normalized(), torch.zeros(3, dtype=torch.float32))

    single = weighted_textrank([1, 1, 1], feature_dim=3, tolerance=1e-12)
    assert single.raw_scores.tolist() == [0.0, pytest.approx(0.15), 0.0]
    assert single.l1_normalized().tolist() == [0.0, 1.0, 0.0]


@pytest.mark.parametrize("raw_ids", [[-1], [3], [True], [1.5], ["1"]])
def test_weighted_textrank_rejects_invalid_feature_ids(raw_ids):
    with pytest.raises(ValueError, match="feature id at position 0"):
        weighted_textrank(raw_ids, feature_dim=3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"feature_dim": 0}, "feature_dim"),
        ({"feature_dim": True}, "feature_dim"),
        ({"feature_dim": 3, "window_size": 1}, "window_size"),
        ({"feature_dim": 3, "window_size": 11}, "window_size"),
        ({"feature_dim": 3, "damping": 1.0}, "damping"),
        ({"feature_dim": 3, "damping": float("nan")}, "damping"),
        ({"feature_dim": 3, "tolerance": 0.0}, "tolerance"),
        ({"feature_dim": 3, "max_iterations": 0}, "max_iterations"),
    ],
)
def test_textrank_parameters_are_strict(kwargs, message):
    with pytest.raises(ValueError, match=message):
        validate_textrank_parameters(**kwargs)


def test_nonconvergence_raises_instead_of_returning_partial_scores():
    with pytest.raises(TextRankConvergenceError, match="after 1 iterations"):
        weighted_textrank(
            [1, 2, 3],
            feature_dim=4,
            max_iterations=1,
            tolerance=1e-12,
        )


def test_weighted_textrank_is_bitwise_deterministic():
    first = weighted_textrank([3, 1, 2, 3, 2, 1], feature_dim=6)
    second = weighted_textrank([3, 1, 2, 3, 2, 1], feature_dim=6)

    assert first.iterations == second.iterations
    assert first.residual == second.residual
    assert torch.equal(first.raw_scores, second.raw_scores)
    assert torch.equal(first.l1_normalized(), second.l1_normalized())


def test_contract_pins_algorithm_graph_iteration_and_output_semantics():
    contract = build_textrank_contract(
        feature_dim=28,
        stream_policy="tokenizer_emitted_truncated_kmer_stream",
    )

    assert contract == {
        "contract_version": TEXTRANK_CONTRACT_VERSION,
        "variant": TEXTRANK_VARIANT,
        "feature_dimension": 28,
        "stream_policy": "tokenizer_emitted_truncated_kmer_stream",
        "graph_direction": "undirected",
        "edge_weight": "cooccurrence_count",
        "self_loops": False,
        "window_size": 2,
        "window_semantics": "sliding_window_width",
        "initial_score": 1.0,
        "damping": 0.85,
        "update_mode": "synchronous_raw_scores",
        "convergence_metric": "max_absolute_node_delta",
        "convergence_tolerance": 1e-4,
        "max_iterations": 10_000,
        "on_nonconvergence": "error",
        "internal_dtype": "float64",
        "post_normalization": "l1_after_convergence",
        "output_dtype": "float32",
    }
