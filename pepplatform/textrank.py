"""Strict weighted TextRank kernel and reproducibility contract.

The ranking update follows Mihalcea and Tarau (2004), Section 2.2. Graph
construction and the final L1 feature normalization are platform policies kept
outside the iterative kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterable

import torch


TEXTRANK_CONTRACT_VERSION = 1
TEXTRANK_VARIANT = "weighted_textrank_2004_cooccurrence_count"
TEXTRANK_DEFAULT_WINDOW_SIZE = 2
TEXTRANK_DEFAULT_DAMPING = 0.85
TEXTRANK_DEFAULT_TOLERANCE = 1e-4
TEXTRANK_DEFAULT_MAX_ITERATIONS = 10_000


class TextRankConvergenceError(RuntimeError):
    """Raised instead of returning scores that did not satisfy tolerance."""


@dataclass(frozen=True)
class TextRankResult:
    raw_scores: torch.Tensor
    iterations: int
    residual: float

    def l1_normalized(self) -> torch.Tensor:
        total = float(self.raw_scores.sum().item())
        if total == 0.0 and not bool(self.raw_scores.any()):
            return self.raw_scores.to(dtype=torch.float32)
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError("TextRank converged to an invalid score total")
        return (self.raw_scores / total).to(dtype=torch.float32)


def _positive_integer(value, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return normalized


def validate_textrank_parameters(
    *,
    feature_dim,
    window_size=TEXTRANK_DEFAULT_WINDOW_SIZE,
    damping=TEXTRANK_DEFAULT_DAMPING,
    tolerance=TEXTRANK_DEFAULT_TOLERANCE,
    max_iterations=TEXTRANK_DEFAULT_MAX_ITERATIONS,
) -> tuple[int, int, float, float, int]:
    dimension = _positive_integer(feature_dim, name="TextRank feature_dim")
    window = _positive_integer(window_size, name="TextRank window_size", minimum=2)
    if window > 10:
        raise ValueError("TextRank window_size must be in [2, 10]")
    if isinstance(damping, bool) or not isinstance(damping, Real):
        raise ValueError("TextRank damping must be a finite number in [0, 1)")
    damping_value = float(damping)
    if not math.isfinite(damping_value) or not 0.0 <= damping_value < 1.0:
        raise ValueError("TextRank damping must be a finite number in [0, 1)")
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        raise ValueError("TextRank tolerance must be a finite positive number")
    tolerance_value = float(tolerance)
    if not math.isfinite(tolerance_value) or tolerance_value <= 0.0:
        raise ValueError("TextRank tolerance must be a finite positive number")
    iteration_limit = _positive_integer(
        max_iterations,
        name="TextRank max_iterations",
    )
    return dimension, window, damping_value, tolerance_value, iteration_limit


def validate_feature_ids(
    raw_ids: Iterable[object],
    *,
    feature_dim: int,
    context: str,
) -> list[int]:
    ids: list[int] = []
    for position, value in enumerate(raw_ids):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(
                f"{context} feature id at position {position} must be an integer, got {value!r}"
            )
        feature_id = int(value)
        if not 0 <= feature_id < feature_dim:
            raise ValueError(
                f"{context} feature id at position {position} is outside "
                f"[0, {feature_dim}): {feature_id}"
            )
        ids.append(feature_id)
    return ids


def weighted_textrank(
    raw_ids: Iterable[object],
    *,
    feature_dim,
    window_size=TEXTRANK_DEFAULT_WINDOW_SIZE,
    damping=TEXTRANK_DEFAULT_DAMPING,
    tolerance=TEXTRANK_DEFAULT_TOLERANCE,
    max_iterations=TEXTRANK_DEFAULT_MAX_ITERATIONS,
    context="TextRank",
) -> TextRankResult:
    """Compute raw weighted TextRank scores without iteration-time normalization."""

    dimension, window, damping_value, tolerance_value, iteration_limit = (
        validate_textrank_parameters(
            feature_dim=feature_dim,
            window_size=window_size,
            damping=damping,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    )
    ids = validate_feature_ids(raw_ids, feature_dim=dimension, context=context)
    raw_output = torch.zeros(dimension, dtype=torch.float64)
    if not ids:
        return TextRankResult(raw_output, iterations=0, residual=0.0)

    adjacency: dict[int, dict[int, float]] = {}
    for feature_id in ids:
        adjacency.setdefault(feature_id, {})
    for position, source in enumerate(ids):
        # A window of N tokens contains N-1 following positions.
        right = min(position + window, len(ids))
        for neighbor_position in range(position + 1, right):
            target = ids[neighbor_position]
            if source == target:
                continue
            adjacency[source][target] = adjacency[source].get(target, 0.0) + 1.0
            adjacency[target][source] = adjacency[target].get(source, 0.0) + 1.0

    nodes = sorted(adjacency)
    out_weight = {node: math.fsum(adjacency[node].values()) for node in nodes}
    rank = {node: 1.0 for node in nodes}
    residual = math.inf
    for iteration in range(1, iteration_limit + 1):
        new_rank: dict[int, float] = {}
        for node in nodes:
            incoming = math.fsum(
                (weight / out_weight[neighbor]) * rank[neighbor]
                for neighbor, weight in sorted(adjacency[node].items())
                if out_weight[neighbor] > 0.0
            )
            new_rank[node] = (1.0 - damping_value) + damping_value * incoming
        residual = max(abs(new_rank[node] - rank[node]) for node in nodes)
        rank = new_rank
        if residual < tolerance_value:
            for node in nodes:
                raw_output[node] = rank[node]
            if not bool(torch.isfinite(raw_output).all()) or bool((raw_output < 0).any()):
                raise RuntimeError("TextRank converged to non-finite or negative scores")
            return TextRankResult(raw_output, iteration, residual)

    raise TextRankConvergenceError(
        f"{context} did not converge after {iteration_limit} iterations: "
        f"residual={residual:.12g}, tolerance={tolerance_value:.12g}"
    )


def build_textrank_contract(*, feature_dim: int, stream_policy: str) -> dict[str, object]:
    dimension, window, damping, tolerance, max_iterations = validate_textrank_parameters(
        feature_dim=feature_dim,
    )
    return {
        "contract_version": TEXTRANK_CONTRACT_VERSION,
        "variant": TEXTRANK_VARIANT,
        "feature_dimension": dimension,
        "stream_policy": str(stream_policy),
        "graph_direction": "undirected",
        "edge_weight": "cooccurrence_count",
        "self_loops": False,
        "window_size": window,
        "window_semantics": "sliding_window_width",
        "initial_score": 1.0,
        "damping": damping,
        "update_mode": "synchronous_raw_scores",
        "convergence_metric": "max_absolute_node_delta",
        "convergence_tolerance": tolerance,
        "max_iterations": max_iterations,
        "on_nonconvergence": "error",
        "internal_dtype": "float64",
        "post_normalization": "l1_after_convergence",
        "output_dtype": "float32",
    }
