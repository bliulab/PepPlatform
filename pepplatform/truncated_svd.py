"""Shared TruncatedSVD primitive used by semantic LSA and generic ML reduction.

The estimator is shared deliberately, while route validation and artifact
contracts remain owned by their respective features.
"""

from __future__ import annotations

from typing import Any

from sklearn.decomposition import TruncatedSVD


TRUNCATED_SVD_PARAMS: dict[str, Any] = {
    "algorithm": "randomized",
    "n_iter": 5,
    "n_oversamples": 10,
    "power_iteration_normalizer": "auto",
    "tol": 0.0,
}


def truncated_svd_params(seed: int) -> dict[str, Any]:
    """Return the platform's stable uncentered randomized SVD parameters."""
    return {**TRUNCATED_SVD_PARAMS, "random_state": int(seed)}


def new_truncated_svd(n_components: int, seed: int) -> TruncatedSVD:
    """Build the shared estimator without imposing route-specific semantics."""
    return TruncatedSVD(
        n_components=int(n_components),
        **truncated_svd_params(seed),
    )


# Explicit alias for callers that prefer a factory-style name.
make_truncated_svd = new_truncated_svd
