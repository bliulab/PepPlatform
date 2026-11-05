"""Canonical names for pair input organization. Legacy inputs normalize at entry."""

from pepplatform.stream_compat import STRATEGY_ALIASES

PAIR_STRATEGY_ALIASES = STRATEGY_ALIASES
PAIR_STRATEGY_LABELS = {
    "single_stream": {"en": "Single-stream", "zh": "单流"},
    "two_stream": {"en": "Two-stream", "zh": "双流"},
}
PAIR_STRATEGY_PUBLIC_VALUES = {key: key for key in PAIR_STRATEGY_LABELS}


def normalize_pair_strategy(value):
    key = str(value or "").strip().lower()
    if not key:
        return ""
    try:
        return PAIR_STRATEGY_ALIASES[key]
    except KeyError:
        raise ValueError(f"unsupported pair_strategy: {value}; use single_stream or two_stream") from None


def public_pair_strategy(value):
    return normalize_pair_strategy(value)
