"""Compatibility import for frozen scripts; new code uses stream_spec."""

from argparse import Namespace
from functools import wraps

from . import stream_spec as _current
from .stream_compat import canonical_identifier, legacy_identifier, legacy_stream_fields


def _historical(value):
    if isinstance(value, Namespace):
        original = legacy_stream_fields(vars(value))
        vars(value).clear()
        vars(value).update(original)
        return value
    if isinstance(value, dict):
        return legacy_stream_fields(value)
    if isinstance(value, tuple):
        return tuple(_historical(v) for v in value)
    if isinstance(value, str):
        return legacy_identifier(value)
    return value


def __getattr__(name):
    current = getattr(_current, canonical_identifier(name))
    if not callable(current):
        return _historical(current)

    @wraps(current)
    def historical_call(*args, **kwargs):
        return _historical(current(*args, **kwargs))

    return historical_call


__all__ = [legacy_identifier(name) for name in vars(_current) if not name.startswith("_")]
