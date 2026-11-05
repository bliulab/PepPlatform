"""Read compatibility for historical tower names; current runtime writes stream names.

Original files and their hashes are never modified by these adapters. Frozen
protocols are validated by their original readers before conversion.
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
import types
import functools
from pathlib import Path

STRATEGY_ALIASES = {
    "single_stream": "single_stream",
    "single-stream": "single_stream",
    "two_stream": "two_stream",
    "two-stream": "two_stream",
    "single_tower": "single_stream",
    "single-tower": "single_stream",
    "dual_tower": "two_stream",
    "dual-tower": "two_stream",
}
_REPLACEMENTS = (
    ("DUAL_TOWER", "TWO_STREAM"), ("DualTower", "TwoStream"),
    ("dualTower", "twoStream"), ("dual_tower", "two_stream"),
    ("dual-tower", "two-stream"), ("SINGLE_TOWER", "SINGLE_STREAM"),
    ("SingleTower", "SingleStream"), ("singleTower", "singleStream"),
    ("single_tower", "single_stream"), ("single-tower", "single-stream"),
    ("TOWER", "STREAM"), ("Tower", "Stream"), ("tower", "stream"),
)
_CONTRACT_KEYS = frozenset({"pair_strategy_contract", "pair_bow_joint_contract"})


def canonical_identifier(value):
    """Translate a known field/symbol, never a candidate ID or file path."""
    if (not isinstance(value, str) or re.match(r"^c\d{3}_", value)
            or "/" in value or "\\" in value):
        return value
    for old, new in _REPLACEMENTS:
        value = value.replace(old, new)
    return value


def legacy_identifier(value):
    if re.match(r"^c\d{3}_", value) or "/" in value or "\\" in value:
        return value
    for new, old in (
        ("TWO_STREAM", "DUAL_TOWER"), ("TwoStream", "DualTower"),
        ("twoStream", "dualTower"), ("two_stream", "dual_tower"),
        ("SINGLE_STREAM", "SINGLE_TOWER"), ("SingleStream", "SingleTower"),
        ("singleStream", "singleTower"), ("single_stream", "single_tower"),
    ):
        value = value.replace(new, old)
    value = re.sub(r"(?<![a-z])stream(?=s?(?:_|$))", "tower", value)
    value = re.sub(r"STREAM(?=S?(?:_|$))", "TOWER", value)
    value = re.sub(r"Stream(?=s?(?:[A-Z_]|$))", "Tower", value)
    return value


def _equal(left, right):
    if left is right:
        return True
    try:
        return bool(left == right)
    except (ValueError, TypeError):
        return False


def normalize_stream_fields(payload):
    """Copy configuration views, retaining raw protocol/provenance records.

    Only identifier keys and explicit strategy/adapter values are translated.
    Paths, record IDs, labels and source manifests retain their exact spelling.
    Protocol subobjects are left intact for version-aware validation.
    """
    if isinstance(payload, list):
        return [normalize_stream_fields(v) for v in payload]
    if isinstance(payload, tuple):
        return tuple(normalize_stream_fields(v) for v in payload)
    if not isinstance(payload, dict):
        return payload
    result = {}
    for key, value in payload.items():
        renamed = canonical_identifier(key) if isinstance(key, str) and key.isidentifier() else key
        if key in _CONTRACT_KEYS or (isinstance(key, str) and key.endswith(("_contract", "_manifest"))):
            normalized = copy.deepcopy(value)
        else:
            normalized = normalize_stream_fields(value)
        if key in {"pair_strategy", "strategy"} and isinstance(value, str):
            normalized = STRATEGY_ALIASES.get(value.strip().lower(), value)
        elif key in {"input_adapter", "route", "pair_route", "code", "reason_code", "inference_unavailable_code"} and isinstance(value, str):
            normalized = canonical_identifier(value)
        if renamed in result and not _equal(result[renamed], normalized):
            raise ValueError(f"Conflicting historical and current stream fields: {renamed}")
        result[renamed] = normalized
    return result


def normalize_stream_namespace(args):
    values = normalize_stream_fields(vars(args))
    vars(args).clear()
    vars(args).update(values)
    return args


def legacy_stream_fields(payload):
    """Old import shims expose the calling shape expected by historical scripts."""
    if isinstance(payload, list):
        return [legacy_stream_fields(v) for v in payload]
    if not isinstance(payload, dict):
        return payload
    result = {}
    for key, value in payload.items():
        renamed = legacy_identifier(key) if isinstance(key, str) and key.isidentifier() else key
        value = legacy_stream_fields(value)
        if key in {"pair_strategy", "strategy"} and isinstance(value, str) and value in {"single_stream", "two_stream"}:
            value = {"single_stream": "single_tower", "two_stream": "dual_tower"}[value]
        result[renamed] = value
    return result


def upgrade_strategy_contract(contract):
    if not isinstance(contract, dict) or contract.get("contract_version") != 1:
        return contract
    from ._legacy_stream import pair_strategy as legacy
    from .pair_strategy import PairStrategyContractError

    try:
        if contract.get("strategy") == "single_tower":
            original = legacy.validate_single_tower_strategy_contract(contract)
        else:
            original = legacy.validate_dual_tower_strategy_contract(contract)
    except legacy.PairStrategyContractError as exc:
        raise PairStrategyContractError(canonical_identifier(exc.code), str(exc)) from exc
    result = normalize_stream_fields(original)
    result["contract_version"] = 2
    return result


def upgrade_pair_bow_contract(contract):
    if not isinstance(contract, dict) or contract.get("contract_version") != 1:
        return contract
    from ._legacy_stream import pair_bow as legacy

    legacy.validate_pair_bow_joint_contract(contract)
    result = normalize_stream_fields(contract)
    result["contract_version"] = 2
    return result


def is_legacy_pair_bow_route(value):
    return value == "pair_dual_tower_classic_bow_v2"


def is_pair_bow_route(value):
    from .pair_bow import PAIR_BOW_ROUTE

    return value == PAIR_BOW_ROUTE or is_legacy_pair_bow_route(value)


def normalize_attention_contract(contract):
    """Adapt the single historical descriptive literal, retaining all checks."""
    result = dict(contract)
    if result.get("masking") == "opposite-tower key padding masks and both query masked-mean readouts":
        result["masking"] = "opposite-stream key padding masks and both query masked-mean readouts"
    return result


def stream_cache_root(value):
    """New training writes into a separate namespace; old caches stay intact."""
    root = Path(value)
    return str(root if root.name == "stream_v2" else root / "stream_v2")


def install_legacy_class_exports(cls):
    for name, value in tuple(vars(cls).items()):
        historical = legacy_identifier(name)
        if historical != name and not hasattr(cls, historical):
            setattr(cls, historical, value)
    return cls


def install_legacy_exports(namespace):
    """Keep frozen import/pickle names as aliases, with no legacy runtime branches."""
    for name, value in tuple(namespace.items()):
        historical = legacy_identifier(name)
        if historical != name:
            if isinstance(value, types.FunctionType):
                @functools.wraps(value)
                def historical_call(*args, _function=value, **kwargs):
                    supplied = {canonical_identifier(k): v for k, v in kwargs.items()}
                    result = _function(*args, **supplied)
                    for item in args:
                        if isinstance(item, (argparse.Namespace, types.SimpleNamespace)):
                            for key, child in tuple(vars(item).items()):
                                alias = legacy_identifier(key)
                                if alias != key:
                                    setattr(item, alias, child)
                    return result
                namespace.setdefault(historical, historical_call)
            else:
                namespace.setdefault(historical, value)


def register_legacy_cli_aliases(parser):
    """Accept old options without advertising them; reject conflicting duplicates."""
    if getattr(parser, "_stream_compat_registered", False):
        return parser
    parser._stream_compat_registered = True
    for option, action in list(parser._option_string_actions.items()):
        if option.startswith("--"):
            historical = "--" + legacy_identifier(option[2:])
            if historical != option:
                parser._option_string_actions.setdefault(historical, action)
    original = parser.parse_known_args

    def parse_known_args(self, args=None, namespace=None):
        values = list(sys.argv[1:] if args is None else args)
        seen = {}
        for i, token in enumerate(values):
            if not token.startswith("--"):
                continue
            option, sep, raw = token.partition("=")
            action = self._option_string_actions.get(option)
            if action is None or action.nargs == 0 or isinstance(action, argparse._AppendAction):
                continue
            if "stream" not in action.dest and action.dest != "pair_strategy":
                continue
            if not sep and i + 1 < len(values):
                raw = values[i + 1]
            try:
                converted = action.type(raw) if action.type else raw
            except (ValueError, TypeError, argparse.ArgumentTypeError):
                continue  # argparse emits its normal typed error.
            if action.dest in seen and not _equal(seen[action.dest], converted):
                self.error(f"conflicting historical/current options for {action.dest}")
            seen[action.dest] = converted
        return original(values, namespace)

    parser.parse_known_args = types.MethodType(parse_known_args, parser)
    return parser
