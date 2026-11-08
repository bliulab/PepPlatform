"""Structured, line-oriented failure records for training subprocesses."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any


TRAINING_FAILURE_PREFIX = "[training-failure]"
TRAINING_FAILURE_SCHEMA_VERSION = 1
_MAX_MESSAGE_LENGTH = 2_000
_MAX_CONTEXT_STRING_LENGTH = 500
_GENERIC_CODES = {
    "process_exit_nonzero",
    "process_terminated_by_signal",
    "training_configuration_error",
    "training_data_invalid",
    "training_runtime_error",
}


def _bounded_text(value: object, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _sanitize_context(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    sanitized: dict[str, object] = {}
    for raw_key, raw_value in value.items():
        key = _bounded_text(raw_key, 80)
        if not key or raw_value is None:
            continue
        if isinstance(raw_value, bool):
            sanitized[key] = raw_value
        elif isinstance(raw_value, int):
            sanitized[key] = raw_value
        elif isinstance(raw_value, float) and math.isfinite(raw_value):
            sanitized[key] = raw_value
        elif isinstance(raw_value, str):
            sanitized[key] = _bounded_text(
                raw_value,
                _MAX_CONTEXT_STRING_LENGTH,
            )
    return sanitized


def _metadata_for_code(code: str) -> tuple[str, str]:
    if code == "svm_convergence_failed":
        return "optimization", "training"
    if code == "svm_artifact_contract_failed":
        return "runtime", "artifact_export"
    if code == "xgb_training_data_invalid":
        return "data", "training"
    if code == "xgb_artifact_contract_failed":
        return "runtime", "artifact_export"
    if code == "lgb_training_data_invalid":
        return "data", "training"
    if code == "lgb_artifact_contract_failed":
        return "runtime", "artifact_export"
    if code == "histgb_training_data_invalid":
        return "data", "training"
    if code == "histgb_artifact_contract_failed":
        return "runtime", "artifact_export"
    if code == "logreg_convergence_failed":
        return "optimization", "training"
    if code == "logreg_artifact_contract_failed":
        return "runtime", "artifact_export"
    if code == "profile_binary_missing":
        return "dependency", "setup"
    if code.startswith("profile_database_"):
        return "profile", "profile_database"
    if code.startswith(("profile_", "psfm_")):
        return "profile", "profile_generation"
    if (
        code
        in {
            "cuda_out_of_memory",
            "host_out_of_memory",
            "process_terminated_by_signal",
        }
        or "memory_limit_exceeded" in code
    ):
        return "resource", "training"
    if code in {"dependency_missing", "file_not_found"}:
        return ("dependency", "setup") if code == "dependency_missing" else ("data", "setup")
    if code.startswith(("dr_", "dt_", "top_ngram_")):
        stage = "feature_encoding"
        category = "resource" if "memory" in code else "data"
        return category, stage
    if code == "training_configuration_error":
        return "configuration", "setup"
    if code == "training_data_invalid":
        return "data", "training"
    return "runtime", "training"


def make_training_failure(
    code: str,
    message: object,
    *,
    category: str | None = None,
    stage: str | None = None,
    context: Mapping[str, object] | None = None,
    exit_code: int | None = None,
    specific: bool | None = None,
) -> dict[str, object]:
    normalized_code = _bounded_text(code, 120) or "training_runtime_error"
    inferred_category, inferred_stage = _metadata_for_code(normalized_code)
    failure: dict[str, object] = {
        "schema_version": TRAINING_FAILURE_SCHEMA_VERSION,
        "code": normalized_code,
        "category": category or inferred_category,
        "stage": stage or inferred_stage,
        "message": _bounded_text(message, _MAX_MESSAGE_LENGTH) or "The training process failed.",
        "context": _sanitize_context(context),
        "_specific": (
            normalized_code not in _GENERIC_CODES if specific is None else bool(specific)
        ),
    }
    if exit_code is not None:
        failure["exit_code"] = int(exit_code)
    return failure


def _system_exit_code(exc: SystemExit) -> int:
    if exc.code is None:
        return 0
    if isinstance(exc.code, int):
        return exc.code
    return 1


def failure_from_exception(
    exc: BaseException,
    *,
    default_stage: str = "training",
) -> dict[str, object]:
    message = str(exc) or exc.__class__.__name__
    lowered = message.lower()
    class_name = exc.__class__.__name__.lower()
    details = getattr(exc, "details", None)
    stable_code = _bounded_text(getattr(exc, "code", ""), 120)
    if not isinstance(exc, SystemExit) and re.fullmatch(r"[a-z][a-z0-9_]*", stable_code):
        return make_training_failure(
            stable_code,
            message,
            context=details,
            specific=True,
        )
    if "outofmemory" in class_name or "cuda out of memory" in lowered:
        return make_training_failure(
            "cuda_out_of_memory",
            message,
            category="resource",
            stage=default_stage,
            specific=True,
        )
    if isinstance(exc, MemoryError) or "std::bad_alloc" in lowered:
        return make_training_failure(
            "host_out_of_memory",
            message,
            category="resource",
            stage=default_stage,
            specific=True,
        )
    if isinstance(exc, ModuleNotFoundError):
        dependency = getattr(exc, "name", None)
        return make_training_failure(
            "dependency_missing",
            message,
            context={"dependency": dependency} if dependency else None,
            specific=True,
        )
    if isinstance(exc, FileNotFoundError):
        return make_training_failure(
            "file_not_found",
            message,
            context={"filename": exc.filename} if exc.filename else None,
            specific=True,
        )
    if isinstance(exc, SystemExit):
        return make_training_failure(
            "training_configuration_error",
            message,
            category="configuration",
            stage="setup",
            exit_code=_system_exit_code(exc),
            specific=False,
        )
    if isinstance(exc, ValueError):
        return make_training_failure(
            "training_data_invalid",
            message,
            category="data",
            stage=default_stage,
            specific=False,
        )
    return make_training_failure(
        "training_runtime_error",
        message,
        category="runtime",
        stage=default_stage,
        specific=False,
    )


def emit_training_failure(failure: Mapping[str, object]) -> None:
    print(
        f"{TRAINING_FAILURE_PREFIX} "
        + json.dumps(dict(failure), ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )


def run_with_failure_reporting(
    main: Callable[[], object],
    *,
    default_stage: str = "training",
) -> object:
    """Run an entrypoint while preserving its normal traceback and exit code."""
    try:
        return main()
    except SystemExit as exc:
        if _system_exit_code(exc) != 0:
            emit_training_failure(failure_from_exception(exc, default_stage=default_stage))
        raise
    except Exception as exc:
        emit_training_failure(failure_from_exception(exc, default_stage=default_stage))
        raise


def parse_training_failure_line(line: str) -> dict[str, object] | None:
    if not str(line).startswith(f"{TRAINING_FAILURE_PREFIX} "):
        return None
    try:
        payload = json.loads(str(line)[len(TRAINING_FAILURE_PREFIX) :].strip())
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        schema_version = int(payload.get("schema_version", 0) or 0)
    except (TypeError, ValueError):
        return None
    if schema_version != TRAINING_FAILURE_SCHEMA_VERSION:
        return None
    code = payload.get("code")
    message = payload.get("message")
    if not isinstance(code, str) or not code.strip() or not isinstance(message, str):
        return None
    return make_training_failure(
        code,
        message,
        category=_bounded_text(payload.get("category"), 40) or None,
        stage=_bounded_text(payload.get("stage"), 60) or None,
        context=payload.get("context") if isinstance(payload.get("context"), dict) else None,
        exit_code=payload.get("exit_code") if isinstance(payload.get("exit_code"), int) else None,
        specific=bool(payload.get("_specific", code not in _GENERIC_CODES)),
    )


def prefer_training_failure(
    current: Mapping[str, object] | None,
    candidate: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if not candidate:
        return dict(current) if current else None
    if not current:
        return dict(candidate)
    current_specific = bool(current.get("_specific"))
    candidate_specific = bool(candidate.get("_specific"))
    if current_specific and not candidate_specific:
        return dict(current)
    return dict(candidate)


def failure_from_process_exit(
    exit_code: int,
    tail_lines: Sequence[str] = (),
) -> dict[str, object]:
    tail = "\n".join(str(line) for line in tail_lines)[-32_000:]
    lowered = tail.lower()
    if "cuda out of memory" in lowered or "cublas_status_alloc_failed" in lowered:
        return make_training_failure(
            "cuda_out_of_memory",
            "CUDA ran out of memory while executing the training process.",
            category="resource",
            stage="training",
            exit_code=exit_code,
            specific=True,
        )
    if "memoryerror" in lowered or "std::bad_alloc" in lowered:
        return make_training_failure(
            "host_out_of_memory",
            "The training process ran out of host memory.",
            category="resource",
            stage="training",
            exit_code=exit_code,
            specific=True,
        )
    dependency_match = re.search(
        r"ModuleNotFoundError:\s+No module named ['\"]([^'\"]+)['\"]",
        tail,
    )
    if dependency_match:
        dependency = dependency_match.group(1)
        return make_training_failure(
            "dependency_missing",
            f"Required Python dependency is unavailable: {dependency}",
            context={"dependency": dependency},
            exit_code=exit_code,
            specific=True,
        )
    if exit_code < 0:
        return make_training_failure(
            "process_terminated_by_signal",
            f"Training process was terminated by signal {-exit_code}.",
            category="resource",
            stage="training",
            context={"signal": -exit_code},
            exit_code=exit_code,
            specific=False,
        )
    return make_training_failure(
        "process_exit_nonzero",
        f"Training process exited with code {exit_code}.",
        category="unknown",
        stage="unknown",
        exit_code=exit_code,
        specific=False,
    )


def finalize_training_failure(
    failure: Mapping[str, Any],
    *,
    exit_code: int | None,
) -> dict[str, object]:
    finalized = {key: value for key, value in dict(failure).items() if not str(key).startswith("_")}
    if exit_code is not None:
        finalized["exit_code"] = int(exit_code)
    return finalized
