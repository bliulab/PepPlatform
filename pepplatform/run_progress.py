"""Structured, newline-delimited experiment progress records."""

import math
import re


RUN_PROGRESS_PREFIX = "[run-progress]"
RUN_PROGRESS_STAGES = (
    "representation",
    "embedding_export",
    "classifier_fit",
    "finalization",
)
RUN_PROGRESS_STATUSES = ("running", "complete", "failed")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.+:-]+$")
_FIELD_RE = re.compile(r"(?P<key>[a-z_]+)=(?P<value>[^\s]+)")
_INTEGER_FIELDS = {
    "current",
    "total",
    "output",
    "outputs",
    "iteration",
    "iterations",
    "threads",
}
_FLOAT_FIELDS = {"elapsed_seconds", "eta_seconds"}


def _format_value(key, value):
    if key in _INTEGER_FIELDS:
        return str(int(value))
    if key in _FLOAT_FIELDS:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"{key} must be a finite non-negative number")
        return f"{numeric:.3f}"
    token = str(value).strip()
    if not token or not _TOKEN_RE.fullmatch(token):
        raise ValueError(f"{key} must be a non-empty progress token")
    return token


def format_run_progress(stage, status, current, total, **fields):
    """Return one parseable progress line with a stable field order."""
    stage = str(stage).strip()
    status = str(status).strip()
    current = int(current)
    total = int(total)
    if stage not in RUN_PROGRESS_STAGES:
        raise ValueError(f"unsupported run progress stage: {stage}")
    if status not in RUN_PROGRESS_STATUSES:
        raise ValueError(f"unsupported run progress status: {status}")
    if total < 1 or current < 0 or current > total:
        raise ValueError("run progress requires 0 <= current <= total and total >= 1")
    ordered = [
        ("stage", stage),
        ("status", status),
        ("current", current),
        ("total", total),
    ]
    ordered.extend((key, value) for key, value in fields.items() if value is not None)
    return (
        RUN_PROGRESS_PREFIX
        + " "
        + " ".join(f"{key}={_format_value(key, value)}" for key, value in ordered)
    )


def emit_run_progress(stage, status, current, total, **fields):
    """Print one progress record immediately for CLI and Web runners."""
    print(format_run_progress(stage, status, current, total, **fields), flush=True)


def parse_run_progress_line(line):
    """Parse and validate one progress record into a metric stage event."""
    raw = str(line or "").strip()
    if not raw.startswith(RUN_PROGRESS_PREFIX + " "):
        return None
    fields = dict(_FIELD_RE.findall(raw[len(RUN_PROGRESS_PREFIX) + 1 :]))
    stage = fields.get("stage", "")
    status = fields.get("status", "")
    if stage not in RUN_PROGRESS_STAGES or status not in RUN_PROGRESS_STATUSES:
        return None
    try:
        parsed = {key: int(fields[key]) for key in _INTEGER_FIELDS if key in fields}
        parsed.update({key: float(fields[key]) for key in _FLOAT_FIELDS if key in fields})
        current = parsed["current"]
        total = parsed["total"]
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if total < 1 or current < 0 or current > total:
        return None
    for key in _FLOAT_FIELDS:
        if key in parsed and (not math.isfinite(parsed[key]) or parsed[key] < 0):
            return None
    for current_key, total_key in (("output", "outputs"), ("iteration", "iterations")):
        if current_key in parsed or total_key in parsed:
            if current_key not in parsed or total_key not in parsed:
                return None
            if (
                parsed[total_key] < 1
                or parsed[current_key] < 0
                or parsed[current_key] > parsed[total_key]
            ):
                return None
    if "threads" in parsed and parsed["threads"] < 1:
        return None
    event = {
        "kind": "stage",
        "stage": stage,
        "status": status,
        "current": current,
        "total": total,
    }
    event.update(parsed)
    for key in ("feature", "unit", "split"):
        value = fields.get(key, "").strip()
        if value:
            if not _TOKEN_RE.fullmatch(value):
                return None
            event[key] = value
    return event
