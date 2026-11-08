"""Line-oriented progress records for long-running profile generation."""

from __future__ import annotations

import math
import time
from collections.abc import Callable


PROFILE_PROGRESS_PREFIX = "[profile-progress]"


def emit_profile_progress(
    stage: str,
    status: str,
    *,
    current: int,
    total: int,
    feature: str,
    elapsed_seconds: float | None = None,
) -> None:
    """Emit one newline-delimited record that can cross a subprocess pipe."""
    fields = [
        PROFILE_PROGRESS_PREFIX,
        f"stage={stage}",
        f"status={status}",
        f"current={max(0, int(current))}",
        f"total={max(0, int(total))}",
        f"feature={feature}",
    ]
    if elapsed_seconds is not None and math.isfinite(float(elapsed_seconds)):
        fields.append(f"elapsed_seconds={max(0.0, float(elapsed_seconds)):.3f}")
    print(" ".join(fields), flush=True)


class ProfileProgressReporter:
    """Throttle profile progress to useful log updates without flooding logs."""

    def __init__(
        self,
        total: int,
        feature: str,
        *,
        stage: str = "profiles",
        minimum_interval_seconds: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.total = max(0, int(total))
        self.feature = feature
        self.stage = stage
        self.minimum_step = max(1, math.ceil(self.total / 100)) if self.total else 1
        self.minimum_interval_seconds = float(minimum_interval_seconds)
        self.clock = clock
        self.last_reported = 0
        self.last_reported_at = 0.0
        self.started_at = 0.0

    def _elapsed(self, now: float) -> float:
        return max(0.0, now - self.started_at)

    def start(self) -> None:
        now = self.clock()
        self.started_at = now
        emit_profile_progress(
            self.stage,
            "running",
            current=0,
            total=self.total,
            feature=self.feature,
            elapsed_seconds=0.0,
        )
        self.last_reported_at = now

    def update(self, current: int) -> None:
        current = min(self.total, max(0, int(current)))
        now = self.clock()
        should_report = (
            current == 1
            or current >= self.total
            or current - self.last_reported >= self.minimum_step
            or now - self.last_reported_at >= self.minimum_interval_seconds
        )
        if not should_report:
            return
        emit_profile_progress(
            self.stage,
            "running",
            current=current,
            total=self.total,
            feature=self.feature,
            elapsed_seconds=self._elapsed(now),
        )
        self.last_reported = current
        self.last_reported_at = now

    def complete(self) -> None:
        now = self.clock()
        emit_profile_progress(
            self.stage,
            "complete",
            current=self.total,
            total=self.total,
            feature=self.feature,
            elapsed_seconds=self._elapsed(now),
        )
