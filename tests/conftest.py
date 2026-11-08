import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def controlled_resource_volume(request, monkeypatch):
    """Web policy tests must not depend on the developer host's free disk space.

    Only the governor's disk probe is replaced; filesystem operations remain real
    and resource tests can override this probe to exercise pressure thresholds.
    """
    if not request.node.path.name.startswith(
        ("test_web_", "test_resource_", "test_dataset_uploads")
    ):
        return
    import shutil

    from pepplatform.web import resource_governor

    capacity = shutil._ntuple_diskusage(100 * 1024**3, 20 * 1024**3, 80 * 1024**3)
    monkeypatch.setattr(
        resource_governor,
        "shutil",
        SimpleNamespace(
            disk_usage=lambda _path: capacity,
            rmtree=shutil.rmtree,
        ),
    )


@pytest.fixture
def quota_clock(monkeypatch):
    import pepplatform.web.resource_governor as governance
    import pepplatform.web.resource_policy as policies
    import pepplatform.web.resource_runner as scheduling
    import pepplatform.web.store as metadata

    clock = SimpleNamespace(value=datetime.now(timezone.utc))
    for module in (governance, policies, scheduling):
        monkeypatch.setattr(module, "now_utc", lambda: clock.value)
    monkeypatch.setattr(metadata, "quota_usage_date", lambda: clock.value.date().isoformat())

    def next_day():
        clock.value = (clock.value + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    clock.next_day = next_day
    return clock
