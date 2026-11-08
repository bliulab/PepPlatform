"""Private JSON-lines protocol for a single supervised inference worker."""

import json
import os
import resource
import sys
from contextlib import redirect_stdout

from .worker_limits import parent_death_signal


def apply_file_limit(limits):
    """Reset the soft limit for every request in a reused worker."""
    file_limit = (
        resource.RLIM_INFINITY
        if limits.get("enforcement") == "exempt_admin"
        else limits.get("max_model_file_bytes", resource.RLIM_INFINITY)
    )
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, resource.RLIM_INFINITY))


def main():
    parent_death_signal(int(os.environ["PEP_RESOURCE_PARENT_PID"]))
    output = sys.stdout

    def send(event, data=None, **extra):
        def encode(value):
            if hasattr(value, "tolist"):
                return value.tolist()
            if hasattr(value, "item"):
                return value.item()
            raise TypeError(f"unsupported inference response type: {type(value).__name__}")

        output.write(json.dumps({"event": event, "data": data, **extra}, default=encode) + "\n")
        output.flush()

    with redirect_stdout(sys.stderr):
        from pathlib import Path

        from pepplatform.versioning.core import VersionRepository
        from pepplatform.versioning.web import VersionedModelService

        from .ensemble_math import EnsembleError
        from .inference import InferenceUnsupportedError
        from .store import MetadataStore

        store = MetadataStore(os.environ["PEP_RESOURCE_DATABASE_URL"], initialize=False)
        versions = VersionRepository(
            store.engine, Path(os.environ["PEP_RESOURCE_RUNTIME_ROOT"]) / "versions"
        )
        models = VersionedModelService(store, versions)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                limits = request.get("resource_limits") or {}
                apply_file_limit(limits)
                result = models.predict(
                    request["model_id"],
                    sequences=request.get("sequences"),
                    pairs=request.get("pairs"),
                    progress_callback=lambda event: send("stage", event),
                )
                send("done", result)
            except Exception as exc:  # noqa: BLE001 -- Convert every task failure to the private worker protocol.
                if isinstance(exc, EnsembleError):
                    status, detail = 409, exc.detail()
                elif isinstance(exc, InferenceUnsupportedError):
                    status, detail = 409, exc.to_detail()
                elif isinstance(exc, KeyError):
                    status, detail = 404, {"code": "model_not_found", "message": "model not found"}
                elif isinstance(exc, ValueError):
                    status, detail = 400, {"code": "invalid_prediction", "message": str(exc)}
                else:
                    status, detail = (
                        500,
                        {"code": "inference_failed", "stage": "inference", "message": str(exc)},
                    )
                send("error", detail, status_code=status)
        models.close()


if __name__ == "__main__":
    main()
