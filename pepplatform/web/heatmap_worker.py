"""One short-lived CPU worker, isolated from the training runner."""

import json
import sys
from pathlib import Path

from .heatmap_data import HeatmapError, input_files
from .heatmap_compute import compute


def run(request_path):
    request = json.loads(Path(request_path).read_text())
    try:
        if input_files(request["context"], request["params"]) != request["inputs"]:
            raise HeatmapError("input_changed")
        result = compute(request["context"], request["params"])
        if input_files(request["context"], request["params"]) != request["inputs"]:
            raise HeatmapError("input_changed")
        result["provenance"] = request["provenance"]
        target = Path(request_path).with_name("result.json")
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, allow_nan=False, ensure_ascii=False))
        temporary.replace(target)
    except Exception as exc:
        Path(request_path).with_name("error.json").write_text(
            json.dumps({"code": exc.code if isinstance(exc, HeatmapError) else "analysis_failed"})
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
