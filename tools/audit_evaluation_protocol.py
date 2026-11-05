#!/usr/bin/env python3
"""Verify saved evaluation metadata and metrics without retraining models."""

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pepplatform.evaluation_audit import audit_artifact
from pepplatform.evaluators import EVALUATION_PROTOCOL_VERSION


def main():
    parser = argparse.ArgumentParser(
        description=f"Verify saved evaluation files (format {EVALUATION_PROTOCOL_VERSION}) without retraining models."
    )
    parser.add_argument(
        "root", help="artifact directory or a parent directory containing evaluation.json files"
    )
    parser.add_argument("--tolerance", type=float, default=1e-7)
    args = parser.parse_args()

    root = Path(args.root)
    paths = [root] if root.name == "evaluation.json" else sorted(root.rglob("evaluation.json"))
    if not paths:
        raise SystemExit(f"no evaluation.json files found under {root}")
    failures = 0
    for path in paths:
        errors = audit_artifact(path, tolerance=args.tolerance)
        if errors:
            failures += 1
            print(f"[FAIL] {path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"[OK] {path}")
    if failures:
        raise SystemExit(
            f"{failures}/{len(paths)} evaluation files failed the format {EVALUATION_PROTOCOL_VERSION} audit"
        )
    print(f"{len(paths)} evaluation files passed the format {EVALUATION_PROTOCOL_VERSION} audit")


if __name__ == "__main__":
    main()
