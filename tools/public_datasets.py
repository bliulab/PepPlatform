#!/usr/bin/env python3
"""Fetch public database snapshots and build reproducible PepPlatform ZIPs."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pepplatform.public_data import DataError, build, fetch, verify


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("fetch", help="freeze official API responses and local resources")
    get.add_argument("--recipe", required=True, type=Path)
    get.add_argument("--out", required=True, type=Path)
    make = commands.add_parser("build", help="build offline from a complete, verified snapshot")
    make.add_argument("--snapshot", required=True, type=Path)
    make.add_argument("--recipe", required=True, type=Path)
    make.add_argument("--out", required=True, type=Path)
    check = commands.add_parser(
        "verify", help="verify hashes, provenance, split assignments and ZIP"
    )
    check.add_argument("--dataset", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            result = fetch(args.recipe, args.out)
            result = {
                "complete": result["complete"],
                "records": result["record_count"],
                "snapshot": str(args.out),
            }
        elif args.command == "build":
            result = build(args.snapshot, args.recipe, args.out)
        else:
            result = verify(args.dataset)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (DataError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
