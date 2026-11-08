"""Independent local CLI; existing scientific entrypoints remain unchanged."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .storage import (CLIError, DEFAULT_WORKSPACE, export_csv, finish_run, guard_output,
                      history, new_run, read_json, write_json)

TASKS = ("pep_function", "pep_hlai", "pep_hlaii", "pep_protein_interaction")


def build_parser():
    parser = argparse.ArgumentParser(description="Local PepPlatform training, ensembles and inference")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    sub = parser.add_subparsers(dest="command")

    def workspace(p):
        p.add_argument("--workspace", default=argparse.SUPPRESS)

    train = sub.add_parser("train", help="Run unchanged trainers with a provenance envelope")
    workspace(train)
    train.add_argument("--task-type", choices=TASKS, default="pep_function")
    train.add_argument("--interactive", action="store_true")
    train.add_argument("--advanced-wizard", action="store_true")
    train.add_argument("--artifact-dir")
    train.add_argument("trainer_args", nargs=argparse.REMAINDER, help="Original trainer arguments after --")

    ensemble = sub.add_parser("ensemble", help="Preflight or create a local ensemble")
    actions = ensemble.add_subparsers(dest="action", required=True)
    for action in ("preflight", "create"):
        p = actions.add_parser(action)
        workspace(p)
        p.add_argument("--config")
        p.add_argument("--task-type", choices=TASKS, default=argparse.SUPPRESS)
        p.add_argument("--data-dir", default=argparse.SUPPRESS)
        p.add_argument("--source-root", default=argparse.SUPPRESS, help="HLA-I pseudosequence mapping directory")
        p.add_argument("--members", nargs="+", default=argparse.SUPPRESS)
        p.add_argument("--member-provenance", nargs="+", default=argparse.SUPPRESS,
                       help="Training receipts in member order; use - for automatic discovery")
        p.add_argument("--method", choices=("probability_mean", "logit_mean", "rank_mean"), default=argparse.SUPPRESS)
        p.add_argument("--weight-source", choices=("equal", "custom", "validation"), default=argparse.SUPPRESS)
        p.add_argument("--weights", nargs="+", type=float, default=argparse.SUPPRESS)
        p.add_argument("--objective", default=argparse.SUPPRESS)
        p.add_argument("--objective-params", type=json.loads, default=argparse.SUPPRESS)
        p.add_argument("--threshold-objective", default=argparse.SUPPRESS)
        p.add_argument("--threshold-objective-params", type=json.loads, default=argparse.SUPPRESS)
        p.add_argument("--grid-step", type=float, default=argparse.SUPPRESS)
        p.add_argument("--seed", type=int, default=argparse.SUPPRESS)
        p.add_argument("--evaluate-test", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        p.add_argument("--test-splits", nargs="+", default=argparse.SUPPRESS)
        p.add_argument("--allow-unverified-provenance", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        if action == "create":
            p.add_argument("--artifact-dir", default=argparse.SUPPRESS)
        else:
            p.add_argument("--report")

    predict = sub.add_parser("predict", help="Predict from a local model or ensemble")
    workspace(predict)
    predict.add_argument("--artifact-dir", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--source-root", help="HLA-I pseudosequence mapping directory for CSV input")
    hist = sub.add_parser("history", help="Inspect local run history")
    hs = hist.add_subparsers(dest="action", required=True)
    for action in ("list", "show", "export"):
        p = hs.add_parser(action)
        workspace(p)
        if action != "list":
            p.add_argument("run_id")
        if action == "export":
            p.add_argument("--output", required=True)
    return parser


def ensemble_config(args):
    config = read_json(args.config) if args.config else {}
    excluded = {"command", "action", "config", "workspace", "report"}
    config.update({k: v for k, v in vars(args).items() if k not in excluded})
    proofs = config.pop("member_provenance", None)
    if proofs is not None:
        members = config.get("members", [])
        if len(proofs) != len(members):
            raise CLIError("provenance_count_mismatch", "Provenance paths must match member order/count")
        config["members"] = [{**(m if isinstance(m, dict) else {"artifact_dir": m}),
                              **({"provenance": p} if p != "-" else {})} for m, p in zip(members, proofs)]
    return config


def create_ensemble(config, workspace):
    from .ensemble import prepare, assemble
    request = prepare(config)
    if "test_splits" in config:
        request["test_splits"] = config["test_splits"]
    inputs = [request["dataset"]["root_path"], *[r["bundle_path"] for r in request["members"]]]
    run = new_run(workspace, "ensemble", inputs=inputs, artifact_dir=config.get("artifact_dir"))
    directory = Path(run["directory"])
    try:
        write_json(directory / "config.json", config)
        result = assemble(request, run["artifact_dir"], report=lambda event: print(
            json.dumps({"progress": event}), file=sys.stderr, flush=True))
        return finish_run(run, status="succeeded", **{k: v for k, v in result.items() if k != "metadata"})
    except Exception as exc:
        finish_run(run, status="failed", error=str(exc))
        raise


def predict(args):
    from pepplatform.task_compat import normalize_task_type
    from .data import prediction_inputs
    from .inference import predict_artifact
    metadata = read_json(Path(args.artifact_dir) / "metadata.json")
    task = normalize_task_type(metadata.get("args", {}).get("task_type") or metadata.get("task_type"))
    inputs = prediction_inputs(args.input, task, args.source_root)
    run = new_run(args.workspace, "predict", inputs=[args.artifact_dir, args.input])
    directory = Path(run["directory"])
    try:
        write_json(directory / "request.json", inputs)
        response = predict_artifact(args.artifact_dir, inputs, directory / "runtime")
        response["run_id"] = run["run_id"]
        write_json(directory / "response.json", response)
        export_csv(directory / "predictions.csv", response)
        return finish_run(run, status="succeeded", source_artifact=str(Path(args.artifact_dir).resolve()),
                          response=str(directory / "response.json"), csv=str(directory / "predictions.csv"),
                          provenance_status=response["provenance_status"])
    except Exception as exc:
        finish_run(run, status="failed", error=str(exc))
        raise


def dispatch(args):
    if args.command is None:
        from .wizard import navigate
        return navigate(args.workspace)
    if args.command == "train":
        from .training import train
        raw = args.trainer_args[1:] if args.trainer_args[:1] == ["--"] else args.trainer_args
        return train(raw, task=args.task_type, interactive=args.interactive, advanced=args.advanced_wizard,
                     workspace=args.workspace, artifact_dir=args.artifact_dir)
    if args.command == "ensemble":
        from .ensemble import preflight
        config = ensemble_config(args)
        if args.action == "create":
            return create_ensemble(config, args.workspace)
        report = preflight(config)
        if args.report:
            sources = [config.get("data_dir", ""), *[
                m.get("artifact_dir") if isinstance(m, dict) else m for m in config.get("members", [])]]
            destination = guard_output(args.report, [p for p in sources if p])
            if destination.exists():
                raise CLIError("output_exists", f"Report already exists: {destination}")
            write_json(destination, report)
        return report
    if args.command == "predict":
        return predict(args)
    if args.command == "history":
        result = history(args.workspace, getattr(args, "run_id", None))
        if args.action == "export":
            if result.get("kind") != "predict" or result.get("status") != "succeeded":
                raise CLIError("history_not_prediction", "Select a successful inference run")
            destination = guard_output(args.output, [result["directory"], result.get("source_artifact", "")])
            if destination.exists():
                raise CLIError("output_exists", f"Export already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            export_csv(destination, read_json(result["response"]))
            return {"run_id": result["run_id"], "csv": str(destination)}
        return result
    raise CLIError("command_invalid", "Unsupported command")


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = dispatch(args)
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 2 if isinstance(result, dict) and result.get("valid") is False else 0
    except (KeyboardInterrupt, EOFError):
        print("Cancelled.", file=sys.stderr)
        return 130
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"error": {"code": getattr(exc, "code", "cli_failed"), "message": str(exc)}},
                         ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
