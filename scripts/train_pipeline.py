# Resolve local package imports before accepting historical CLI entry points.
import sys as _stream_sys
from pathlib import Path as _StreamPath
_stream_root = str(_StreamPath(__file__).resolve().parents[1])
if _stream_root not in _stream_sys.path:
    _stream_sys.path.insert(0, _stream_root)

from pepplatform.task_compat import normalize_task_type
from pepplatform.cli_training import add_language_argument

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from scripts.train_sequence_pipeline import (  # noqa: E402
    Pipeline,
    SequencePipeline,
    _resolve_pretrained_model_source,
    set_global_seed,
)


PAIR_TASK_TYPES = {"pep_protein_interaction", "pep_hlaii", "pep_hlai"}
SEQUENCE_TASK_TYPES = {"pep_function"}
TASK_ALIASES = {
    "pep_function_multilabel": "pep_function",
    "pep_hlaii_interaction": "pep_hlaii",
    "peptide_hlaii_interaction": "pep_hlaii",
    "pep_hlai_interaction": "pep_hlai",
    "peptide_hlai_interaction": "pep_hlai",
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Dispatcher for PepPlatform training entrypoints",
        epilog=(
            "Recommended entrypoints:\n"
            "  Single-sequence tasks: python scripts/train_sequence_pipeline.py --task_type pep_function ...\n"
            "  Pair-interaction tasks: python scripts/train_pair_pipeline.py --task_type pep_hlaii ...\n\n"
            "This dispatcher keeps old scripts/train_pipeline.py commands working by forwarding\n"
            "arguments to the task-specific entrypoint."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="pep_function",
        help="Task used for dispatch: pep_function, pep_protein_interaction, pep_hlaii, or pep_hlai.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Forward interactive mode to the selected task-specific entrypoint.",
    )
    parser.add_argument(
        "--advanced_wizard",
        action="store_true",
        help="Forward advanced wizard mode to the selected task-specific entrypoint.",
    )
    add_language_argument(parser)
    return register_legacy_cli_aliases(parser)


def _run_target(module_name, script_name, argv):
    old_argv = sys.argv[:]
    try:
        sys.argv = [os.path.join(SCRIPT_DIR, script_name), *argv]
        if module_name == "pair":
            from scripts import train_pair_pipeline

            return train_pair_pipeline.main()
        from scripts import train_sequence_pipeline

        return train_sequence_pipeline.main()
    finally:
        sys.argv = old_argv


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    dispatch_args, _unknown = parser.parse_known_args(argv)
    task_type = normalize_task_type(dispatch_args.task_type)
    if normalize_task_type(task_type) in PAIR_TASK_TYPES:
        return _run_target("pair", "train_pair_pipeline.py", argv)
    if normalize_task_type(task_type) in SEQUENCE_TASK_TYPES:
        return _run_target("sequence", "train_sequence_pipeline.py", argv)
    supported = ", ".join(sorted(SEQUENCE_TASK_TYPES | PAIR_TASK_TYPES))
    raise SystemExit(f"Unsupported task_type={task_type!r}. Supported tasks: {supported}.")



from pepplatform.stream_compat import register_legacy_cli_aliases, install_legacy_exports as install_legacy_stream_exports

install_legacy_stream_exports(globals())


if __name__ == "__main__":
    raise SystemExit(main() or 0)
