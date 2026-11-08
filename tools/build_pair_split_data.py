import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pepplatform.task_compat import normalize_pair_tool_key, resolve_hlai_root
from pepplatform.web.datasets import DatasetValidationError, normalize_builtin_pair_root
from pepplatform.web.paths import RuntimePaths


TASKS = {
    "ppi": {
        "id": "builtin-ppi",
        "task_type": "pep_protein_interaction",
        "root": Path("datasets/task_peptide_protein_interaction"),
    },
    "hlai": {
        "id": "builtin-hlai",
        "task_type": "pep_hlai",
        "root": Path("datasets/task_peptide_HLAI_interaction"),
    },
}


def build_task(project_root, task_key, overwrite=False):
    task_key = normalize_pair_tool_key(task_key)
    task = TASKS[task_key]
    source_root = (
        resolve_hlai_root(project_root) if task_key == "hlai" else project_root / task["root"]
    )
    target_root = source_root / "data"
    if target_root.exists() and not overwrite:
        print(f"[skip] {task_key}: {target_root} already exists; pass --overwrite to rebuild")
        return target_root
    paths = RuntimePaths.from_project_root(project_root, project_root / ".pepplatform_web").ensure()
    _, metadata = normalize_builtin_pair_root(
        {
            "id": task["id"],
            "task_type": task["task_type"],
            "root_path": source_root,
        },
        paths,
        target_root=target_root,
        source_root=source_root,
    )
    print(f"[ok] {task_key}: {target_root} splits={metadata['splits']}")
    return target_root


def main():
    parser = argparse.ArgumentParser(
        description="Build local split-directory data views for pair tasks."
    )
    parser.add_argument("--project_root", default=ROOT, type=Path)
    parser.add_argument(
        "--task", type=normalize_pair_tool_key, choices=["all", *TASKS], default="all"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    task_keys = list(TASKS) if args.task == "all" else [args.task]
    for task_key in task_keys:
        try:
            build_task(project_root, task_key, overwrite=args.overwrite)
        except DatasetValidationError as exc:
            raise SystemExit(f"[error] {task_key}: {exc}") from exc


if __name__ == "__main__":
    main()
