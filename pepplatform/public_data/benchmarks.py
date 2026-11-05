"""Read-only overlap reports for explicitly supplied reference indexes."""

from pepplatform.task_compat import normalize_task_type

from .common import DataError, canonical, digest


def sample_key(task, peptide, partner_id=""):
    return digest(canonical([task, peptide, partner_id]))


def overlap_report(rows, index, task):
    task = normalize_task_type(task)
    if normalize_task_type(index.get("task")) != normalize_task_type(task) or index.get(
        "status"
    ) not in {"checked", "not_available"}:
        raise DataError("invalid benchmark index")
    peptides = {digest(row["peptide"].encode()) for row in rows}
    samples = {sample_key(index["task"], row["peptide"], row["partner_id"]) for row in rows}
    return {
        "status": index["status"],
        "independence_claimed": False,
        "homology_checked": False,
        "files": [
            {
                "path": item["path"],
                "sha256": item["sha256"],
                "exact_peptide_overlap": len(peptides & set(item["peptides"])),
                "exact_sample_overlap": len(samples & set(item["samples"])),
            }
            for item in index["files"]
        ],
    }
