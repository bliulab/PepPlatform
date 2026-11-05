import shutil
import subprocess
import tempfile
import threading
import weakref
from datetime import datetime, timezone
from pathlib import Path

from pepplatform.profile_generation import (
    ProfileGenerationConfig,
    build_psiblast_command,
)
from pepplatform.sequence_profiles import (
    parse_ascii_pssm,
    parse_psiblast_psfm,
)


_ENVIRONMENT_LOCK = threading.Lock()
_ENVIRONMENT_REPORT = None
_RESOURCE_LOCKS = weakref.WeakValueDictionary()
_RESOURCE_LOCKS_GUARD = threading.Lock()


def profile_resource_lock(key):
    normalized = str(Path(key).resolve())
    with _RESOURCE_LOCKS_GUARD:
        lock = _RESOURCE_LOCKS.get(normalized)
        if lock is None:
            lock = threading.Lock()
            _RESOURCE_LOCKS[normalized] = lock
        return lock


def reset_profile_environment_cache():
    global _ENVIRONMENT_REPORT
    with _ENVIRONMENT_LOCK:
        _ENVIRONMENT_REPORT = None


def _version_check(name):
    path = shutil.which(name)
    if path is None:
        return {
            "id": name,
            "ok": False,
            "status": "error",
            "code": "blast_binary_missing",
            "message": f"{name} was not found in PATH",
            "hint": "Install ncbi-blast+ in the backend runtime.",
        }
    try:
        result = subprocess.run(
            [path, "-version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "id": name,
            "ok": False,
            "status": "error",
            "code": "blast_binary_unusable",
            "message": f"{name} could not be executed: {exc}",
            "hint": "Verify the ncbi-blast+ installation and executable permissions.",
        }
    version_output = "\n".join(part for part in [result.stdout, result.stderr] if part).strip()
    version = version_output.splitlines()[0].strip() if version_output else ""
    return {
        "id": name,
        "ok": True,
        "status": "ok",
        "path": path,
        "version": version,
        "message": f"{name} is available",
    }


def _smoke_test(makeblastdb_path, psiblast_path):
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    try:
        with tempfile.TemporaryDirectory(prefix="pepplatform_blast_probe_") as temp_dir:
            root = Path(temp_dir)
            subject = root / "subject.fasta"
            query = root / "query.fasta"
            db_prefix = root / "profile_db"
            pssm = root / "query.pssm"
            checkpoint = root / "query.asn1"
            hits = root / "query.hits.tsv"
            subject.write_text(f">subject_1\n{sequence}{sequence}\n", encoding="utf-8")
            query.write_text(f">query_1\n{sequence}\n", encoding="utf-8")
            subprocess.run(
                [
                    makeblastdb_path,
                    "-in",
                    str(subject),
                    "-dbtype",
                    "prot",
                    "-out",
                    str(db_prefix),
                    "-parse_seqids",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            command = build_psiblast_command(
                query_fasta=query,
                database_prefix=db_prefix,
                ascii_pssm=pssm,
                checkpoint=checkpoint,
                hits=hits,
                config=ProfileGenerationConfig(iterations=2),
            )
            command[0] = psiblast_path
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if not pssm.exists():
                raise FileNotFoundError("PSI-BLAST completed without producing an ASCII PSSM")
            if not checkpoint.exists():
                raise FileNotFoundError("PSI-BLAST completed without producing an ASN.1 checkpoint")
            parse_ascii_pssm(pssm, sequence)
            parse_psiblast_psfm(
                checkpoint,
                expected_sequence=sequence,
                ascii_pssm=pssm,
            )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {
            "id": "profile_smoke_test",
            "ok": False,
            "status": "error",
            "code": "blast_probe_failed",
            "message": f"BLAST+ profile smoke test failed: {exc}",
            "hint": "Run makeblastdb and psiblast manually and verify that ASCII PSSM output is supported.",
        }
    return {
        "id": "profile_smoke_test",
        "ok": True,
        "status": "ok",
        "message": (
            "BLAST+ built a temporary database and produced valid PSSM and "
            "weighted-observed PSFM outputs"
        ),
    }


def profile_environment_readiness(force=False):
    global _ENVIRONMENT_REPORT
    with _ENVIRONMENT_LOCK:
        if _ENVIRONMENT_REPORT is not None and not force:
            return _ENVIRONMENT_REPORT

        makeblastdb = _version_check("makeblastdb")
        psiblast = _version_check("psiblast")
        checks = [makeblastdb, psiblast]
        if makeblastdb["ok"] and psiblast["ok"]:
            checks.append(_smoke_test(makeblastdb["path"], psiblast["path"]))
        else:
            checks.append(
                {
                    "id": "profile_smoke_test",
                    "ok": False,
                    "status": "skipped",
                    "code": "blast_probe_skipped",
                    "message": "BLAST+ profile smoke test was skipped because a required binary is missing",
                }
            )

        errors = [
            {
                "code": check.get("code", "profile_environment_failed"),
                "message": check.get("message", ""),
                "hint": check.get("hint", ""),
            }
            for check in checks
            if check.get("status") == "error"
        ]
        _ENVIRONMENT_REPORT = {
            "ready": not errors,
            "status": "ready" if not errors else "blocked",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "errors": errors,
        }
        return _ENVIRONMENT_REPORT
