import json

from pepplatform.profile_generation import ProfileGenerationConfig
from pepplatform.sequence_profiles import (
    BLOSUM62_BACKGROUND_SHA256,
    BLOSUM62_MATRIX_SHA256,
    PROFILE_SCIENTIFIC_REFERENCES,
    PROFILE_HIT_SUMMARY_FIELDS,
    PROFILE_CONTRACT_VERSION,
    PROFILE_QUERY_COVERAGE_FORMULA,
    sha256_file,
)
from pepplatform.web import inference
from pepplatform.web.inference import ModelService
from pepplatform.web import profile_runtime


class _Store:
    def __init__(self, model):
        self.model = model

    def get_model(self, model_id):
        return self.model if self.model["id"] == model_id else None


def _current_profile_generation_metadata(subject_fasta):
    config = ProfileGenerationConfig(
        iterations=2,
        evalue=0.001,
        inclusion_evalue=0.001,
        threads=1,
    )
    provenance = {
        "profile_contract_version": PROFILE_CONTRACT_VERSION,
        "algorithm": "NCBI PSI-BLAST",
        "algorithm_lineage": "Altschul_et_al_1997",
        "implementation": "NCBI BLAST+",
        "blast_version": "psiblast: test",
        "makeblastdb_version": "makeblastdb: test",
        "scientific_parameters": config.scientific_parameters(),
        "runtime_parameters": config.runtime_parameters(),
        "database_scope": "train_only",
        "subject_fasta_sha256": sha256_file(subject_fasta),
        "subject_fasta_artifact": "subjects/test.fasta",
        "database": {"train_fasta_sha256": "b" * 64},
        "self_exclusion": ("all subject IDs with an identical normalized query sequence"),
        "sequence_normalization": {
            "uppercase": True,
            "strip_whitespace": True,
            "J_U_O_to_X": True,
            "B_Z_X_preserved": True,
            "non_letters": "error",
        },
        "blosum62_matrix_sha256": BLOSUM62_MATRIX_SHA256,
        "blosum62_background_sha256": BLOSUM62_BACKGROUND_SHA256,
        "scientific_references": [dict(reference) for reference in PROFILE_SCIENTIFIC_REFERENCES],
        "quality_summary": {
            "outfmt_fields": list(PROFILE_HIT_SUMMARY_FIELDS),
            "query_coverage": PROFILE_QUERY_COVERAGE_FORMULA,
            "iteration_scope": "final_completed_iteration",
            "hit_count_unit": "unique_subject",
        },
        "profile_semantics": {
            "method_name": "pssm",
            "matrix_layout": "query_position_by_amino_acid",
        },
        "audit_artifact": "profile_audit.json",
        "audit_sha256": "e" * 64,
        "audit_summary": {
            "sample_count": 1,
            "samples_with_zero_observations": 0,
            "zero_observation_position_count": 0,
            "zero_observation_fraction": 0.0,
            "final_iteration_unique_subject_count": 1,
        },
        "generation_fingerprint": "c" * 64,
        "feature_type": "pssm",
        "fallback_policy": "error",
    }
    return {
        "profile_contract_version": PROFILE_CONTRACT_VERSION,
        "iterations": config.iterations,
        "evalue": config.evalue,
        "inclusion_evalue": config.inclusion_evalue,
        "threads": config.threads,
        "scientific_parameters": config.scientific_parameters(),
        "on_missing": "error",
        "db_scope": "train_only",
        "subject_fasta_sha256": sha256_file(subject_fasta),
        "manifest_sha256": "d" * 64,
        "training_profile_provenance": provenance,
    }


def test_profile_environment_reports_missing_binaries(monkeypatch):
    profile_runtime.reset_profile_environment_cache()
    monkeypatch.setattr(profile_runtime.shutil, "which", lambda _name: None)

    report = profile_runtime.profile_environment_readiness()

    assert report["ready"] is False
    assert report["status"] == "blocked"
    assert {error["code"] for error in report["errors"]} == {"blast_binary_missing"}
    profile_runtime.reset_profile_environment_cache()


def test_profile_model_readiness_uses_prebuilt_database_without_makeblastdb(tmp_path, monkeypatch):
    profile_dir = tmp_path / "profile_generation"
    db_dir = profile_dir / "blastdb"
    db_dir.mkdir(parents=True)
    (profile_dir / "subject.fasta").write_text(">subject\nACDE\n", encoding="utf-8")
    db_prefix = db_dir / "profile_db"
    for suffix in [".pin", ".phr", ".psq"]:
        (db_dir / f"profile_db{suffix}").write_text("db", encoding="utf-8")
    metadata = {
        "bundle_type": "sequence_nn",
        "supports_inference": True,
        "args": {"tokenizer": "kmer", "encoder": "pssm", "model": "cnn"},
        "profile_generation": {
            **_current_profile_generation_metadata(profile_dir / "subject.fasta"),
            "subject_fasta": "profile_generation/subject.fasta",
            "db_prefix": "profile_generation/blastdb/profile_db",
        },
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(
        inference,
        "profile_environment_readiness",
        lambda: {
            "ready": False,
            "status": "blocked",
            "checked_at": "now",
            "errors": [{"code": "blast_binary_missing"}],
            "checks": [
                {
                    "id": "makeblastdb",
                    "ok": False,
                    "status": "error",
                    "code": "blast_binary_missing",
                    "message": "makeblastdb missing",
                    "hint": "install",
                },
                {
                    "id": "psiblast",
                    "ok": True,
                    "status": "ok",
                    "message": "psiblast available",
                    "version": "psiblast 2.0",
                },
                {
                    "id": "profile_smoke_test",
                    "ok": False,
                    "status": "skipped",
                    "message": "skipped",
                },
            ],
        },
    )
    service = ModelService(
        _Store(
            {
                "id": "model",
                "bundle_path": str(tmp_path),
                "supports_inference": True,
            }
        )
    )

    report = service.readiness("model")

    assert report["ready"] is True
    assert report["status"] == "warning"
    assert report["mode"] == "prebuilt_db"
    assert any(
        check["id"] == "makeblastdb" and check["required"] is False for check in report["checks"]
    )


def test_profile_model_readiness_blocks_missing_subject(tmp_path, monkeypatch):
    profile_dir = tmp_path / "profile_generation"
    profile_dir.mkdir()
    metadata = {
        "bundle_type": "sequence_nn",
        "supports_inference": True,
        "args": {"tokenizer": "kmer", "encoder": "pssm", "model": "cnn"},
        "profile_generation": {
            "subject_fasta": "profile_generation/subject.fasta",
            "iterations": 2,
        },
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(
        inference,
        "profile_environment_readiness",
        lambda: {
            "ready": True,
            "status": "ready",
            "checked_at": "now",
            "errors": [],
            "checks": [
                {"id": "makeblastdb", "ok": True, "status": "ok", "message": "available"},
                {"id": "psiblast", "ok": True, "status": "ok", "message": "available"},
                {"id": "profile_smoke_test", "ok": True, "status": "ok", "message": "passed"},
            ],
        },
    )
    service = ModelService(
        _Store(
            {
                "id": "model",
                "bundle_path": str(tmp_path),
                "supports_inference": True,
            }
        )
    )

    report = service.readiness("model")

    assert report["ready"] is False
    assert report["status"] == "blocked"
    assert any(error["code"] == "profile_subject_missing" for error in report["errors"])
