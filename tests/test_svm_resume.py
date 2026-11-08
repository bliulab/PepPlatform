import torch

from pepplatform.svm_resume import (
    SVM_RESUME_RELATIVE_DIR,
    inspect_svm_resume_bundle,
    load_svm_resume_bundle,
    materialize_svm_resume_bundle,
    save_svm_resume_bundle,
)


def _save_bundle(artifact_dir):
    return save_svm_resume_bundle(
        artifact_dir,
        train_x=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        train_y=torch.tensor([[0], [1], [0], [1]]),
        val_x=torch.ones((2, 3)),
        val_y=torch.tensor([[0], [1]]),
        test_x=torch.zeros((2, 3)),
        test_y=torch.tensor([[1], [0]]),
        task_type="pep_function",
        artifact_source="sequence_embedding_features",
        artifact_info={"embedding_dim": 3},
    )


def test_svm_resume_bundle_round_trip_and_materialization(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    saved = _save_bundle(source)

    inspected = inspect_svm_resume_bundle(source)
    materialized = materialize_svm_resume_bundle(source, destination)
    manifest, tensors = load_svm_resume_bundle(destination)

    assert inspected["tensor_sha256"] == saved["tensor_sha256"]
    assert materialized["tensor_sha256"] == saved["tensor_sha256"]
    assert manifest["artifact_source"] == "sequence_embedding_features"
    torch.testing.assert_close(tensors["train_x"], torch.arange(12).reshape(4, 3).float())


def test_svm_resume_bundle_rejects_tampered_tensor_payload(tmp_path):
    artifact_dir = tmp_path / "artifact"
    _save_bundle(artifact_dir)
    tensor_path = artifact_dir / SVM_RESUME_RELATIVE_DIR / "embeddings.pt"
    with tensor_path.open("ab") as handle:
        handle.write(b"tampered")

    assert inspect_svm_resume_bundle(artifact_dir, verify_integrity=False) is not None
    assert inspect_svm_resume_bundle(artifact_dir) is None
