import asyncio
import json

import pytest
import torch

fastapi = pytest.importorskip("fastapi")
HTTPException = fastapi.HTTPException

from pepplatform.web.app import create_app
from pepplatform.web.paths import RuntimePaths
from pepplatform.web.store import MetadataStore
from pepplatform.svm_resume import save_svm_resume_bundle


@pytest.fixture(autouse=True)
def _disable_auth_for_legacy_web_tests(monkeypatch):
    monkeypatch.setenv("PEP_AUTH_ENABLED", "0")


def _create_app(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    return create_app(project_root=project_root, runtime_root=tmp_path / "runtime")


def _create_experiment(app, status="succeeded", name_kind=None, name_qualifier=None):
    store = app.state.store
    paths = app.state.paths
    dataset = store.create_dataset(
        "demo", "pep_function", "upload", "sequence_dir", paths.datasets / "demo"
    )
    experiment_id = "experiment-delete"
    experiment_dir = paths.experiments / experiment_id
    artifact_dir = experiment_dir / "artifact"
    artifact_dir.mkdir(parents=True)
    (experiment_dir / "train.log").write_text("done\n", encoding="utf-8")
    experiment = store.create_experiment(
        "demo run",
        dataset["id"],
        "pep_function",
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
        experiment_dir / "train.log",
        artifact_dir,
        experiment_id=experiment_id,
        name_kind=name_kind,
        name_qualifier=name_qualifier,
    )
    model = store.create_model(experiment["id"], "demo model", "pep_function", artifact_dir, True)
    store.update_experiment(experiment["id"], status=status, model_id=model["id"])
    return dataset, experiment, model, experiment_dir


def _delete_experiment_endpoint(app):
    for route in app.routes:
        if getattr(route, "path", "") == "/api/experiments/{experiment_id}" and "DELETE" in getattr(
            route, "methods", set()
        ):
            return route.endpoint
    raise AssertionError("delete experiment endpoint was not registered")


def _update_experiment_endpoint(app):
    for route in app.routes:
        if getattr(route, "path", "") == "/api/experiments/{experiment_id}" and "PATCH" in getattr(
            route, "methods", set()
        ):
            return route.endpoint
    raise AssertionError("update experiment endpoint was not registered")


def _list_experiments_endpoint(app):
    for route in app.routes:
        if getattr(route, "path", "") == "/api/experiments" and "GET" in getattr(
            route, "methods", set()
        ):
            return route.endpoint
    raise AssertionError("list experiments endpoint was not registered")


def _resume_downstream_endpoint(app):
    for route in app.routes:
        if getattr(
            route, "path", ""
        ) == "/api/experiments/{experiment_id}/resume-downstream" and "POST" in getattr(
            route, "methods", set()
        ):
            return route.endpoint
    raise AssertionError("resume downstream endpoint was not registered")


def test_resume_downstream_svm_creates_lineage_linked_child(tmp_path, monkeypatch):
    app = _create_app(tmp_path)
    try:
        store = app.state.store
        paths = app.state.paths
        dataset_root = paths.datasets / "svm-source"
        dataset_root.mkdir(parents=True)
        dataset = store.create_dataset(
            "svm source", "pep_function", "upload", "sequence_dir", dataset_root
        )
        source_dir = paths.experiments / "svm-source"
        artifact_dir = source_dir / "artifact"
        artifact_dir.mkdir(parents=True)
        source = store.create_experiment(
            "svm run",
            dataset["id"],
            "pep_function",
            {
                "tokenizer": "esm2",
                "encoder": "esm2",
                "model": "svm",
                "classifier": "svm",
                "training_mode": "export_embeddings",
                "svm_max_iter": 2000,
                "ml_reduction": "kpca",
                "ml_reduction_dim": 2,
                "kpca_kernel": "rbf",
                "kpca_gamma": 0.1,
            },
            source_dir / "train.log",
            artifact_dir,
            experiment_id="svm-source",
        )
        save_svm_resume_bundle(
            artifact_dir,
            train_x=torch.ones((4, 3)),
            train_y=torch.tensor([[0], [1], [0], [1]]),
            val_x=torch.ones((2, 3)),
            val_y=torch.tensor([[0], [1]]),
            task_type="pep_function",
            artifact_source="sequence_embedding_features",
        )
        store.update_experiment(
            source["id"],
            status="failed",
            failure={
                "schema_version": 1,
                "code": "training_data_invalid",
                "category": "data",
                "stage": "training",
                "message": "SVM predictor contract requires shared parameters across outputs",
                "context": {},
            },
        )
        listed_source = next(
            item for item in _list_experiments_endpoint(app)() if item["id"] == source["id"]
        )
        assert listed_source["failure"]["code"] == "svm_artifact_contract_failed"
        assert listed_source["failure"]["stage"] == "artifact_export"
        assert listed_source["resume_capabilities"]["downstream_svm"]["available"] is True
        monkeypatch.setattr(
            "pepplatform.web.app.preflight_experiment",
            lambda dataset, task_type, config: {"valid": True, "normalized_config": {**config}},
        )
        submitted = []
        monkeypatch.setattr(app.state.runner, "submit", submitted.append)

        child = _resume_downstream_endpoint(app)(
            source["id"],
            {
                "overrides": {
                    "svm_max_iter": 2000,
                    "svm_convergence_policy": "bounded_retry",
                    "kpca_kernel": "linear",
                }
            },
        )

        assert child["id"] != source["id"]
        assert child["resume"]["source_experiment_id"] == source["id"]
        assert child["resume"]["stage"] == "downstream_svm"
        assert child["config"]["svm_max_iter"] == 2000
        assert child["config"]["kpca_kernel"] == "linear"
        assert child["config"]["kpca_gamma"] is None
        assert submitted == [child["id"]]
        assert store.get_experiment(source["id"])["status"] == "failed"
    finally:
        app.state.runner.stop()


def _delete_dataset_endpoint(app):
    for route in app.routes:
        if getattr(route, "path", "") == "/api/datasets/{dataset_id}" and "DELETE" in getattr(
            route, "methods", set()
        ):
            return route.endpoint
    raise AssertionError("delete dataset endpoint was not registered")


def _cancel_experiment_endpoint(app):
    for route in app.routes:
        if getattr(
            route, "path", ""
        ) == "/api/experiments/{experiment_id}/cancel" and "POST" in getattr(
            route, "methods", set()
        ):
            return route.endpoint
    raise AssertionError("cancel experiment endpoint was not registered")


def _analysis_endpoint(app):
    for route in app.routes:
        if getattr(
            route, "path", ""
        ) == "/api/experiments/{experiment_id}/analysis" and "GET" in getattr(
            route, "methods", set()
        ):
            return route.endpoint
    raise AssertionError("analysis endpoint was not registered")


def _predict_endpoint(app):
    for route in app.routes:
        if getattr(route, "path", "") == "/api/models/{model_id}/predict" and "POST" in getattr(
            route, "methods", set()
        ):
            return route.endpoint
    raise AssertionError("predict endpoint was not registered")


def _predict_stream_endpoint(app):
    for route in app.routes:
        if getattr(
            route, "path", ""
        ) == "/api/models/{model_id}/predict/stream" and "POST" in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError("predict stream endpoint was not registered")


def _readiness_endpoint(app):
    for route in app.routes:
        if getattr(
            route, "path", ""
        ) == "/api/models/{model_id}/inference-readiness" and "GET" in getattr(
            route, "methods", set()
        ):
            return route.endpoint
    raise AssertionError("inference readiness endpoint was not registered")


def _update_model_endpoint(app):
    for route in app.routes:
        if getattr(route, "path", "") == "/api/models/{model_id}" and "PATCH" in getattr(
            route, "methods", set()
        ):
            return route.endpoint
    raise AssertionError("update model endpoint was not registered")


def _list_inference_runs_endpoint(app):
    for route in app.routes:
        if getattr(
            route, "path", ""
        ) == "/api/models/{model_id}/inference-runs" and "GET" in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError("list inference runs endpoint was not registered")


def _download_inference_run_endpoint(app):
    for route in app.routes:
        if getattr(route, "path", "") == "/api/inference-runs/{run_id}/csv" and "GET" in getattr(
            route, "methods", set()
        ):
            return route.endpoint
    raise AssertionError("download inference run endpoint was not registered")


def _delete_inference_run_endpoint(app):
    for route in app.routes:
        if getattr(route, "path", "") == "/api/inference-runs/{run_id}" and "DELETE" in getattr(
            route, "methods", set()
        ):
            return route.endpoint
    raise AssertionError("delete inference run endpoint was not registered")


def _leaderboard_endpoint(app):
    for route in app.routes:
        if getattr(route, "path", "") == "/api/leaderboard" and "GET" in getattr(
            route, "methods", set()
        ):
            return route.endpoint
    raise AssertionError("leaderboard endpoint was not registered")


class _FakeModelService:
    def predict(self, model_id, sequences=None, pairs=None):
        if pairs:
            return {
                "model_id": model_id,
                "labels": ["interaction"],
                "predictions": [
                    {
                        "peptide": pair["peptide"],
                        "protein": pair["protein"],
                        "score": 0.75,
                        "score_kind": "probability",
                        "threshold": 0.5,
                        "is_interaction": True,
                        "prediction_label": "Interact",
                        "scores": [0.75],
                    }
                    for pair in pairs
                ],
            }
        return {
            "model_id": model_id,
            "labels": ["active"],
            "predictions": [
                {"sequence": sequence, "scores": [0.8]} for sequence in sequences or []
            ],
        }


class _StreamingFakeModelService(_FakeModelService):
    def predict(self, model_id, sequences=None, pairs=None, progress_callback=None):
        if progress_callback:
            progress_callback({"stage": "extracting_features", "completed": 1, "total": 1})
        response = super().predict(model_id, sequences=sequences, pairs=pairs)
        if progress_callback:
            progress_callback({"stage": "running_predictor", "completed": 1, "total": 1})
        return response


def test_list_experiments_materializes_effective_defaults_for_legacy_records(tmp_path):
    app = _create_app(tmp_path)
    try:
        _dataset, experiment, _model, _experiment_dir = _create_experiment(app)

        listed = _list_experiments_endpoint(app)()
        record = next(item for item in listed if item["id"] == experiment["id"])

        assert record["requested_config"] == {
            "tokenizer": "kmer",
            "encoder": "index",
            "model": "cnn",
        }
        assert record["config"]["batch_size"] == 32
        assert record["config"]["lr"] == 1e-3
        assert record["config"]["epochs"] == 30
        assert record["config_parameter_sources"]["batch_size"] == "default"
        assert record["effective_route"]["route_id"] == "sequence.classic_positional"
        assert record["effective_route"]["predictor"]["input_adapter"]["id"] == "learned_embedding"
    finally:
        app.state.runner.stop()


def test_predict_records_inference_run_and_downloads_csv(tmp_path):
    app = _create_app(tmp_path)
    try:
        _dataset, _experiment, model, _experiment_dir = _create_experiment(app)
        app.state.models = _FakeModelService()

        response = _predict_endpoint(app)(model["id"], {"sequences": ["ACD", "GG"]})

        assert response["run_id"]
        runs = _list_inference_runs_endpoint(app)(model["id"])
        assert len(runs) == 1
        assert runs[0]["input_kind"] == "sequences"
        assert runs[0]["input_count"] == 2
        assert runs[0]["prediction_count"] == 2

        csv_response = _download_inference_run_endpoint(app)(response["run_id"])
        csv_text = csv_response.body.decode("utf-8")
        assert "input_index,sequence,top_label,top_score,score_active" in csv_text
        assert "1,ACD,active,0.8,0.8" in csv_text
    finally:
        app.state.runner.stop()


def test_predict_stream_emits_stages_and_saves_history(tmp_path):
    app = _create_app(tmp_path)
    try:
        _dataset, _experiment, model, _experiment_dir = _create_experiment(app)
        app.state.models = _StreamingFakeModelService()

        response = _predict_stream_endpoint(app)(model["id"], {"sequences": ["ACD"]})

        async def collect():
            return [chunk async for chunk in response.body_iterator]

        chunks = asyncio.run(collect())
        text = b"".join(
            chunk if isinstance(chunk, bytes) else chunk.encode() for chunk in chunks
        ).decode()

        assert "event: stage" in text
        assert '"stage":"extracting_features"' in text
        assert "event: done" in text
        assert app.state.store.list_inference_runs(model["id"])[0]["input_count"] == 1
    finally:
        app.state.runner.stop()


def test_predict_records_pair_inference_csv(tmp_path):
    app = _create_app(tmp_path)
    try:
        _dataset, _experiment, model, _experiment_dir = _create_experiment(app)
        app.state.models = _FakeModelService()

        response = _predict_endpoint(app)(
            model["id"], {"pairs": [{"peptide": "AC", "protein": "GG"}]}
        )
        csv_response = _download_inference_run_endpoint(app)(response["run_id"])
        csv_text = csv_response.body.decode("utf-8")

        assert (
            "input_index,peptide,protein,prediction_label,score,score_kind,threshold,is_interaction,score_interaction"
            in csv_text
        )
        assert "1,AC,GG,Interact,0.75,probability,0.5,True,0.75" in csv_text
    finally:
        app.state.runner.stop()


def test_delete_inference_run_removes_history_and_csv(tmp_path):
    app = _create_app(tmp_path)
    try:
        _dataset, _experiment, model, _experiment_dir = _create_experiment(app)
        app.state.models = _FakeModelService()
        response = _predict_endpoint(app)(model["id"], {"sequences": ["ACD"]})

        deleted = _delete_inference_run_endpoint(app)(response["run_id"])

        assert deleted == {"run_id": response["run_id"], "deleted": True}
        assert _list_inference_runs_endpoint(app)(model["id"]) == []
        with pytest.raises(HTTPException) as exc_info:
            _download_inference_run_endpoint(app)(response["run_id"])
        assert exc_info.value.status_code == 404
    finally:
        app.state.runner.stop()


def test_predict_profile_model_returns_409_instead_of_500(tmp_path):
    app = _create_app(tmp_path)
    try:
        store = app.state.store
        paths = app.state.paths
        dataset = store.create_dataset(
            "demo", "pep_function", "upload", "sequence_dir", paths.datasets / "demo"
        )
        experiment_dir = paths.experiments / "profile-predict"
        artifact_dir = experiment_dir / "artifact"
        artifact_dir.mkdir(parents=True)
        metadata = {
            "bundle_type": "sequence_classical",
            "supports_inference": True,
            "args": {"tokenizer": "kmer", "encoder": "pssm", "model": "rf"},
        }
        (artifact_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        experiment = store.create_experiment(
            "profile run",
            dataset["id"],
            "pep_function",
            {"tokenizer": "kmer", "encoder": "pssm", "model": "rf"},
            experiment_dir / "train.log",
            artifact_dir,
        )
        model = store.create_model(
            experiment["id"], "profile model", "pep_function", artifact_dir, True, metadata
        )

        with pytest.raises(HTTPException) as exc_info:
            _predict_endpoint(app)(model["id"], {"sequences": ["FLPILGNLLSGL"]})

        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["code"] == "profile_features_require_sample_id"
        assert "PSSM/PSFM" in exc_info.value.detail["message"]
    finally:
        app.state.runner.stop()


def test_inference_readiness_reports_standard_model_ready(tmp_path):
    app = _create_app(tmp_path)
    try:
        _dataset, _experiment, model, experiment_dir = _create_experiment(app)
        artifact_dir = experiment_dir / "artifact"
        (artifact_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "bundle_type": "sequence_nn",
                    "supports_inference": True,
                    "args": {"tokenizer": "kmer", "encoder": "onehot", "model": "cnn"},
                }
            ),
            encoding="utf-8",
        )

        report = _readiness_endpoint(app)(model["id"])

        assert report["ready"] is True
        assert report["profile_required"] is False
        assert report["status"] == "not_required"
    finally:
        app.state.runner.stop()


def test_predict_rejects_invalid_pair_payload(tmp_path):
    app = _create_app(tmp_path)
    try:
        _dataset, _experiment, model, _experiment_dir = _create_experiment(app)

        with pytest.raises(HTTPException) as exc_info:
            _predict_endpoint(app)(model["id"], {"pairs": {"peptide": "AC", "protein": "GG"}})

        assert exc_info.value.status_code == 400
        assert "pairs must be a list" in exc_info.value.detail
    finally:
        app.state.runner.stop()


def test_update_model_renames_model_without_renaming_experiment(tmp_path):
    app = _create_app(tmp_path)
    try:
        _dataset, experiment, model, _experiment_dir = _create_experiment(app)

        response = _update_model_endpoint(app)(model["id"], {"name": "  GRU · kmer/pssm  "})

        assert response["name"] == "GRU · kmer/pssm"
        assert app.state.store.get_model(model["id"])["name"] == "GRU · kmer/pssm"
        assert app.state.store.get_experiment(experiment["id"])["name"] == "demo run"
    finally:
        app.state.runner.stop()


def test_update_experiment_renames_inherited_model_and_clears_generated_name(tmp_path):
    app = _create_app(tmp_path)
    try:
        _dataset, experiment, model, _experiment_dir = _create_experiment(
            app,
            name_kind="config",
            name_qualifier="balanced",
        )
        original_updated_at = app.state.store.get_experiment(experiment["id"])["updated_at"]

        response = _update_experiment_endpoint(app)(
            experiment["id"], {"name": "  Regularized MLP  "}
        )

        assert response["name"] == "Regularized MLP"
        assert response["generated_name"] is None
        assert response["updated_at"] == original_updated_at
        assert app.state.store.get_model(model["id"])["name"] == "Regularized MLP"
        assert app.state.store.get_model(model["id"])["name_is_custom"] is False
    finally:
        app.state.runner.stop()


@pytest.mark.parametrize(
    "name, code",
    [
        ("   ", "experiment_name_required"),
        ("x" * 121, "experiment_name_too_long"),
        ("bad\nname", "experiment_name_invalid"),
    ],
)
def test_update_experiment_rejects_invalid_names(tmp_path, name, code):
    app = _create_app(tmp_path)
    try:
        _dataset, experiment, _model, _experiment_dir = _create_experiment(app)

        with pytest.raises(HTTPException) as exc_info:
            _update_experiment_endpoint(app)(experiment["id"], {"name": name})

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["code"] == code
    finally:
        app.state.runner.stop()


def test_update_experiment_returns_404_for_missing_id(tmp_path):
    app = _create_app(tmp_path)
    try:
        with pytest.raises(HTTPException) as exc_info:
            _update_experiment_endpoint(app)("missing", {"name": "Renamed"})

        assert exc_info.value.status_code == 404
    finally:
        app.state.runner.stop()


def test_update_model_rejects_empty_name(tmp_path):
    app = _create_app(tmp_path)
    try:
        _dataset, _experiment, model, _experiment_dir = _create_experiment(app)

        with pytest.raises(HTTPException) as exc_info:
            _update_model_endpoint(app)(model["id"], {"name": "   "})

        assert exc_info.value.status_code == 400
    finally:
        app.state.runner.stop()


def test_update_model_returns_404_for_missing_id(tmp_path):
    app = _create_app(tmp_path)
    try:
        with pytest.raises(HTTPException) as exc_info:
            _update_model_endpoint(app)("missing", {"name": "Renamed"})

        assert exc_info.value.status_code == 404
    finally:
        app.state.runner.stop()


def test_delete_experiment_removes_record_models_and_artifacts(tmp_path):
    app = _create_app(tmp_path)
    try:
        dataset, experiment, model, experiment_dir = _create_experiment(app)

        response = _delete_experiment_endpoint(app)(experiment["id"])

        assert response["experiment_id"] == experiment["id"]
        assert app.state.store.get_experiment(experiment["id"]) is None
        assert app.state.store.get_model(model["id"]) is None
        assert app.state.store.get_dataset(dataset["id"]) is not None
        assert not experiment_dir.exists()
    finally:
        app.state.runner.stop()


def test_delete_dataset_removes_uploaded_record_and_managed_artifact(tmp_path):
    app = _create_app(tmp_path)
    try:
        dataset_root = app.state.paths.datasets / "uploaded-dataset"
        dataset_root.mkdir(parents=True)
        (dataset_root / "train.csv").write_text("sequence,label\nACD,1\n", encoding="utf-8")
        dataset = app.state.store.upsert_dataset(
            "uploaded-dataset",
            "Uploaded",
            "pep_function",
            "upload",
            "sequence_dir",
            dataset_root,
            {"managed": True},
        )

        response = _delete_dataset_endpoint(app)(dataset["id"])

        assert response == {"dataset_id": dataset["id"], "artifact_deleted": True}
        assert app.state.store.get_dataset(dataset["id"]) is None
        assert not dataset_root.exists()
    finally:
        app.state.runner.stop()


def test_delete_dataset_returns_404_for_missing_id(tmp_path):
    app = _create_app(tmp_path)
    try:
        with pytest.raises(HTTPException) as exc_info:
            _delete_dataset_endpoint(app)("missing")

        assert exc_info.value.status_code == 404
    finally:
        app.state.runner.stop()


def test_delete_dataset_rejects_builtin_dataset(tmp_path):
    app = _create_app(tmp_path)
    try:
        dataset = app.state.store.upsert_dataset(
            "builtin-demo",
            "Built-in",
            "pep_function",
            "builtin",
            "sequence_dir",
            app.state.paths.project_root / "datasets" / "demo",
            {"managed": False},
        )

        with pytest.raises(HTTPException) as exc_info:
            _delete_dataset_endpoint(app)(dataset["id"])

        assert exc_info.value.status_code == 409
        assert app.state.store.get_dataset(dataset["id"]) is not None
    finally:
        app.state.runner.stop()


def test_delete_dataset_rejects_referenced_upload(tmp_path):
    app = _create_app(tmp_path)
    try:
        dataset, _experiment, _model, _experiment_dir = _create_experiment(app)
        dataset_root = app.state.paths.datasets / dataset["id"]
        dataset_root.mkdir(parents=True)

        with pytest.raises(HTTPException) as exc_info:
            _delete_dataset_endpoint(app)(dataset["id"])

        assert exc_info.value.status_code == 409
        assert app.state.store.get_dataset(dataset["id"]) is not None
        assert dataset_root.exists()
    finally:
        app.state.runner.stop()


def test_delete_dataset_unreferenced_external_upload_deletes_record_only(tmp_path):
    app = _create_app(tmp_path)
    try:
        external_root = app.state.paths.project_root / "datasets" / "external"
        external_root.mkdir(parents=True)
        dataset = app.state.store.create_dataset(
            "External upload",
            "pep_function",
            "upload",
            "sequence_dir",
            external_root,
            {"managed": False},
        )

        response = _delete_dataset_endpoint(app)(dataset["id"])

        assert response == {"dataset_id": dataset["id"], "artifact_deleted": False}
        assert app.state.store.get_dataset(dataset["id"]) is None
        assert external_root.exists()
    finally:
        app.state.runner.stop()


def test_delete_experiment_returns_404_for_missing_id(tmp_path):
    app = _create_app(tmp_path)
    try:
        with pytest.raises(HTTPException) as exc_info:
            _delete_experiment_endpoint(app)("missing")

        assert exc_info.value.status_code == 404
    finally:
        app.state.runner.stop()


def test_experiment_analysis_endpoint_reads_artifact(tmp_path):
    app = _create_app(tmp_path)
    try:
        _dataset, experiment, _model, experiment_dir = _create_experiment(app)
        payload = {
            "schema_version": 1,
            "curves": {
                "validation": {
                    "roc": {"available": True, "auc": 1.0, "points": [{"fpr": 0.0, "tpr": 0.0}]},
                    "pr": {
                        "available": True,
                        "aupr": 1.0,
                        "points": [{"recall": 1.0, "precision": 1.0}],
                    },
                }
            },
        }
        (experiment_dir / "artifact" / "analysis.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

        response = _analysis_endpoint(app)(experiment["id"])

        assert response["schema_version"] == 1
        assert response["curves"]["validation"]["roc"]["auc"] == 1.0
        assert response["projection_3d"]["available"] is False
    finally:
        app.state.runner.stop()


def test_experiment_analysis_endpoint_returns_empty_for_missing_artifact(tmp_path):
    app = _create_app(tmp_path)
    try:
        _dataset, experiment, _model, _experiment_dir = _create_experiment(app)

        response = _analysis_endpoint(app)(experiment["id"])

        assert response["schema_version"] == 6
        assert response["curves"] == {}
        assert response["projection_2d"]["available"] is False
        assert response["projection_3d"]["available"] is False
    finally:
        app.state.runner.stop()


def test_leaderboard_endpoint_returns_sorted_rows_and_filters(tmp_path):
    app = _create_app(tmp_path)
    try:
        store = app.state.store
        paths = app.state.paths
        dataset = store.create_dataset(
            "leaderboard", "pep_function", "upload", "sequence_dir", paths.datasets / "lb"
        )
        first = store.create_experiment(
            "first",
            dataset["id"],
            "pep_function",
            {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
            paths.experiments / "first.log",
            paths.experiments / "first",
        )
        second = store.create_experiment(
            "second",
            dataset["id"],
            "pep_function",
            {"tokenizer": "kmer", "encoder": "index", "model": "rf"},
            paths.experiments / "second.log",
            paths.experiments / "second",
        )
        store.append_metric_event(first["id"], {"kind": "test", "test_f1": 0.70, "test_auc": 0.81})
        store.append_metric_event(
            second["id"], {"kind": "validation", "val_f1": 0.82, "val_auc": 0.76}
        )
        first_model = store.create_model(
            first["id"], "first model", "pep_function", paths.models / "first", True
        )
        second_model = store.create_model(
            second["id"], "second model", "pep_function", paths.models / "second", False
        )
        store.update_experiment(first["id"], status="succeeded", model_id=first_model["id"])
        store.update_experiment(second["id"], status="succeeded", model_id=second_model["id"])

        response = _leaderboard_endpoint(app)(
            task_type="pep_function",
            dataset_id=dataset["id"],
            metric="f1",
            split="auto",
            supports_inference=None,
            status="succeeded",
            limit=20,
        )
        ready_response = _leaderboard_endpoint(app)(
            task_type="",
            dataset_id="",
            metric="f1",
            split="auto",
            supports_inference=True,
            status="succeeded",
            limit=20,
        )

        assert response["total"] == 2
        assert [row["experiment"]["name"] for row in response["rows"]] == ["second", "first"]
        assert response["rows"][0]["score"] == 0.82
        assert response["rows"][0]["score_source"] == "val_f1"
        assert response["rows"][0]["model"]["supports_inference"] is False
        assert [row["experiment"]["name"] for row in ready_response["rows"]] == ["first"]
    finally:
        app.state.runner.stop()


def test_leaderboard_endpoint_rejects_unknown_metric(tmp_path):
    app = _create_app(tmp_path)
    try:
        with pytest.raises(HTTPException) as exc_info:
            _leaderboard_endpoint(app)(
                task_type="",
                dataset_id="",
                metric="loss",
                split="auto",
                supports_inference=None,
                status="succeeded",
                limit=20,
            )

        assert exc_info.value.status_code == 400
    finally:
        app.state.runner.stop()


def test_cancel_experiment_marks_running_run_as_stopped(tmp_path):
    app = _create_app(tmp_path)
    try:
        _dataset, experiment, model, experiment_dir = _create_experiment(app, status="running")

        response = _cancel_experiment_endpoint(app)(experiment["id"])

        assert response["status"] == "canceled"
        stored = app.state.store.get_experiment(experiment["id"])
        assert stored["status"] == "canceled"
        assert stored["finished_at"]
        assert app.state.store.get_model(model["id"]) is not None
        assert experiment_dir.exists()
    finally:
        app.state.runner.stop()


def test_cancel_experiment_rejects_completed_run(tmp_path):
    app = _create_app(tmp_path)
    try:
        _dataset, experiment, _model, _experiment_dir = _create_experiment(app, status="succeeded")

        with pytest.raises(HTTPException) as exc_info:
            _cancel_experiment_endpoint(app)(experiment["id"])

        assert exc_info.value.status_code == 409
    finally:
        app.state.runner.stop()


def test_delete_experiment_allows_stopped_status(tmp_path):
    app = _create_app(tmp_path)
    try:
        dataset, experiment, model, experiment_dir = _create_experiment(app, status="canceled")

        response = _delete_experiment_endpoint(app)(experiment["id"])

        assert response["experiment_id"] == experiment["id"]
        assert app.state.store.get_experiment(experiment["id"]) is None
        assert app.state.store.get_model(model["id"]) is None
        assert app.state.store.get_dataset(dataset["id"]) is not None
        assert not experiment_dir.exists()
    finally:
        app.state.runner.stop()


def test_delete_experiment_rejects_non_terminal_status(tmp_path):
    app = _create_app(tmp_path)
    try:
        _dataset, experiment, model, experiment_dir = _create_experiment(app, status="running")

        with pytest.raises(HTTPException) as exc_info:
            _delete_experiment_endpoint(app)(experiment["id"])

        assert exc_info.value.status_code == 409
        assert app.state.store.get_experiment(experiment["id"]) is not None
        assert app.state.store.get_model(model["id"]) is not None
        assert experiment_dir.exists()
    finally:
        app.state.runner.stop()


@pytest.mark.parametrize(
    "task", ["pep_function", "pep_protein_interaction", "pep_hlaii", "pep_hlai"]
)
@pytest.mark.parametrize(
    "counts,expected",
    [
        ({}, (1, 8)),
        ({"semantic_positive_partners": 4, "semantic_zero_partners": 0}, (4, 0)),
    ],
)
def test_create_experiment_persists_semantic_counts_without_changing_source(
    tmp_path, monkeypatch, task, counts, expected
):
    app = _create_app(tmp_path)
    try:
        store, paths = app.state.store, app.state.paths
        dataset = store.create_dataset(
            "semantic data",
            task,
            "upload",
            "sequence_dir" if task == "pep_function" else "pair_split_dir",
            paths.datasets / "semantic",
            metadata={"splits": {"train": 10, "val": 2}},
        )
        original_config = {
            "tokenizer": "kmer",
            "encoder": "onehot",
            "model": "cnn",
            "pair_strategy": "single_stream",
            "enable_semantic_similarity": True,
        }
        source = store.create_experiment(
            "historical semantic run",
            dataset["id"],
            task,
            original_config,
            paths.experiments / "source/train.log",
            paths.experiments / "source/artifact",
        )
        submitted = []
        monkeypatch.setattr(app.state.runner, "submit", submitted.append)
        endpoint = next(
            route.endpoint
            for route in app.routes
            if getattr(route, "path", "") == "/api/experiments"
            and "POST" in getattr(route, "methods", set())
        )
        child = endpoint(
            {
                "dataset_id": dataset["id"],
                "task_type": task,
                "config": {**original_config, **counts},
            }
        )
        saved = store.get_experiment(child["id"])
        assert saved["config"]["semantic_positive_partners"] == expected[0]
        assert saved["config"]["semantic_zero_partners"] == expected[1]
        assert "semantic_loss_type" not in saved["config"]
        assert submitted == [child["id"]]
        assert child["id"] != source["id"] and child["artifact_dir"] != source["artifact_dir"]
        assert store.get_experiment(source["id"])["config"] == original_config
        command = app.state.runner.build_command(saved, dataset)
        assert command[command.index("--semantic_positive_partners") + 1] == str(expected[0])
        assert command[command.index("--semantic_zero_partners") + 1] == str(expected[1])
    finally:
        app.state.runner.stop()


@pytest.mark.parametrize("task,overrides", [
    ("pep_function", {}),
    ("pep_function", {"multilabel_loss": "auto"}),
    ("pep_function", {"multilabel_loss": "bce"}),
    ("pep_function", {"multilabel_loss": "balanced_bce", "multilabel_pos_weight_cap": 7}),
    ("pep_function", {"multilabel_loss": "asymmetric_focal", "asl_gamma_pos": 0, "asl_gamma_neg": 3, "asl_clip": 0}),
    *[(task, {"loss_type": loss, "pos_weight_max": 7}) for task in ("pep_protein_interaction", "pep_hlai", "pep_hlaii") for loss in ("bce", "wbce")],
])
def test_create_experiment_preserves_explicit_loss_and_historical_source(tmp_path, monkeypatch, task, overrides):
    from pepplatform.web.classification_losses import CLASSIFICATION_LOSS_FIELDS

    app = _create_app(tmp_path)
    try:
        store, paths = app.state.store, app.state.paths
        dataset = store.create_dataset(
            "loss data", task, "upload",
            "sequence_dir" if task == "pep_function" else "pair_split_dir",
            paths.datasets / "loss", metadata={"splits": {"train": 10, "val": 2}},
        )
        original = {"tokenizer": "kmer", "encoder": "onehot", "model": "cnn", "pair_strategy": "single_stream"}
        source = store.create_experiment(
            "historical run", dataset["id"], task, original,
            paths.experiments / "source/train.log", paths.experiments / "source/artifact",
        )
        submitted = []
        monkeypatch.setattr(app.state.runner, "submit", submitted.append)
        endpoint = next(route.endpoint for route in app.routes
                        if getattr(route, "path", "") == "/api/experiments"
                        and "POST" in getattr(route, "methods", set()))
        child = endpoint({"dataset_id": dataset["id"], "task_type": task, "config": {**original, **overrides}})
        saved = store.get_experiment(child["id"])
        actual = {key: saved["config"][key] for key in CLASSIFICATION_LOSS_FIELDS if key in saved["config"]}
        assert actual == overrides
        assert {key: child["requested_config"][key] for key in overrides} == overrides
        assert all(child["config_parameter_sources"][key] == "explicit" for key in overrides)
        assert submitted == [child["id"]]
        assert child["id"] != source["id"] and child["artifact_dir"] != source["artifact_dir"]
        assert store.get_experiment(source["id"])["config"] == original
    finally:
        app.state.runner.stop()


def test_app_startup_marks_stale_active_experiments_stopped(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    runtime_root = tmp_path / "runtime"
    paths = RuntimePaths.from_project_root(project_root, runtime_root).ensure()
    store = MetadataStore(paths.database)
    dataset = store.create_dataset(
        "demo", "pep_function", "upload", "sequence_dir", paths.datasets / "demo"
    )
    experiment_dir = paths.experiments / "stale-running"
    artifact_dir = experiment_dir / "artifact"
    artifact_dir.mkdir(parents=True)
    experiment = store.create_experiment(
        "stale run",
        dataset["id"],
        "pep_function",
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
        experiment_dir / "train.log",
        artifact_dir,
        experiment_id="stale-running",
    )
    store.update_experiment(experiment["id"], status="running")

    app = create_app(project_root=project_root, runtime_root=runtime_root)
    try:
        stopped = app.state.store.get_experiment(experiment["id"])
        assert stopped["status"] == "canceled"
        assert stopped["finished_at"]
        assert "restarted" in stopped["error"]
    finally:
        app.state.runner.stop()
