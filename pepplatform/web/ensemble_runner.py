"""Application-injected extension; the shared ExperimentRunner stays intact."""

import sys
from pathlib import Path

from .ensemble_data import PROVENANCE_FILE, file_digest, read_json, validation_identity, write_json
from .ensemble_service import EnsembleService
from .ensemble_math import EnsembleError
from .runner import ExperimentRunner


class WebExperimentRunner(ExperimentRunner):
    def _run_experiment(self, experiment_id):
        experiment = self.store.get_experiment(experiment_id)
        identity = None
        if (
            experiment
            and experiment["status"] == "queued"
            and not experiment["config"].get("web_ensemble")
        ):
            # Capture provenance before training, never retroactively certify an
            # older model from whatever dataset happens to exist today.
            try:
                identity = validation_identity(self.store.get_dataset(experiment["dataset_id"]))
            except (OSError, ValueError, KeyError, TypeError):
                pass  # Unsupported provenance must not change old training behavior.
        self._web_training_identity = identity
        try:
            return super()._run_experiment(experiment_id)
        finally:
            self._web_training_identity = None

    def build_command(self, experiment, dataset):
        definition = experiment.get("config", {}).get("web_ensemble")
        if not definition:
            return super().build_command(experiment, dataset)
        user = (
            self.store.get_user(experiment["owner_user_id"])
            if experiment.get("owner_user_id")
            else None
        )
        if experiment.get("owner_user_id") and (not user or user.get("status") != "active"):
            raise EnsembleError(
                "ensemble_owner_unavailable", "The ensemble owner is no longer active"
            )
        request = EnsembleService(self.store, self.paths).prepare(
            {**definition, "dataset_id": dataset["id"]}, user=user
        )
        request_path = Path(experiment["artifact_dir"]).parent / "ensemble_request.json"
        write_json(request_path, request)
        return [
            sys.executable,
            "-m",
            "pepplatform.web.ensemble_worker",
            "--request",
            str(request_path),
            "--artifact-dir",
            experiment["artifact_dir"],
        ]

    def _register_model_if_present(self, experiment_id, experiment):
        artifact = Path(experiment["artifact_dir"])
        if experiment.get("config", {}).get("web_ensemble"):
            metadata = read_json(artifact / "metadata.json")
            if not read_json(artifact / "audit.json").get("passed"):
                raise ValueError("Ensemble artifact audit did not pass")
            evaluation = read_json(artifact / "evaluation.json")
            for split in ("validation", "test"):
                values = evaluation.get(split)
                if values:
                    prefix = "val" if split == "validation" else "test"
                    self.store.append_metric_event(
                        experiment_id,
                        {
                            "kind": split,
                            **{
                                f"{prefix}_{key}": value
                                for key, value in values.items()
                                if isinstance(value, (int, float))
                            },
                        },
                    )
            return self.store.create_model(
                experiment_id, experiment["name"], experiment["task_type"], artifact, True, metadata
            )
        record = super()._register_model_if_present(experiment_id, experiment)
        identity = getattr(self, "_web_training_identity", None)
        if identity is not None and record and record.get("supports_inference"):
            try:
                dataset = self.store.get_dataset(experiment["dataset_id"])
                if validation_identity(dataset) == identity:
                    scores_path = artifact / "evaluation_scores.npz"
                    if scores_path.is_file():
                        identity = {
                            **identity,
                            "evaluation_scores_sha256": file_digest(scores_path),
                        }
                    write_json(artifact / PROVENANCE_FILE, identity)
            except (OSError, ValueError, KeyError, TypeError):
                pass  # No proof => ineligible, without changing legacy run status.
        return record
