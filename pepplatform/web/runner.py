from pepplatform.task_compat import normalize_task_type

from pepplatform.pair_terminology import public_pair_strategy
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import traceback
from collections import deque
from pathlib import Path

from pepplatform.route_registry import (
    annotate_legacy_prediction_head,
    canonicalize_encoder_config,
    resolve_effective_route,
)
from .metrics import parse_metric_line
from pepplatform.pr_metrics import metric_definition
from .preflight import missing_bundle_metadata
from .classification_losses import (
    CLASSIFICATION_LOSS_FIELDS,
    PAIR_LOSS_FIELDS,
    SEQUENCE_LOSS_FIELDS,
)
from .store import utcnow
from pepplatform.stream_spec import (
    STREAM_CORE_FIELDS,
    STREAM_PARAM_FIELDS,
    STREAM_ROLES,
    flatten_streams_for_cli,
)
from pepplatform.training_failures import (
    failure_from_exception,
    failure_from_process_exit,
    finalize_training_failure,
    parse_training_failure_line,
    prefer_training_failure,
)
from pepplatform.svm_resume import (
    inspect_svm_resume_bundle,
    remove_svm_resume_bundle,
)


logger = logging.getLogger(__name__)


BOOL_FIELDS = {
    "enable_semantic_similarity",
    "ml_auto_tune",
    "report_aupr",
    "cache_features",
}

COMMON_FIELDS = {
    "checkpoint_objective",
    "checkpoint_objective_params",
    "threshold_objective",
    "threshold_objective_params",
    "threshold_search_profile",
    "tuning_objective",
    "tuning_objective_params",
    "checkpoint_threshold_objective",
    "checkpoint_threshold_objective_params",
    "checkpoint_threshold_search_profile",
    "task_type",
    "tokenizer",
    "encoder",
    "model",
    "sequence_model",
    "training_mode",
    "classifier",
    "head_hidden_dims",
    "head_activation",
    "head_dropout",
    "residue_adapter_dim",
    "pair_strategy",
    "interaction_level",
    "interaction_model",
    "predictor_type",
    "graph_predictor",
    "fusion_mode",
    "stream_alignment_dim",
    "interaction_hidden_dim",
    "interaction_heads",
    "k",
    "dr_max_distance",
    "dr_max_raw_length",
    "dr_unknown_residue_policy",
    "frequency_max_raw_length",
    "covariance_max_raw_length",
    "dr_dense_memory_limit_gib",
    "top_ngram_n",
    "topngram_dense_memory_limit_gib",
    "dt_max_distance",
    "dt_dense_memory_limit_gib",
    "embedding_vector_size",
    "embedding_window",
    "embedding_min_count",
    "embedding_epochs",
    "embedding_workers",
    "fasttext_bucket",
    "static_embedding_memory_limit_gib",
    "glove_x_max",
    "glove_alpha",
    "glove_learning_rate",
    "glove_batch_size",
    "glove_path",
    "max_length",
    "batch_size",
    "lr",
    "epochs",
    "dropout",
    "weight_decay",
    "seed",
    "grad_clip_norm",
    "early_stop_patience",
    "early_stop_min_delta",
    "reduction",
    "lsa_dim",
    "lsa_dense_memory_limit_gib",
    "lda_topics",
    "lda_dense_memory_limit_gib",
    "plsa_topics",
    "plsa_memory_limit_gib",
    "tsvd_dense_memory_limit_gib",
    "pca_dense_memory_limit_gib",
    "kpca_memory_limit_gib",
    "ml_scaler",
    "dl_scaler",
    "ml_reduction",
    "ml_reduction_dim",
    "kpca_kernel",
    "kpca_gamma",
    "ml_auto_tune",
    "ml_tuning_budget",
    "ml_tuning_objective",
    "ml_tuning_mode",
    "ml_tuning_max_candidates",
    "ml_cv_folds",
    "ensemble_members",
    "ensemble_method",
    "ensemble_weight_source",
    "ensemble_grid_step",
    "svm_c",
    "svm_class_weight",
    "svm_max_iter",
    "svm_tol",
    "svm_kernel",
    "svm_gamma",
    "svm_convergence_policy",
    "logreg_c",
    "logreg_class_weight",
    "logreg_max_iter",
    "logreg_convergence_policy",
    "rf_estimators",
    "rf_max_depth",
    "rf_min_samples_leaf",
    "et_estimators",
    "et_max_depth",
    "et_min_samples_leaf",
    "histgb_max_iter",
    "histgb_max_leaf_nodes",
    "histgb_lr",
    "histgb_l2_regularization",
    "xgb_estimators",
    "xgb_max_depth",
    "xgb_lr",
    "xgb_subsample",
    "xgb_colsample",
    "xgb_reg_alpha",
    "xgb_reg_lambda",
    "lgb_estimators",
    "lgb_max_depth",
    "lgb_lr",
    "lgb_subsample",
    "lgb_colsample",
    "lgb_reg_alpha",
    "lgb_reg_lambda",
    "enable_semantic_similarity",
    "semantic_similarity_weight",
    "semantic_loss_type",
    "semantic_zero_weight",
    "semantic_positive_partners",
    "semantic_zero_partners",
    "report_aupr",
    "cache_features",
    "feature_dir",
    "profile_cache_dir",
    "pssm_index_path",
    "ac_pssm_blosum62_hybrid_index_path",
    "cc_pssm_blosum62_hybrid_index_path",
    "acc_pssm_blosum62_hybrid_index_path",
    "psfm_index_path",
    "frequency_profile_index_path",
    "ac_lag",
    "cc_lag",
    "acc_lag",
    "pdt_lag",
    "pdt_max_raw_length",
    "ac_pssm_lag",
    "cc_pssm_lag",
    "acc_pssm_lag",
    "ac_pssm_blosum62_hybrid_lag",
    "cc_pssm_blosum62_hybrid_lag",
    "acc_pssm_blosum62_hybrid_lag",
    "profile_iterations",
    "profile_evalue",
    "profile_inclusion_evalue",
    "profile_threads",
    "profile_on_missing",
    "esm2_model",
    "protbert_model",
    "probert_model",
    "prott5_model",
    "esmc_model",
    "esm3_model",
}
COMMON_FIELDS.update(CLASSIFICATION_LOSS_FIELDS)
for _stream_role in STREAM_ROLES:
    COMMON_FIELDS.update(
        f"{_stream_role}_{field}" for field in (*STREAM_CORE_FIELDS, *STREAM_PARAM_FIELDS)
    )

PAIR_TRAINING_TASKS = {"pep_protein_interaction", "pep_hlaii", "pep_hlai"}
PAIR_ONLY_FIELDS = {
    "pair_strategy",
    "interaction_level",
    "interaction_model",
    "predictor_type",
    "graph_predictor",
    "fusion_mode",
    "stream_alignment_dim",
    "interaction_hidden_dim",
    "interaction_heads",
    "gnn_k",
    "gnn_hidden_dim",
    "gnn_exact_knn_max_nodes",
    "gnn_rp_projections",
    "gnn_rp_window",
    "cache_features",
}
SEQUENCE_ONLY_FIELDS = {
    "ensemble_members",
    "ensemble_method",
    "ensemble_weight_source",
    "ensemble_grid_step",
}


class TrainingProcessFailed(RuntimeError):
    def __init__(self, failure):
        super().__init__(str(failure.get("message") or "The training process failed."))
        self.failure = dict(failure)


def field_supported_for_task(key, task_type):
    task_type = normalize_task_type(task_type)
    is_pair = normalize_task_type(task_type) in PAIR_TRAINING_TASKS
    if is_pair:
        return key not in SEQUENCE_ONLY_FIELDS and key not in SEQUENCE_LOSS_FIELDS
    return key not in PAIR_ONLY_FIELDS and key not in PAIR_LOSS_FIELDS


class ExperimentRunner:
    CANCEL_REASON = "Training was stopped by the user."

    def __init__(self, store, paths, before_experiment=None, after_experiment=None):
        self.store = store
        self.paths = paths
        self.before_experiment = before_experiment
        self.after_experiment = after_experiment
        self.pending = queue.Queue()
        self.thread = None
        self._stopped = threading.Event()
        self._lock = threading.Lock()
        self._active_processes = {}
        self._canceled = set()

    def start(self):
        if self.thread and self.thread.is_alive():
            return self
        self.thread = threading.Thread(
            target=self._loop, name="pepplatform-experiment-runner", daemon=True
        )
        self.thread.start()
        return self

    def stop(self):
        self._stopped.set()
        self.pending.put(None)
        with self._lock:
            active = list(self._active_processes.items())
            self._canceled.update(experiment_id for experiment_id, _process in active)
        for _experiment_id, process in active:
            self._terminate_process(process)

    def submit(self, experiment_id):
        self.pending.put(experiment_id)

    def prepare_process_command(self, command, experiment):
        """Web-only supervisors may wrap execution without rewriting trainer flags."""
        return command

    def cancel(self, experiment_id, reason=None):
        reason = reason or self.CANCEL_REASON
        experiment = self.store.get_experiment(experiment_id)
        if not experiment:
            return None
        if experiment["status"] == "canceled":
            return experiment
        if experiment["status"] in {"succeeded", "failed"}:
            raise ValueError("completed or failed experiments cannot be stopped")

        with self._lock:
            self._canceled.add(experiment_id)
            process = self._active_processes.get(experiment_id)
        if process:
            self._terminate_process(process)
        updated = self._mark_canceled(experiment_id, reason)
        if not process:
            self._clear_canceled(experiment_id)
        return updated

    def _loop(self):
        while not self._stopped.is_set():
            experiment_id = self.pending.get()
            try:
                if experiment_id is None:
                    return
                self._run_experiment(experiment_id)
            except Exception as exc:
                # A failed task (including its version finalizer) must not take
                # down the only consumer of the Web training queue.
                logger.exception("Unhandled training task failure: %s", experiment_id)
                try:
                    with self._lock:
                        process = self._active_processes.get(experiment_id)
                    if process is not None:
                        self._terminate_process(process)
                    self._record_preparation_failure(experiment_id, exc)
                except Exception:
                    logger.exception("Could not persist task failure: %s", experiment_id)
                finally:
                    with self._lock:
                        self._active_processes.pop(experiment_id, None)
                        self._canceled.discard(experiment_id)
            finally:
                self.pending.task_done()

    def _record_preparation_failure(self, experiment_id, exc):
        failure = finalize_training_failure(
            failure_from_exception(exc, default_stage="setup"),
            exit_code=None,
        )
        experiment = self.store.get_experiment(experiment_id)
        if not experiment:
            return
        message = str(failure.get("message") or exc or "Failed to prepare training.")
        try:
            log_path = Path(experiment["log_path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[web-runner] {message}\n")
                log.writelines(traceback.format_exception(type(exc), exc, exc.__traceback__))
        except OSError:
            logger.exception("Could not write training failure log: %s", experiment_id)
        if experiment["status"] not in {"succeeded", "failed", "canceled"}:
            self.store.update_experiment(
                experiment_id,
                status="failed",
                error=message,
                failure=failure,
                finished_at=utcnow(),
            )

    def _run_experiment(self, experiment_id):
        experiment = self.store.get_experiment(experiment_id)
        if not experiment or experiment["status"] == "canceled" or self._is_canceled(experiment_id):
            self._clear_canceled(experiment_id)
            return
        if experiment["status"] != "queued":
            return
        dataset = self.store.get_dataset(experiment["dataset_id"])
        log_path = Path(experiment["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_dir = Path(experiment["artifact_dir"])
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write("[web-runner] Preparing training resources and input versions.\n")
        if self._is_canceled(experiment_id):
            self._mark_canceled(experiment_id, self.CANCEL_REASON)
            self._clear_canceled(experiment_id)
            return
        resources_prepared = False
        try:
            if callable(self.before_experiment):
                resources_prepared = True
                self.before_experiment()
        except Exception as exc:
            if resources_prepared and callable(self.after_experiment):
                try:
                    self.after_experiment()
                except Exception:
                    pass
            self._record_preparation_failure(experiment_id, exc)
            self._clear_canceled(experiment_id)
            return
        current = self.store.get_experiment(experiment_id)
        if self._is_canceled(experiment_id) or (current and current["status"] == "canceled"):
            if resources_prepared and callable(self.after_experiment):
                try:
                    self.after_experiment()
                except Exception:
                    pass
            self._mark_canceled(experiment_id, self.CANCEL_REASON)
            self._clear_canceled(experiment_id)
            return
        try:
            self.store.update_experiment(experiment_id, status="running", started_at=utcnow())
            command = self.build_command(experiment, dataset)
        except Exception as exc:
            if resources_prepared and callable(self.after_experiment):
                try:
                    self.after_experiment()
                except Exception:
                    pass
            self._record_preparation_failure(experiment_id, exc)
            self._clear_canceled(experiment_id)
            return
        with open(log_path, "a", encoding="utf-8") as log:
            log.write("$ " + " ".join(command) + "\n")
            log.flush()
            process = None
            reported_failure = None
            log_tail = deque(maxlen=200)
            try:
                process = subprocess.Popen(
                    self.prepare_process_command(command, experiment),
                    cwd=self.paths.project_root,
                    env=self.build_subprocess_env(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=os.name != "nt",
                )
                with self._lock:
                    self._active_processes[experiment_id] = process
                for line in process.stdout or []:
                    log.write(line)
                    log.flush()
                    log_tail.append(line.rstrip("\n"))
                    reported_failure = prefer_training_failure(
                        reported_failure,
                        parse_training_failure_line(line),
                    )
                    event = parse_metric_line(line)
                    if event:
                        event.update(metric_definition())
                        self.store.append_metric_event(experiment_id, event)
                return_code = process.wait()
                current = self.store.get_experiment(experiment_id)
                if self._is_canceled(experiment_id) or (
                    current and current["status"] == "canceled"
                ):
                    self._mark_canceled(experiment_id, self.CANCEL_REASON)
                    return
                if return_code != 0:
                    reported_failure = prefer_training_failure(
                        failure_from_process_exit(return_code, log_tail),
                        reported_failure,
                    )
                    raise TrainingProcessFailed(
                        finalize_training_failure(
                            reported_failure,
                            exit_code=return_code,
                        )
                    )
                remove_svm_resume_bundle(experiment["artifact_dir"])
                model = self._register_model_if_present(experiment_id, experiment)
                self.store.update_experiment(
                    experiment_id,
                    status="succeeded",
                    model_id=model["id"] if model else None,
                    finished_at=utcnow(),
                )
            except Exception as exc:
                current = self.store.get_experiment(experiment_id)
                if self._is_canceled(experiment_id) or (
                    current and current["status"] == "canceled"
                ):
                    self._mark_canceled(experiment_id, self.CANCEL_REASON)
                    return
                failure = (
                    dict(exc.failure)
                    if isinstance(exc, TrainingProcessFailed)
                    else finalize_training_failure(
                        failure_from_exception(exc),
                        exit_code=process.poll() if process is not None else None,
                    )
                )
                resume_bundle = inspect_svm_resume_bundle(experiment["artifact_dir"])
                if resume_bundle:
                    context = dict(failure.get("context") or {})
                    context.update(
                        {
                            "resume_available": True,
                            "resume_stage": "downstream_svm",
                            "resume_schema_version": resume_bundle["schema_version"],
                        }
                    )
                    failure["context"] = context
                message = str(failure.get("message") or exc or "The training process failed.")
                log.write(f"\n[web-runner] {message}\n")
                self.store.update_experiment(
                    experiment_id,
                    status="failed",
                    error=message,
                    failure=failure,
                    finished_at=utcnow(),
                )
            finally:
                if process is not None and process.poll() is None:
                    self._terminate_process(process)
                if resources_prepared and callable(self.after_experiment):
                    try:
                        self.after_experiment()
                    except Exception as cleanup_exc:
                        # Resource cleanup must not overwrite a completed training status.
                        log.write(f"\n[web-runner] resource cleanup warning: {cleanup_exc}\n")
                        log.flush()
                with self._lock:
                    if process is not None and self._active_processes.get(experiment_id) is process:
                        self._active_processes.pop(experiment_id, None)
                    self._canceled.discard(experiment_id)

    def _is_canceled(self, experiment_id):
        with self._lock:
            return experiment_id in self._canceled

    def _clear_canceled(self, experiment_id):
        with self._lock:
            self._canceled.discard(experiment_id)

    def _mark_canceled(self, experiment_id, reason):
        current = self.store.get_experiment(experiment_id)
        if current and current["status"] == "canceled" and current.get("error"):
            reason = current["error"]
        return self.store.update_experiment(
            experiment_id,
            status="canceled",
            error=reason,
            finished_at=utcnow(),
        )

    @staticmethod
    def _terminate_process(process, timeout=10):
        if process.poll() is not None:
            return
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:  # pragma: no cover - Windows-only process handling
                process.terminate()
            process.wait(timeout=timeout)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover - Windows-only process handling
                process.kill()
            process.wait()

    def _register_model_if_present(self, experiment_id, experiment):
        artifact_dir = Path(experiment["artifact_dir"])
        metadata_path = artifact_dir / "metadata.json"
        config = experiment.get("config") or {}
        is_ensemble = bool(config.get("ensemble_members"))
        if not metadata_path.exists():
            metadata = missing_bundle_metadata(
                experiment["task_type"],
                config,
            )
            if is_ensemble:
                route = {
                    "schema_version": 5,
                    "route_id": "sequence.ensemble",
                    "task_type": experiment["task_type"],
                    "family": "ensemble",
                    "tokenizer": "ensemble",
                    "encoder": "probability_mean",
                    "model": "ensemble",
                }
                metadata["route_schema_version"] = route["schema_version"]
                metadata["effective_route"] = route
                metadata["bundle_type"] = "ensemble"
                metadata["supports_inference"] = True
            else:
                route = resolve_effective_route(
                    experiment.get("config") or {}, experiment["task_type"]
                )
                metadata["route_schema_version"] = route["schema_version"]
                metadata["effective_route"] = route
            return self.store.create_model(
                experiment_id,
                experiment["name"],
                experiment["task_type"],
                artifact_dir,
                False,
                metadata,
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if is_ensemble:
            route = {
                "schema_version": 5,
                "route_id": "sequence.ensemble",
                "task_type": experiment["task_type"],
                "family": "ensemble",
                "tokenizer": "ensemble",
                "encoder": "probability_mean",
                "model": "ensemble",
            }
            route = annotate_legacy_prediction_head(route, metadata)
            metadata["route_schema_version"] = route["schema_version"]
            metadata["effective_route"] = route
            metadata["bundle_type"] = metadata.get("bundle_type") or "ensemble"
            metadata["supports_inference"] = bool(metadata.get("supports_inference", True))
            return self.store.create_model(
                experiment_id,
                experiment["name"],
                experiment["task_type"],
                artifact_dir,
                True,
                metadata,
            )
        route = resolve_effective_route(experiment.get("config") or {}, experiment["task_type"])
        route = annotate_legacy_prediction_head(metadata.get("effective_route") or route, metadata)
        metadata["route_schema_version"] = route["schema_version"]
        metadata["effective_route"] = route
        return self.store.create_model(
            experiment_id,
            experiment["name"],
            experiment["task_type"],
            artifact_dir,
            bool(metadata.get("supports_inference", False)),
            metadata,
        )

    @staticmethod
    def build_subprocess_env():
        env = os.environ.copy()
        # Child scripts and their nested profile-cache builders must stream
        # newline-delimited status records through the runner immediately.
        env["PYTHONUNBUFFERED"] = "1"
        offline = str(env.get("PEPPLATFORM_OFFLINE_MODE", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if offline:
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
        elif not str(env.get("HF_ENDPOINT", "")).strip():
            env["HF_ENDPOINT"] = (
                str(env.get("PEPPLATFORM_HF_ENDPOINT", "")).strip() or "https://hf-mirror.com"
            )
        if not str(env.get("HF_HUB_DISABLE_XET", "")).strip():
            env["HF_HUB_DISABLE_XET"] = "1"
        if not str(env.get("HF_HUB_DOWNLOAD_TIMEOUT", "")).strip():
            env["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
        if not str(env.get("HF_HUB_ETAG_TIMEOUT", "")).strip():
            env["HF_HUB_ETAG_TIMEOUT"] = "30"
        return env

    def build_command(self, experiment, dataset):
        config = canonicalize_encoder_config(experiment["config"])
        task_type = experiment["task_type"]
        ensemble_members = config.get("ensemble_members")
        if ensemble_members:
            if normalize_task_type(task_type) in PAIR_TRAINING_TASKS:
                raise ValueError("ensemble is only supported for sequence tasks")
            if isinstance(ensemble_members, str):
                member_values = [
                    item.strip() for item in ensemble_members.split(",") if item.strip()
                ]
            else:
                member_values = [
                    str(item).strip() for item in ensemble_members if str(item).strip()
                ]
            if len(member_values) < 2:
                raise ValueError("ensemble_members must contain at least two entries")
            resolved_members = []
            resolved_model_ids = []
            for value in member_values:
                record = self.store.get_model(value) if self.store is not None else None
                if record and record.get("bundle_path"):
                    resolved_members.append(str(record["bundle_path"]))
                    resolved_model_ids.append(value)
                else:
                    path = Path(value)
                    if path.is_dir():
                        resolved_members.append(str(path.resolve()))
                    else:
                        raise ValueError(
                            "ensemble member is neither a registered model id "
                            f"nor an artifact directory: {value}"
                        )
            script = self.paths.project_root / "scripts" / "train_ensemble.py"
            command = [
                sys.executable,
                str(script),
                "--task_type",
                task_type,
                "--members",
                ",".join(resolved_members),
                "--ensemble-method",
                str(config.get("ensemble_method") or "probability_mean"),
                "--ensemble-weight-source",
                str(config.get("ensemble_weight_source") or "validation"),
                "--artifact-dir",
                experiment["artifact_dir"],
                "--data-dir",
                dataset["root_path"],
            ]
            if resolved_model_ids:
                command.extend(["--member-model-ids", ",".join(resolved_model_ids)])
            if config.get("ensemble_grid_step") is not None:
                command.extend(["--ensemble-grid-step", str(config["ensemble_grid_step"])])
            if config.get("seed") is not None:
                command.extend(["--seed", str(config["seed"])])
            if config.get("evaluate_test") is not None:
                command.append(
                    "--evaluate-test" if config["evaluate_test"] else "--no-evaluate-test"
                )
            return command
        if normalize_task_type(task_type) in PAIR_TRAINING_TASKS:
            script = self.paths.project_root / "scripts" / "train_pair_pipeline.py"
            config = flatten_streams_for_cli(config)
        else:
            script = self.paths.project_root / "scripts" / "train_sequence_pipeline.py"
            config.setdefault(
                "profile_cache_dir",
                str(Path(experiment["artifact_dir"]).parent / "profile_cache"),
            )
        command = [sys.executable, str(script)]
        for key in sorted(COMMON_FIELDS):
            if not field_supported_for_task(key, task_type):
                continue
            if key not in config or config[key] in [None, ""]:
                continue
            value = config[key]
            if key == "pair_strategy":
                value = public_pair_strategy(value)
            if key in BOOL_FIELDS:
                command.append(f"--{key}" if bool(value) else f"--no-{key}")
            else:
                command.extend(
                    [
                        f"--{key}",
                        json.dumps(value, sort_keys=True, allow_nan=False)
                        if key.endswith("_objective_params") and isinstance(value, dict)
                        else str(value),
                    ]
                )
        if normalize_task_type(task_type) in PAIR_TRAINING_TASKS:
            if dataset["format"] in {"pair_csv_bundle", "pair_split_dir", "sequence_dir"}:
                root = Path(dataset["root_path"])
                metadata = dataset.get("metadata", {}) or {}
                task_root = metadata.get("source_root") or str(root)
                command.extend(["--task_root", str(task_root)])
                if dataset["format"] in {"pair_split_dir", "sequence_dir"}:
                    train_csv = root / "train" / "pairs.csv"
                    val_csv = root / "val" / "pairs.csv"
                    test_csv = root / "test" / "pairs.csv"
                else:
                    train_csv = root / "train.csv"
                    val_csv = root / "val.csv"
                    test_csv = root / "test.csv"
                command.extend(["--train_csv", str(train_csv)])
                command.extend(["--val_csv", str(val_csv)])
                if test_csv.exists():
                    command.extend(["--test_csv", str(test_csv)])
                if metadata.get("feature_dir") and not config.get("feature_dir"):
                    command.extend(["--feature_dir", str(metadata["feature_dir"])])
            else:
                command.extend(["--task_root", dataset["root_path"]])
        else:
            command.extend(["--data_dir", dataset["root_path"]])
        command.extend(["--artifact_dir", experiment["artifact_dir"]])
        resume = experiment.get("resume") or {}
        if resume.get("stage") == "downstream_svm":
            command.extend(["--resume_embedding_bundle", experiment["artifact_dir"]])
        return command
