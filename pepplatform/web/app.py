from pepplatform.task_compat import normalize_task_type

import csv
import hmac
import io
import json
import logging
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

from pepplatform.analysis import empty_analysis, load_analysis
from pepplatform.route_registry import annotate_legacy_prediction_head, resolve_effective_route
from pepplatform.svm_resume import (
    SVM_DOWNSTREAM_OVERRIDE_FIELDS,
    inspect_svm_resume_bundle,
    materialize_svm_resume_bundle,
)

from .ai import (
    AiChatClient,
    AiConfigurationError,
    AiProviderError,
    AiStreamInterruptedError,
    api_key_last4,
    build_analysis_messages,
    build_experiment_context,
    decrypt_api_key,
    encrypt_api_key,
    markdown_to_plain_summary,
    normalize_ai_provider_payload,
    provider_presets,
)
from .agent import (
    AgentResponseError,
    STRATEGY_PREFERENCES,
    build_agent_campaign_round_context,
    build_agent_chat_system_prompt,
    build_agent_execution_plan_messages,
    build_agent_execution_plan_repair_messages,
    build_agent_plan_review_context,
    build_agent_plan_review_messages,
    build_experiment_design_context,
    build_experiment_design_messages,
    build_experiment_design_repair_messages,
    experiment_design_context_summary,
    parse_agent_execution_actions,
    parse_agent_plan_review,
    parse_experiment_design_candidates,
    validate_execution_actions_for_repair,
    validate_design_candidates_for_repair,
)
from .auth import (
    CODE_PURPOSE_EMAIL_VERIFY,
    CODE_PURPOSE_PASSWORD_RESET,
    SESSION_COOKIE_NAME,
    AuthSettings,
    clear_cookie_kwargs,
    code_expires_at,
    cookie_kwargs,
    generate_code,
    generate_session_token,
    hash_password,
    log_auth_mail_configuration,
    secret_hash,
    send_verification_email,
    session_expires_at,
    utcnow as auth_utcnow,
    parse_iso,
    verify_password,
)
from .capabilities import build_capabilities
from .config import normalize_experiment_config_with_sources
from .dataset_reports import build_dataset_report
from .datasets import DatasetService, DatasetValidationError, seed_builtin_datasets
from .dataset_uploads import (
    DatasetArchiveError,
    DatasetUploadBodyLimitMiddleware,
    DatasetUploadLimitError,
    DatasetUploadLimits,
)
from .env_file import load_env_file
from .experiment_analysis import (
    ExperimentAnalysisResponseError,
    build_experiment_followup_messages,
    build_experiment_report_messages,
    build_experiment_report_repair_messages,
    enrich_context_manifest,
    parse_experiment_diagnostic_report,
    report_as_markdown,
)
from .model_artifacts import (
    delete_dataset_artifact,
    delete_experiment_artifact,
    delete_model_artifact,
)
from .paths import RuntimePaths
from .preflight import preflight_experiment
from .profile_runtime import profile_environment_readiness
from .runner import COMMON_FIELDS
from .resource_runner import GovernedExperimentRunner as ExperimentRunner
from .resource_governor import ResourceGovernor, BudgetedVersionRepository
from .resource_inference import ManagedModelService, RemoteInferenceError, model_uses_gpu
from .resource_policy import ResourceLimitError, DEFAULT_QUOTAS as RESOURCE_QUOTAS
from .resource_routes import register_resource_routes, serialize_resource_mutations
from contextlib import nullcontext
from .ensemble_service import EnsembleService, WebModelService as ModelService, creation_enabled
from .ensemble_math import EnsembleError
from .store import (
    GENERATED_NAME_QUALIFIERS,
    SUPPORTED_LOCALES,
    MetadataStore,
    default_display_name,
    normalize_email,
    validated_experiment_id,
    utcnow as store_utcnow,
)
from .web_search import (
    TavilyWebSearchClient,
    WebSearchUnavailable,
    build_web_search_query,
    build_web_search_system_message,
)

try:
    from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
    from fastapi.responses import StreamingResponse
    from starlette.concurrency import run_in_threadpool
except ImportError:  # pragma: no cover - exercised only when optional Web deps are absent
    FastAPI = None


DISPLAY_NAME_MAX_LENGTH = 40
ADMIN_ROLES = {"admin", "user"}
ADMIN_STATUSES = {"active", "disabled"}
ADMIN_USER_SORTS = {"created_at", "display_name", "email", "role", "status"}
ADMIN_SORT_ORDERS = {"asc", "desc"}


def _sse_frame(event, payload):
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"


def _with_effective_experiment_config(experiment):
    """Expose effective defaults for legacy records without rewriting stored runs."""
    if not isinstance(experiment, dict):
        return experiment
    stored_config = dict(experiment.get("config") or {})
    if stored_config.get("web_ensemble"):
        return {
            **experiment,
            "requested_config": stored_config,
            "config_parameter_sources": {},
            "effective_route": {
                "schema_version": 5,
                "family": "ensemble",
                "model": "ensemble",
                "task_type": experiment["task_type"],
                "encoder": stored_config["web_ensemble"]["method"],
            },
        }
    try:
        effective_config, parameter_sources = normalize_experiment_config_with_sources(
            experiment.get("task_type"),
            stored_config,
        )
    except (TypeError, ValueError):
        return experiment
    route = resolve_effective_route(effective_config, experiment.get("task_type"))
    raw_artifact_dir = str(experiment.get("artifact_dir") or "").strip()
    metadata_path = Path(raw_artifact_dir) / "metadata.json" if raw_artifact_dir else None
    if metadata_path and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            metadata = {}
        route = annotate_legacy_prediction_head(route, metadata)
    failure = dict(experiment.get("failure") or {})
    failure_context = dict(failure.get("context") or {})
    classifier = (
        str(effective_config.get("classifier") or effective_config.get("model") or "")
        .strip()
        .lower()
    )
    resume_manifest = None
    if experiment.get("status") == "failed" and classifier == "svm" and raw_artifact_dir:
        resume_manifest = inspect_svm_resume_bundle(
            raw_artifact_dir,
            verify_integrity=False,
        )
    resume_available = bool(resume_manifest and resume_manifest.get("stage") == "downstream_svm")
    if (
        failure.get("code") == "training_data_invalid"
        and failure.get("message")
        == "SVM predictor contract requires shared parameters across outputs"
    ):
        failure.update(
            {
                "code": "svm_artifact_contract_failed",
                "category": "runtime",
                "stage": "artifact_export",
            }
        )
    if resume_available:
        failure_context.update(
            {
                "resume_available": True,
                "resume_stage": "downstream_svm",
                "resume_schema_version": resume_manifest["schema_version"],
            }
        )
        failure["context"] = failure_context
    return {
        **experiment,
        "config": effective_config,
        "requested_config": stored_config,
        "config_parameter_sources": parameter_sources,
        "effective_route": route,
        "failure": failure or None,
        "resume_capabilities": {
            "downstream_svm": {
                "available": resume_available,
                "reason": "" if resume_available else "resume_bundle_unavailable",
            }
        },
    }


def _prediction_csv(response_payload):
    labels = list((response_payload or {}).get("labels") or [])
    predictions = list((response_payload or {}).get("predictions") or [])
    ensemble = (response_payload or {}).get("ensemble")
    is_pair = any(
        row.get("peptide") or row.get("protein") for row in predictions if isinstance(row, dict)
    )
    score_headers = [f"score_{label or index + 1}" for index, label in enumerate(labels)]
    if not score_headers:
        max_scores = max(
            (len(row.get("scores") or []) for row in predictions if isinstance(row, dict)),
            default=0,
        )
        score_headers = [f"score_{index + 1}" for index in range(max_scores)]
    headers = ["input_index"]
    if is_pair:
        headers.extend(
            [
                "peptide",
                "protein",
                "prediction_label",
                "score",
                "score_kind",
                "threshold",
                "is_interaction",
            ]
        )
    else:
        headers.extend(["sequence", "top_label", "top_score"])
    headers.extend(score_headers)
    threshold_headers = []
    if not is_pair:
        headers.append("predicted_labels")
        threshold_headers = [header.replace("score_", "threshold_", 1) for header in score_headers]
        headers.extend(threshold_headers)
    if ensemble:
        if not is_pair:
            headers.append("score_kind")
        headers.extend(["ensemble_method", "batch_digest", "batch_input_count", "numpy_version"])

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()
    for index, row in enumerate(predictions, start=1):
        row = row if isinstance(row, dict) else {}
        scores = list(row.get("scores") or [])
        record = {"input_index": index}
        if is_pair:
            record.update(
                {
                    "peptide": row.get("peptide", ""),
                    "protein": row.get("protein", ""),
                    "prediction_label": row.get("prediction_label", ""),
                    "score": row.get("score", ""),
                    "score_kind": row.get("score_kind", ""),
                    "threshold": row.get("threshold", ""),
                    "is_interaction": row.get("is_interaction", ""),
                }
            )
        else:
            top_index = max(range(len(scores)), key=lambda item: scores[item]) if scores else None
            record.update(
                {
                    "sequence": row.get("sequence", ""),
                    "top_label": labels[top_index]
                    if top_index is not None and top_index < len(labels)
                    else "",
                    "top_score": scores[top_index] if top_index is not None else "",
                    "predicted_labels": ";".join(
                        str(value) for value in (row.get("predicted_labels") or [])
                    ),
                }
            )
        for score_index, header in enumerate(score_headers):
            record[header] = scores[score_index] if score_index < len(scores) else ""
        if not is_pair:
            thresholds = list((response_payload or {}).get("thresholds") or [])
            for threshold_index, header in enumerate(threshold_headers):
                record[header] = (
                    thresholds[threshold_index] if threshold_index < len(thresholds) else ""
                )
        if ensemble:
            record.update(
                score_kind=row.get("score_kind", response_payload.get("score_kind", "")),
                ensemble_method=ensemble.get("method", ""),
                batch_digest=ensemble.get("batch_digest", ""),
                batch_input_count=ensemble.get("input_count", ""),
                numpy_version=ensemble.get("numpy_version", ""),
            )
        writer.writerow(record)
    return output.getvalue()


def _public_user(user):
    if not user:
        return None
    return {
        "id": user["id"],
        "email": user["email"],
        "display_name": user.get("display_name") or default_display_name(user["email"]),
        "preferred_locale": user.get("preferred_locale") or "en",
        "email_verified": bool(user["email_verified"]),
        "role": user["role"],
        "status": user["status"],
    }


def _is_admin(user):
    return bool(user and user.get("role") == "admin")


def secrets_compare(left, right):
    return hmac.compare_digest(str(left), str(right))


def validate_display_name(value):
    display_name = str(value or "").strip()
    if not display_name:
        raise HTTPException(status_code=400, detail="display name is required")
    if len(display_name) > DISPLAY_NAME_MAX_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"display name must be at most {DISPLAY_NAME_MAX_LENGTH} characters",
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in display_name):
        raise HTTPException(status_code=400, detail="display name contains invalid characters")
    return display_name


def validate_preferred_locale(value):
    preferred_locale = str(value or "en").strip().lower()
    if preferred_locale not in SUPPORTED_LOCALES:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "unsupported_locale",
                "message": "preferred_locale must be en or zh",
            },
        )
    return preferred_locale


def _reload_watches_runtime_root(argv, runtime_root):
    """Report whether a uvicorn ``--reload`` watcher covers the runtime root.

    Starting an experiment stages ``*.py`` source snapshots under the runtime root, so a
    reloader watching that tree restarts the backend in the middle of the launch and the
    startup reconciliation then cancels the experiment.
    """
    reload_enabled = False
    reload_dirs = []
    excluded_dirs = []
    for index, arg in enumerate(argv):
        name, sep, value = arg.partition("=")
        if name == "--reload":
            reload_enabled = value.lower() not in {"false", "0", "no"} if sep else True
        elif name in {"--reload-dir", "--reload-exclude"}:
            if sep:
                candidate = value
            elif index + 1 < len(argv):
                candidate = argv[index + 1]
            else:
                continue
            target = reload_dirs if name == "--reload-dir" else excluded_dirs
            target.append(Path(candidate).resolve())
    if not reload_enabled:
        return False
    runtime_root = Path(runtime_root).resolve()
    if any(entry == runtime_root or entry in runtime_root.parents for entry in excluded_dirs):
        return False
    if reload_dirs:
        return any(entry == runtime_root or entry in runtime_root.parents for entry in reload_dirs)
    return True


def create_app(project_root=None, runtime_root=None):
    if FastAPI is None:  # pragma: no cover - depends on optional install state
        raise RuntimeError("FastAPI is not installed. Install requirements-web.txt first.")
    project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    load_env_file(project_root / ".env")
    upload_limits = DatasetUploadLimits.from_env()
    paths = RuntimePaths.from_project_root(project_root, runtime_root).ensure()
    if _reload_watches_runtime_root(sys.argv, paths.root):
        logging.getLogger("uvicorn.error").warning(
            "uvicorn --reload watches the runtime root %s. Starting an experiment stages "
            "*.py source snapshots there, so the reloader will restart the backend and cancel "
            "the run. Use --reload-dir pepplatform --reload-dir scripts, or start without "
            "--reload while training.",
            paths.root,
        )
    settings = AuthSettings.from_env()
    log_auth_mail_configuration(settings, logger=logging.getLogger("uvicorn.error"))
    store = MetadataStore(os.environ.get("PEP_DATABASE_URL") or paths.database)
    metric_summary_refresh = store.refresh_metric_summaries()
    if settings.initial_admin_email and settings.initial_admin_password:
        store.ensure_initial_admin(
            settings.initial_admin_email,
            hash_password(settings.initial_admin_password),
        )
    seed_builtin_datasets(store, paths)
    datasets = DatasetService(store, paths, upload_limits=upload_limits)
    from pepplatform.versioning.web import register_version_routes
    from .training_inputs import WebTrainingInputs

    versions = BudgetedVersionRepository(store.engine, paths.root / "versions")
    governor = ResourceGovernor(store, paths, versions)
    versions.governor = governor
    store.resource_governor = governor
    governor.scan()
    models = ManagedModelService(store, versions, governor)
    app = FastAPI(title="PepPlatform Web API", version="0.1.0")
    app.state.models = models

    def _suspend_inference():
        current = app.state.models
        (current if isinstance(current, ManagedModelService) else models).suspend_esm3_runtime()

    def _resume_inference():
        current = app.state.models
        (current if isinstance(current, ManagedModelService) else models).resume_esm3_runtime()

    runner = ExperimentRunner(
        store,
        paths,
        before_experiment=_suspend_inference,
        after_experiment=_resume_inference,
        versions=versions,
        input_policy=WebTrainingInputs(),
        governor=governor,
    ).start()

    app.add_middleware(DatasetUploadBodyLimitMiddleware, limits=upload_limits)
    app.state.dataset_upload_limits = upload_limits
    app.state.paths = paths
    app.state.store = store
    app.state.versions = versions
    app.state.metric_summary_refresh = metric_summary_refresh
    app.state.runner = runner
    app.state.models = models
    app.state.resource_governor = governor
    from .heatmap_jobs import HeatmapJobs
    from .heatmap_routes import register_heatmap_routes

    app.state.heatmap_jobs = HeatmapJobs(paths.root)
    app.state.ai_client = AiChatClient()
    app.state.web_search_client = TavilyWebSearchClient()
    app.state.auth_settings = settings

    @app.on_event("shutdown")
    def _shutdown_resources():
        try:
            app.state.heatmap_jobs.stop()
        finally:
            try:
                runner.stop()
            finally:
                current = app.state.models
                if current is not models and hasattr(current, "close"):
                    current.close()
                models.close()

    def _current_user(request):
        if not settings.enabled:
            return None
        if request is None:
            raise HTTPException(status_code=401, detail="authentication required")
        token = request.cookies.get(SESSION_COOKIE_NAME)
        if not token:
            raise HTTPException(status_code=401, detail="authentication required")
        token_hash = secret_hash(token, settings.session_secret)
        session = store.get_session_by_hash(token_hash)
        now = auth_utcnow()
        if not session or session.get("revoked_at"):
            raise HTTPException(status_code=401, detail="authentication required")
        try:
            expired = parse_iso(session["expires_at"]) <= now
        except ValueError as exc:
            raise HTTPException(status_code=401, detail="authentication required") from exc
        if expired:
            store.revoke_session(token_hash)
            raise HTTPException(status_code=401, detail="session expired")
        user = store.get_user(session["user_id"])
        if not user or user["status"] != "active":
            raise HTTPException(status_code=401, detail="authentication required")
        store.touch_session(session["id"])
        return user

    def _maybe_current_user(request):
        try:
            return _current_user(request)
        except HTTPException:
            return None

    def _current_profile_user(request):
        user = _current_user(request)
        if not user:
            raise HTTPException(status_code=401, detail="authentication required")
        return user

    def _require_admin(request):
        user = _current_user(request)
        if not _is_admin(user):
            raise HTTPException(status_code=403, detail="admin access required")
        return user

    def _request_ip(request):
        if request is None:
            return ""
        client = getattr(request, "client", None)
        return str(getattr(client, "host", "") or "")

    def _request_user_agent(request):
        if request is None:
            return ""
        headers = getattr(request, "headers", {}) or {}
        return str(headers.get("user-agent", "") or "")

    def _audit(event_type, request=None, actor=None, target_user_id=None, metadata=None):
        return store.create_audit_log(
            event_type,
            actor_user_id=actor.get("id") if actor else None,
            target_user_id=target_user_id,
            metadata=metadata or {},
            ip_address=_request_ip(request),
            user_agent=_request_user_agent(request),
        )

    def _quota_exceeded(quota_type, limit, used, requested):
        from .resource_policy import check_quota

        try:
            check_quota(quota_type, int(limit), int(used), int(requested))
        except ResourceLimitError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail=exc.detail,
                headers={"Retry-After": str(exc.detail["retry_after"])}
                if "retry_after" in exc.detail
                else None,
            ) from exc

    def _check_experiment_quota(user, requested_count=1):
        governor.admit_training(user, max(0, int(requested_count or 0)))
        if not settings.enabled or not user:
            return
        requested_count = max(1, int(requested_count or 1))
        quota = store.get_user_quota(user["id"])
        if quota.get("enforcement") == "exempt_admin":
            return
        usage = store.get_quota_usage(user["id"])
        active_experiments = store.count_active_experiments_for_user(user["id"])
        _quota_exceeded(
            "daily_experiment_limit",
            quota["daily_experiment_limit"],
            usage["experiments_created"],
            requested_count,
        )
        _quota_exceeded(
            "daily_concurrent_experiment_limit",
            quota["daily_concurrent_experiment_limit"],
            active_experiments,
            requested_count,
        )

    def _check_inference_quota(user, input_count, model_id=None):
        try:
            return governor.reserve_inference(user, input_count, model_id=model_id)
        except ResourceLimitError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail=exc.detail,
                headers={"Retry-After": str(exc.detail["retry_after"])}
                if "retry_after" in exc.detail
                else None,
            ) from exc

    def _require_dataset(dataset_id, user, write=False):
        dataset = store.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="dataset not found")
        allowed = (
            not settings.enabled
            or _is_admin(user)
            or (dataset["source"] == "builtin" and not write)
            or dataset.get("owner_user_id") == user["id"]
        )
        if not allowed:
            raise HTTPException(status_code=403, detail="permission denied")
        return datasets.ensure_length_stats(dataset)

    def _require_experiment(experiment_id, user):
        experiment = store.get_experiment(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="experiment not found")
        if (
            settings.enabled
            and not _is_admin(user)
            and experiment.get("owner_user_id") != user["id"]
        ):
            raise HTTPException(status_code=403, detail="permission denied")
        return experiment

    register_heatmap_routes(app, app.state.heatmap_jobs, _current_user, _require_experiment)
    register_version_routes(app, versions, _current_user, _require_experiment)

    def _generated_experiment_name_fields(payload, user):
        generated = (payload or {}).get("generated_name")
        if not generated:
            return {}
        if not isinstance(generated, dict):
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "generated_name_invalid",
                    "message": "generated_name must describe a config name or rerun",
                },
            )
        kind = str(generated.get("kind") or "").strip()
        qualifier = str(generated.get("qualifier") or "").strip()
        if qualifier and qualifier not in GENERATED_NAME_QUALIFIERS:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "generated_name_invalid",
                    "message": "generated_name qualifier is invalid",
                },
            )
        if kind == "config":
            return {
                "name_kind": "config",
                "name_qualifier": qualifier or None,
            }
        if kind != "rerun":
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "generated_name_invalid",
                    "message": "generated_name must describe a config name or rerun",
                },
            )
        source_id = str(generated.get("source_experiment_id") or "").strip()
        if not source_id:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "rerun_source_required",
                    "message": "rerun source experiment is required",
                },
            )
        source = _require_experiment(source_id, user)
        source_generated = source.get("generated_name") or {}
        if source_generated.get("kind") == "rerun" and source_generated.get("base_name"):
            base_name = source_generated["base_name"]
            base_kind = source_generated.get("base_kind") or "literal"
            qualifier = source_generated.get("qualifier") or None
            rerun_count = max(1, int(source_generated.get("count") or 1)) + 1
        elif source_generated.get("kind") == "config":
            base_name = source["name"]
            base_kind = "config"
            qualifier = source_generated.get("qualifier") or None
            rerun_count = 1
        else:
            base_name = source["name"]
            base_kind = "literal"
            qualifier = None
            rerun_count = 1
        return {
            "name_kind": "rerun",
            "name_base": base_name,
            "name_base_kind": base_kind,
            "name_qualifier": qualifier,
            "rerun_count": rerun_count,
            "rerun_source_experiment_id": source_id,
        }

    def _require_model(model_id, user, *, allow_removed=False):
        model = store.get_model(model_id, include_removed=True)
        if not model:
            raise HTTPException(status_code=404, detail="model not found")
        if settings.enabled and not _is_admin(user) and model.get("owner_user_id") != user["id"]:
            raise HTTPException(status_code=403, detail="permission denied")
        if not allow_removed:
            governor.models.require_available(model_id)
        return model

    def _require_inference_run(run_id, user):
        run = store.get_inference_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="inference run not found")
        if settings.enabled and not _is_admin(user) and run.get("owner_user_id") != user["id"]:
            raise HTTPException(status_code=403, detail="permission denied")
        return run

    def _require_ai_provider(provider_id, user, include_secret=False):
        provider = store.get_ai_provider_config(
            provider_id, user=user, include_secret=include_secret
        )
        if not provider:
            raise HTTPException(status_code=404, detail="AI provider config not found")
        return provider

    def _require_agent_execution_plan(plan_id, user):
        plan = store.get_agent_execution_plan(plan_id, user=user)
        if not plan:
            raise HTTPException(status_code=404, detail="agent execution plan not found")
        return plan

    def _require_agent_campaign(campaign_id, user, include_plans=False):
        campaign = store.get_agent_campaign(campaign_id, user=user, include_plans=include_plans)
        if not campaign:
            raise HTTPException(status_code=404, detail="agent campaign not found")
        return campaign

    def _public_ai_provider(provider):
        if not provider:
            return None
        return {key: value for key, value in provider.items() if key not in {"api_key_ciphertext"}}

    def _ai_provider_response(provider):
        return _public_ai_provider(provider)

    def _sanitize_ai_error(exc):
        return str(exc).replace("\n", " ").strip()

    def _chat_ai(
        provider, api_key, messages, max_tokens=1200, temperature=0.2, response_format=None
    ):
        try:
            return app.state.ai_client.chat(
                provider,
                api_key,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )
        except TypeError as exc:
            if response_format is None or "response_format" not in str(exc):
                raise
            return app.state.ai_client.chat(
                provider,
                api_key,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    def _stream_chat_ai(
        provider,
        api_key,
        messages,
        max_tokens=1200,
        temperature=0.2,
        response_format=None,
    ):
        try:
            return app.state.ai_client.stream_chat(
                provider,
                api_key,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )
        except TypeError as exc:
            if response_format is None or "response_format" not in str(exc):
                raise
            return app.state.ai_client.stream_chat(
                provider,
                api_key,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    def _ai_completion_events(
        provider,
        api_key,
        messages,
        max_tokens=1200,
        temperature=0.2,
        response_format=None,
    ):
        answer_parts = []
        response = {
            "provider": provider.get("provider"),
            "model": provider.get("model"),
            "usage": {},
        }
        stream = _stream_chat_ai(
            provider,
            api_key,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )
        try:
            for event in stream:
                event_type = event.get("type")
                if event_type == "delta":
                    content = str(event.get("content") or "")
                    if content:
                        answer_parts.append(content)
                    yield {"type": "heartbeat"}
                elif event_type == "heartbeat":
                    yield {"type": "heartbeat"}
                elif event_type == "done":
                    response.update(
                        {
                            "provider": event.get("provider") or response["provider"],
                            "model": event.get("model") or response["model"],
                            "usage": event.get("usage")
                            if isinstance(event.get("usage"), dict)
                            else {},
                            "raw_id": event.get("raw_id"),
                        }
                    )
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        answer = "".join(answer_parts).strip()
        if not answer:
            raise AiProviderError("AI provider returned an empty answer")
        response["answer"] = answer
        yield {"type": "completion", "data": response}

    def _agent_stream_error(exc):
        if isinstance(exc, HTTPException):
            detail = exc.detail
            if isinstance(detail, dict):
                return {
                    "code": str(detail.get("code") or f"http_{exc.status_code}"),
                    "detail": str(
                        detail.get("message")
                        or detail.get("detail")
                        or detail.get("code")
                        or "Request failed"
                    ),
                }
            return {"code": f"http_{exc.status_code}", "detail": str(detail or "Request failed")}
        return {"code": "ai_provider_error", "detail": _sanitize_ai_error(exc)}

    def _agent_generation_stream(events):
        def event_stream():
            try:
                for event in events:
                    if event.get("type") == "heartbeat":
                        yield ": ping\n\n"
                        continue
                    yield _sse_frame(event["type"], event.get("data") or {})
            except GeneratorExit:
                raise
            except Exception as exc:
                yield _sse_frame("error", _agent_stream_error(exc))
            finally:
                close = getattr(events, "close", None)
                if callable(close):
                    close()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    def _bounded_agent_chat_messages(messages, message_limit=40, char_limit=60000):
        bounded = []
        remaining = char_limit
        for message in reversed(list(messages or [])):
            if len(bounded) >= message_limit or remaining <= 0:
                break
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip()
            content = str(message.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            if len(content) > remaining:
                content = content[-remaining:]
            bounded.append({"role": role, "content": content})
            remaining -= len(content)
        bounded.reverse()
        return bounded

    def _disabled_web_search():
        return {
            "requested": False,
            "status": "disabled",
            "query": "",
            "provider": "tavily",
            "credits": 0,
            "reason": "",
            "sources": [],
        }

    def _unavailable_web_search(query, reason):
        return {
            "requested": True,
            "status": "unavailable",
            "query": str(query or ""),
            "provider": "tavily",
            "credits": 0,
            "reason": str(reason or "unavailable"),
            "sources": [],
        }

    def _perform_web_search(messages):
        query = build_web_search_query(messages)
        if not query:
            return _unavailable_web_search("", "empty_query")
        status = app.state.web_search_client.status()
        if not status.get("available"):
            return _unavailable_web_search(query, status.get("reason") or "unavailable")
        try:
            return app.state.web_search_client.search(query)
        except WebSearchUnavailable as exc:
            return _unavailable_web_search(query, exc.reason)

    def _agent_candidate_config(candidate):
        candidate = candidate if isinstance(candidate, dict) else {}
        raw_config = candidate.get("config")
        if not isinstance(raw_config, dict):
            return {}, []
        cleaned = {key: value for key, value in raw_config.items() if key in COMMON_FIELDS}
        dropped = sorted(str(key) for key in raw_config if key not in COMMON_FIELDS)
        return cleaned, dropped

    def _agent_strategy_preference(payload):
        value = str((payload or {}).get("strategy_preference") or "balanced").strip().lower()
        if value not in STRATEGY_PREFERENCES:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "agent_strategy_preference_invalid",
                    "message": "strategy_preference must be conservative, balanced, or aggressive",
                },
            )
        return value

    def _agent_diagnostic_source(payload, user, dataset, task_type):
        task_type = normalize_task_type(task_type)
        raw_source = (payload or {}).get("diagnostic_source")
        if raw_source in [None, ""]:
            return None
        if not isinstance(raw_source, dict):
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "agent_diagnostic_source_invalid",
                    "message": "diagnostic_source must be an object",
                },
            )
        experiment_id = str(raw_source.get("experiment_id") or "").strip()
        message_id = str(raw_source.get("message_id") or "").strip()
        if not experiment_id or not message_id:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "agent_diagnostic_source_invalid",
                    "message": "diagnostic_source requires experiment_id and message_id",
                },
            )
        experiment = _require_experiment(experiment_id, user)
        if experiment.get("dataset_id") != dataset.get("id") or normalize_task_type(
            experiment.get("task_type")
        ) != normalize_task_type(task_type):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "agent_diagnostic_source_context_mismatch",
                    "message": "diagnostic source must match the plan task and dataset",
                },
            )
        message = store.get_agent_chat_message(
            message_id,
            user=user,
            scope="experiment_analysis",
            experiment_id=experiment_id,
        )
        if not message:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "agent_diagnostic_source_not_found",
                    "message": "diagnostic source report was not found",
                },
            )
        report = (message.get("payload") or {}).get("report")
        if (
            message.get("role") != "assistant"
            or message.get("kind") != "report"
            or message.get("status") != "complete"
            or not isinstance(report, dict)
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "agent_diagnostic_source_not_ready",
                    "message": "diagnostic source must be a completed structured report",
                },
            )

        def compact_items(value, allowed_keys, limit):
            if not isinstance(value, list):
                return []
            return [
                {key: item.get(key) for key in allowed_keys if key in item}
                for item in value[:limit]
                if isinstance(item, dict)
            ]

        return {
            "experiment_id": experiment_id,
            "experiment_name": experiment.get("name") or experiment_id,
            "message_id": message_id,
            "conversation_id": message.get("conversation_id"),
            "captured_at": (message.get("context_manifest") or {}).get("captured_at"),
            "provider_config_id": message.get("provider_config_id"),
            "report": {
                "status": report.get("status"),
                "summary": str(report.get("summary") or "")[:8000],
                "evidence": compact_items(
                    report.get("evidence"),
                    ["label", "value", "interpretation", "source_section"],
                    10,
                ),
                "findings": compact_items(
                    report.get("findings"),
                    ["severity", "title", "detail", "source_sections"],
                    10,
                ),
                "recommendations": compact_items(
                    report.get("recommendations"),
                    ["id", "title", "rationale", "priority", "success_criteria"],
                    3,
                ),
                "limitations": [
                    str(item)[:1000]
                    for item in (
                        report.get("limitations")
                        if isinstance(report.get("limitations"), list)
                        else []
                    )[:10]
                ],
            },
        }

    def _agent_preflight_exhausted_detail(attempts, issues):
        safe_issues = [str(issue).strip()[:500] for issue in issues or [] if str(issue).strip()]
        return {
            "code": "agent_preflight_exhausted",
            "message": "Agent could not produce a fully Preflight-valid plan after repair attempts",
            "attempts": attempts,
            "issues": safe_issues[:20],
        }

    def _agent_experiment_name(candidate, index):
        candidate = candidate if isinstance(candidate, dict) else {}
        lane = (
            str(candidate.get("lane") or f"candidate-{index + 1}").strip()
            or f"candidate-{index + 1}"
        )
        title = str(candidate.get("title") or lane).strip() or lane
        name = f"Agent {lane}: {title}"
        return name[:180]

    def _agent_plan_created_experiment_ids(plan):
        execution = plan.get("execution") if isinstance(plan.get("execution"), dict) else {}
        ids = []
        for entry in execution.get("created_experiments") or []:
            if not isinstance(entry, dict):
                continue
            experiment = (
                entry.get("experiment") if isinstance(entry.get("experiment"), dict) else {}
            )
            experiment_id = str(entry.get("experiment_id") or experiment.get("id") or "").strip()
            if experiment_id and experiment_id not in ids:
                ids.append(experiment_id)
        return ids

    def _agent_cancel_action_for_experiment(plan_id, action, experiment):
        action_id = str(action.get("id") or "").strip()
        experiment_id = experiment["id"]
        return {
            "id": f"cancel-{experiment_id[:10]}",
            "type": "cancel_experiment",
            "lane": str(action.get("lane") or "cancel"),
            "title": f"Cancel {experiment.get('name') or experiment_id}",
            "rationale": "Stop this queued or running experiment created by the approved Agent plan.",
            "expected_tradeoffs": ["Frees the runner slot", "Loses unfinished training progress"],
            "objective_alignment": str(action.get("objective_alignment") or ""),
            "baseline_changes": [],
            "risk_level": "medium",
            "verification_steps": ["Confirm the experiment status becomes stopped."],
            "requires_approval": True,
            "status": "pending",
            "result": {},
            "valid": True,
            "target_experiment_id": experiment_id,
            "source_action_id": action_id,
            "created_by_plan_id": plan_id,
        }

    def _agent_merge_execution(
        plan, executed_action_ids, created_experiments, canceled_experiments, warnings=None
    ):
        execution = dict(plan.get("execution") or {})
        existing_ids = list(execution.get("executed_action_ids") or [])
        for action_id in executed_action_ids:
            if action_id not in existing_ids:
                existing_ids.append(action_id)
        execution["executed_action_ids"] = existing_ids
        execution.setdefault("created_experiments", [])
        execution.setdefault("canceled_experiments", [])
        execution.setdefault("warnings", [])
        execution["created_experiments"].extend(created_experiments)
        execution["canceled_experiments"].extend(canceled_experiments)
        for warning in warnings or []:
            if warning and warning not in execution["warnings"]:
                execution["warnings"].append(warning)
        return execution

    def _agent_create_experiment_for_action(
        plan,
        action,
        dataset,
        user,
        *,
        name=None,
        experiment_id=None,
        generated_name_fields=None,
    ):
        preflight = action.get("preflight") if isinstance(action.get("preflight"), dict) else {}
        if not preflight.get("valid"):
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "agent_plan_action_validation_failed",
                    "message": "agent action did not pass preflight",
                    "action_id": action.get("id"),
                    "preflight": preflight,
                },
            )
        experiment_id = experiment_id or uuid.uuid4().hex
        experiment_dir = paths.experiments / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        experiment = store.create_experiment(
            str(name or _agent_experiment_name(action, 0)).strip()[:180],
            dataset["id"],
            plan["task_type"],
            preflight["normalized_config"],
            experiment_dir / "train.log",
            experiment_dir / "artifact",
            experiment_id=experiment_id,
            owner_user_id=user["id"] if user else None,
            **(generated_name_fields or {}),
        )
        runner.submit(experiment["id"])
        return experiment

    def _agent_experiment_status_counts(experiments):
        counts = {
            "total": len(experiments),
            "queued": 0,
            "running": 0,
            "succeeded": 0,
            "failed": 0,
            "canceled": 0,
            "missing": 0,
        }
        for experiment in experiments:
            status = str((experiment or {}).get("status") or "missing").strip() or "missing"
            counts[status] = int(counts.get(status, 0)) + 1
        return counts

    def _agent_refresh_execution(plan, user):
        execution = dict(plan.get("execution") or {})
        created_entries = []
        refreshed_experiments = []
        for entry in execution.get("created_experiments") or []:
            if not isinstance(entry, dict):
                continue
            experiment = (
                entry.get("experiment") if isinstance(entry.get("experiment"), dict) else {}
            )
            experiment_id = str(entry.get("experiment_id") or experiment.get("id") or "").strip()
            if not experiment_id:
                continue
            current = store.get_experiment(experiment_id, user=user)
            refreshed_entry = dict(entry)
            if current:
                refreshed_entry["experiment"] = current
                refreshed_experiments.append(current)
            else:
                refreshed_entry["experiment"] = {"id": experiment_id, "status": "missing"}
                refreshed_experiments.append(refreshed_entry["experiment"])
            created_entries.append(refreshed_entry)

        canceled_entries = []
        for entry in execution.get("canceled_experiments") or []:
            if not isinstance(entry, dict):
                continue
            experiment = (
                entry.get("experiment") if isinstance(entry.get("experiment"), dict) else {}
            )
            experiment_id = str(entry.get("experiment_id") or experiment.get("id") or "").strip()
            refreshed_entry = dict(entry)
            if experiment_id:
                current = store.get_experiment(experiment_id, user=user)
                if current:
                    refreshed_entry["experiment"] = current
            canceled_entries.append(refreshed_entry)

        counts = _agent_experiment_status_counts(refreshed_experiments)
        unfinished_statuses = {"queued", "running"}
        unfinished = [
            {
                "id": experiment.get("id"),
                "name": experiment.get("name"),
                "status": experiment.get("status"),
            }
            for experiment in refreshed_experiments
            if experiment.get("status") in unfinished_statuses
        ]
        completed = counts["total"] > 0 and not unfinished
        failed = counts["total"] > 0 and counts.get("missing", 0) == counts["total"]
        progress = {
            **counts,
            "completed": completed,
            "unfinished": unfinished,
        }
        execution["created_experiments"] = created_entries
        execution["canceled_experiments"] = canceled_entries
        execution["progress"] = progress
        execution["report_ready"] = bool(completed and not failed)

        status = plan.get("status") or "draft"
        if status in {"executed", "running", "ready_for_report"}:
            status = "ready_for_report" if execution["report_ready"] else "running"
        if failed and status in {"executed", "running", "ready_for_report"}:
            status = "failed"
        return execution, status

    def _agent_refresh_plan(plan, user):
        execution, status = _agent_refresh_execution(plan, user)
        if execution == (plan.get("execution") or {}) and status == plan.get("status"):
            return plan
        return store.update_agent_execution_plan(
            plan["id"],
            status=status,
            execution=execution,
        )

    def _agent_plan_has_review(plan):
        review = plan.get("review") if isinstance(plan.get("review"), dict) else {}
        return bool(str(review.get("summary") or "").strip())

    def _agent_campaign_round_objective(campaign, previous_plans):
        base = (
            str(campaign.get("objective") or "").strip()
            or "Iteratively optimize experiment quality for this dataset."
        )
        reviewed_plans = [
            plan
            for plan in previous_plans
            if isinstance(plan, dict)
            and isinstance(plan.get("review"), dict)
            and _agent_plan_has_review(plan)
        ]
        if not reviewed_plans:
            return base
        last_plan = reviewed_plans[-1]
        review = last_plan.get("review") or {}
        summary = str(review.get("summary") or "").strip()
        recommendations = review.get("recommendations") or review.get("next_recommendations") or []
        recommendations = recommendations if isinstance(recommendations, list) else []
        recommendations = [str(item).strip() for item in recommendations if str(item).strip()][:5]
        risks_payload = review.get("risks") if isinstance(review.get("risks"), list) else []
        risks = [str(item).strip() for item in risks_payload if str(item).strip()][:3]
        parts = [base]
        if summary:
            parts.append(f"Previous round review summary: {summary}")
        if recommendations:
            parts.append("Next-round recommendations: " + "; ".join(recommendations))
        if risks:
            parts.append("Remaining risks to manage: " + "; ".join(risks))
        return "\n\n".join(parts)

    def _agent_maybe_complete_campaign_for_reviewed_plan(plan, user):
        campaign_id = str(plan.get("campaign_id") or "").strip()
        if not campaign_id:
            return
        campaign = store.get_agent_campaign(campaign_id, user=user, include_plans=False)
        if not campaign:
            return
        try:
            max_rounds = int(campaign.get("max_rounds") or 1)
            round_index = int(plan.get("round_index") or 0)
        except (TypeError, ValueError):
            return
        if round_index >= max_rounds and _agent_plan_has_review(plan):
            store.update_agent_campaign(
                campaign_id, current_round_index=round_index, status="completed"
            )

    def _prepare_email_code(user, purpose):
        with store.atomic():
            # A skipped development delivery is uncharged, but cannot bypass a pause.
            if settings.smtp_configured or not store.resources.email_delivery_usage()["remaining"]:
                store.resources.reserve_verification_email()
            code = generate_code()
            store.create_email_code(
                user["id"],
                purpose,
                secret_hash(f"{user['id']}:{purpose}:{code}", settings.session_secret),
                code_expires_at(settings),
            )
        return code

    def _issue_email_code(user, purpose):
        code = _prepare_email_code(user, purpose)
        sent = send_verification_email(user["email"], code, purpose, settings)
        return code, sent

    def _attach_debug_code(payload, code):
        """Expose the plaintext code only where operators expect email to be undeliverable."""
        if settings.debug_codes:
            payload["debug_code"] = code
        return payload

    def _verify_email_code(user, purpose, code):
        record = store.get_active_email_code(user["id"], purpose)
        if not record:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "verification_code_invalid",
                    "message": "verification code is invalid or expired",
                },
            )
        try:
            expired = parse_iso(record["expires_at"]) <= auth_utcnow()
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "verification_code_invalid",
                    "message": "verification code is invalid or expired",
                },
            ) from exc
        if expired:
            store.consume_email_code(record["id"])
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "verification_code_invalid",
                    "message": "verification code is invalid or expired",
                },
            )
        if record["attempts"] >= settings.code_max_attempts:
            store.consume_email_code(record["id"])
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "verification_code_attempt_limit",
                    "message": "verification code attempt limit reached",
                },
            )
        expected = secret_hash(f"{user['id']}:{purpose}:{code}", settings.session_secret)
        if not secrets_compare(expected, record["code_hash"]):
            updated = store.increment_email_code_attempts(record["id"])
            if updated and updated["attempts"] >= settings.code_max_attempts:
                store.consume_email_code(record["id"])
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "verification_code_invalid",
                    "message": "verification code is invalid or expired",
                },
            )
        store.consume_email_code(record["id"])
        return True

    def _auth_response(user=None, extra=None):
        payload = {
            "auth_enabled": settings.enabled,
            "user": _public_user(user),
        }
        if user:
            payload["quota"] = store.get_user_quota(user["id"])
            payload["usage"] = store.get_quota_usage(user["id"])
            payload["resource_counts"] = store.count_user_resources(user["id"])
        if extra:
            payload.update(extra)
        return payload

    @app.post("/api/auth/register")
    def register(payload: dict):
        email = normalize_email(payload.get("email"))
        password = str(payload.get("password") or "")
        preferred_locale = validate_preferred_locale(payload.get("preferred_locale") or "en")
        if not email or "@" not in email:
            raise HTTPException(
                status_code=400,
                detail={"code": "valid_email_required", "message": "valid email is required"},
            )
        if len(password) < 8:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "password_too_short",
                    "message": "password must be at least 8 characters",
                },
            )
        try:
            password_hash = hash_password(password)
            with store.atomic():
                store.resources.register_account()
                user = store.create_user(
                    email,
                    password_hash,
                    role="user",
                    email_verified=False,
                    preferred_locale=preferred_locale,
                )
                code = _prepare_email_code(user, CODE_PURPOSE_EMAIL_VERIFY)
        except ResourceLimitError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail=exc.detail,
                headers={"Retry-After": str(exc.detail["retry_after"])}
                if "retry_after" in exc.detail
                else None,
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": "email_already_registered", "message": str(exc)},
            ) from exc
        sent = send_verification_email(user["email"], code, CODE_PURPOSE_EMAIL_VERIFY, settings)
        extra = {
            "verification_sent": sent,
            "email": user["email"],
        }
        if not sent:
            _attach_debug_code(extra, code)
        return _auth_response(user, extra)

    @app.post("/api/auth/verify-email")
    def verify_email(payload: dict, response: Response):
        email = normalize_email(payload.get("email"))
        code = str(payload.get("code") or "").strip()
        user = store.get_user_by_email(email)
        if not user:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "verification_code_invalid",
                    "message": "verification code is invalid or expired",
                },
            )
        _verify_email_code(user, CODE_PURPOSE_EMAIL_VERIFY, code)
        verified = store.verify_user_email(user["id"])
        token = generate_session_token()
        store.create_session(
            verified["id"],
            secret_hash(token, settings.session_secret),
            session_expires_at(settings),
            "",
        )
        response.set_cookie(SESSION_COOKIE_NAME, token, **cookie_kwargs(settings))
        return _auth_response(verified)

    @app.post("/api/auth/resend-verification")
    def resend_verification(payload: dict, request: Request = None):
        email = normalize_email(payload.get("email"))
        user = store.get_user_by_email(email)
        if not user:
            raise HTTPException(
                status_code=404,
                detail={"code": "user_not_found", "message": "user not found"},
            )
        if user["email_verified"]:
            _audit(
                "auth.verification_resend",
                request=request,
                target_user_id=user["id"],
                metadata={"email": user["email"], "sent": False, "reason": "already_verified"},
            )
            return _auth_response(user, {"verification_sent": False})
        code, sent = _issue_email_code(user, CODE_PURPOSE_EMAIL_VERIFY)
        _audit(
            "auth.verification_resend",
            request=request,
            target_user_id=user["id"],
            metadata={"email": user["email"], "sent": sent},
        )
        extra = {"verification_sent": sent, "email": user["email"]}
        if not sent:
            _attach_debug_code(extra, code)
        return _auth_response(user, extra)

    @app.post("/api/auth/login")
    def login(payload: dict, response: Response, request: Request = None):
        email = normalize_email(payload.get("email"))
        password = str(payload.get("password") or "")
        user = store.get_user_by_email(email)
        if not user or not verify_password(password, user["password_hash"]):
            _audit(
                "auth.login_failed",
                request=request,
                target_user_id=user["id"] if user else None,
                metadata={"email": email, "reason": "invalid_credentials"},
            )
            raise HTTPException(
                status_code=401,
                detail={
                    "code": "invalid_credentials",
                    "message": "invalid email or password",
                },
            )
        if user["status"] != "active":
            _audit(
                "auth.login_failed",
                request=request,
                target_user_id=user["id"],
                metadata={"email": email, "reason": "account_disabled"},
            )
            raise HTTPException(
                status_code=403,
                detail={"code": "account_disabled", "message": "account is disabled"},
            )
        if not user["email_verified"]:
            _audit(
                "auth.login_failed",
                request=request,
                target_user_id=user["id"],
                metadata={"email": email, "reason": "email_unverified"},
            )
            raise HTTPException(
                status_code=403,
                detail={"code": "email_not_verified", "message": "email is not verified"},
            )
        token = generate_session_token()
        store.create_session(
            user["id"],
            secret_hash(token, settings.session_secret),
            session_expires_at(settings),
            "",
        )
        response.set_cookie(SESSION_COOKIE_NAME, token, **cookie_kwargs(settings))
        return _auth_response(store.get_user(user["id"]))

    @app.post("/api/auth/logout")
    def logout(response: Response, request: Request = None):
        if request is not None:
            token = request.cookies.get(SESSION_COOKIE_NAME)
            if token:
                store.revoke_session(secret_hash(token, settings.session_secret))
        response.delete_cookie(SESSION_COOKIE_NAME, **clear_cookie_kwargs(settings))
        return {"ok": True}

    @app.get("/api/auth/me")
    def me(request: Request = None):
        if not settings.enabled:
            return _auth_response(None)
        return _auth_response(_current_user(request))

    @app.patch("/api/auth/profile")
    def update_profile(payload: dict, request: Request = None):
        user = _current_profile_user(request)
        display_name = validate_display_name(payload.get("display_name"))
        updated = store.update_user_profile(user["id"], display_name)
        return _auth_response(updated)

    @app.patch("/api/auth/preferences")
    def update_preferences(payload: dict, request: Request = None):
        user = _current_profile_user(request)
        preferred_locale = validate_preferred_locale(payload.get("preferred_locale"))
        updated = store.update_user_locale(user["id"], preferred_locale)
        return _auth_response(updated)

    @app.get("/api/ai/providers")
    def list_ai_providers(request: Request = None):
        user = _current_user(request)
        return [
            _public_ai_provider(provider) for provider in store.list_ai_provider_configs(user=user)
        ]

    @app.get("/api/ai/provider-presets")
    def ai_provider_presets():
        return provider_presets()

    @app.post("/api/ai/providers")
    def create_ai_provider(payload: dict, request: Request = None):
        user = _current_user(request)
        try:
            normalized = normalize_ai_provider_payload(payload, require_api_key=True)
            encrypted_key = encrypt_api_key(normalized["api_key"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except AiConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        provider = store.create_ai_provider_config(
            user["id"] if user else None,
            normalized["provider"],
            normalized["display_name"],
            normalized["base_url"],
            normalized["model"],
            encrypted_key,
            api_key_last4(normalized["api_key"]),
            enabled=normalized["enabled"],
            capabilities=normalized["capabilities"],
        )
        _audit(
            "ai.provider_created",
            request=request,
            actor=user,
            metadata={
                "provider_id": provider["id"],
                "provider": provider["provider"],
                "model": provider["model"],
                "enabled": provider["enabled"],
                "capabilities": provider.get("capabilities") or {},
            },
        )
        return _ai_provider_response(provider)

    @app.patch("/api/ai/providers/{provider_id}")
    def update_ai_provider(provider_id: str, payload: dict, request: Request = None):
        user = _current_user(request)
        existing = _require_ai_provider(provider_id, user, include_secret=True)
        try:
            normalized = normalize_ai_provider_payload(payload, existing=existing)
            updates = {
                "provider": normalized["provider"],
                "display_name": normalized["display_name"],
                "base_url": normalized["base_url"],
                "model": normalized["model"],
                "enabled": normalized["enabled"],
                "capabilities": normalized["capabilities"],
            }
            if normalized["api_key"]:
                updates["api_key_ciphertext"] = encrypt_api_key(normalized["api_key"])
                updates["api_key_last4"] = api_key_last4(normalized["api_key"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except AiConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        provider = store.update_ai_provider_config(provider_id, updates, user=user)
        _audit(
            "ai.provider_updated",
            request=request,
            actor=user,
            metadata={
                "provider_id": provider_id,
                "provider": provider["provider"] if provider else existing["provider"],
                "model": provider["model"] if provider else existing["model"],
                "api_key_rotated": bool(normalized["api_key"]),
                "enabled": provider["enabled"] if provider else existing["enabled"],
            },
        )
        return _ai_provider_response(provider)

    @app.delete("/api/ai/providers/{provider_id}")
    def delete_ai_provider(provider_id: str, request: Request = None):
        user = _current_user(request)
        provider = store.delete_ai_provider_config(provider_id, user=user)
        if not provider:
            raise HTTPException(status_code=404, detail="AI provider config not found")
        _audit(
            "ai.provider_deleted",
            request=request,
            actor=user,
            metadata={
                "provider_id": provider_id,
                "provider": provider["provider"],
                "model": provider["model"],
            },
        )
        return {"provider_id": provider_id, "deleted": True}

    @app.post("/api/ai/providers/{provider_id}/test")
    def test_ai_provider(provider_id: str, request: Request = None):
        user = _current_user(request)
        provider = _require_ai_provider(provider_id, user, include_secret=True)
        if not provider["enabled"]:
            raise HTTPException(status_code=400, detail="AI provider config is disabled")
        try:
            api_key = decrypt_api_key(provider["api_key_ciphertext"])
            response = app.state.ai_client.chat(
                provider,
                api_key,
                [
                    {"role": "system", "content": "You are a connection test for PepPlatform."},
                    {"role": "user", "content": "Reply with a short OK message."},
                ],
                max_tokens=32,
                temperature=0,
            )
        except AiConfigurationError as exc:
            _audit(
                "ai.provider_test",
                request=request,
                actor=user,
                metadata={
                    "provider_id": provider_id,
                    "status": "failed",
                    "reason": "configuration",
                },
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except AiProviderError as exc:
            _audit(
                "ai.provider_test",
                request=request,
                actor=user,
                metadata={
                    "provider_id": provider_id,
                    "status": "failed",
                    "reason": "provider_error",
                },
            )
            raise HTTPException(status_code=502, detail=_sanitize_ai_error(exc)) from exc
        _audit(
            "ai.provider_test",
            request=request,
            actor=user,
            metadata={
                "provider_id": provider_id,
                "status": "succeeded",
                "model": response.get("model"),
            },
        )
        return {
            "ok": True,
            "provider": provider["provider"],
            "model": response.get("model"),
            "answer_preview": markdown_to_plain_summary(response.get("answer"), limit=80),
            "usage": response.get("usage") or {},
        }

    def _agent_generate_execution_plan_event_core(
        request,
        user,
        provider,
        provider_id,
        dataset,
        task_type,
        objective,
        language,
        max_tokens,
        context,
        manifest,
        strategy_preference="balanced",
        campaign_id=None,
        round_index=0,
        audit_extra=None,
    ):
        task_type = normalize_task_type(task_type)
        messages, warnings = build_agent_execution_plan_messages(
            context,
            manifest,
            objective,
            language,
            strategy_preference,
        )
        repair_attempted = False
        repair_succeeded = False
        repair_issues = []
        attempts = 0
        try:
            api_key = decrypt_api_key(provider["api_key_ciphertext"])
            actions = []
            action_warnings = []
            previous_answer = ""
            response = {}
            for attempt_index in range(3):
                if attempt_index:
                    messages, _repair_warnings = build_agent_execution_plan_repair_messages(
                        context,
                        manifest,
                        objective,
                        language,
                        previous_answer,
                        repair_issues,
                        strategy_preference,
                    )
                yield {
                    "type": "stage",
                    "data": {
                        "stage": "repairing" if attempt_index else "generating",
                        "attempt": attempt_index + 1,
                        "max_attempts": 3,
                    },
                }
                response = {}
                for event in _ai_completion_events(
                    provider,
                    api_key,
                    messages,
                    max_tokens=max_tokens,
                    temperature=0.2 if attempt_index == 0 else 0.1,
                    response_format={"type": "json_object"},
                ):
                    if event.get("type") == "completion":
                        response = event["data"]
                    else:
                        yield event
                attempts += 1
                previous_answer = response.get("answer") or ""
                yield {
                    "type": "stage",
                    "data": {"stage": "validating", "attempt": attempts, "max_attempts": 3},
                }
                try:
                    actions, action_warnings = parse_agent_execution_actions(
                        previous_answer, dataset, task_type
                    )
                    repair_issues = validate_execution_actions_for_repair(
                        actions,
                        action_warnings,
                        strategy_preference,
                        require_three=campaign_id is None,
                    )
                except AgentResponseError as exc:
                    repair_issues = [str(exc)]
                if not repair_issues:
                    break

            repair_attempted = attempts > 1
            repair_succeeded = repair_attempted and not repair_issues
            if repair_issues:
                _audit(
                    "ai.agent_plan_created",
                    request=request,
                    actor=user,
                    metadata={
                        "dataset_id": dataset["id"],
                        "provider_id": provider_id,
                        "status": "failed",
                        "reason": "preflight_exhausted",
                        "attempts": attempts,
                        **(audit_extra or {}),
                    },
                )
                raise HTTPException(
                    status_code=502,
                    detail=_agent_preflight_exhausted_detail(attempts, repair_issues),
                )
        except AiConfigurationError as exc:
            _audit(
                "ai.agent_plan_created",
                request=request,
                actor=user,
                metadata={
                    "dataset_id": dataset["id"],
                    "provider_id": provider_id,
                    "status": "failed",
                    "reason": "configuration",
                    **(audit_extra or {}),
                },
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except AiProviderError as exc:
            _audit(
                "ai.agent_plan_created",
                request=request,
                actor=user,
                metadata={
                    "dataset_id": dataset["id"],
                    "provider_id": provider_id,
                    "status": "failed",
                    "reason": "provider_error",
                    **(audit_extra or {}),
                },
            )
            raise HTTPException(status_code=502, detail=_sanitize_ai_error(exc)) from exc
        except AgentResponseError as exc:
            _audit(
                "ai.agent_plan_created",
                request=request,
                actor=user,
                metadata={
                    "dataset_id": dataset["id"],
                    "provider_id": provider_id,
                    "status": "failed",
                    "reason": "invalid_response",
                    **(audit_extra or {}),
                },
            )
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        warnings.extend(action_warnings)
        warnings = list(dict.fromkeys(str(warning) for warning in warnings if str(warning).strip()))
        trace = {
            "repair_attempted": repair_attempted,
            "repair_succeeded": repair_succeeded,
            "repair_issues": repair_issues,
            "attempts": attempts,
            "strategy_preference": strategy_preference,
            "action_count": len(actions),
            "valid_action_count": sum(1 for action in actions if action.get("valid")),
            "context_chars": manifest.get("context_chars"),
        }
        diagnostic_source = context.get("diagnostic_source") if isinstance(context, dict) else None
        if isinstance(diagnostic_source, dict):
            trace["diagnostic_source"] = {
                "experiment_id": diagnostic_source.get("experiment_id"),
                "message_id": diagnostic_source.get("message_id"),
                "conversation_id": diagnostic_source.get("conversation_id"),
            }
        yield {"type": "stage", "data": {"stage": "saving", "attempt": attempts, "max_attempts": 3}}
        plan = store.create_agent_execution_plan(
            user["id"] if user else None,
            provider_id,
            provider["provider"],
            response.get("model") or provider["model"],
            task_type,
            dataset["id"],
            objective,
            actions,
            usage=response.get("usage") or {},
            context_manifest=manifest,
            warnings=warnings,
            campaign_id=campaign_id,
            round_index=round_index,
            trace=trace,
        )
        _audit(
            "ai.agent_plan_created",
            request=request,
            actor=user,
            metadata={
                "plan_id": plan["id"],
                "dataset_id": dataset["id"],
                "task_type": task_type,
                "provider_id": provider_id,
                "status": "succeeded",
                "provider": provider["provider"],
                "model": response.get("model"),
                "action_count": len(actions),
                "valid_action_count": trace["valid_action_count"],
                "repair_attempted": repair_attempted,
                "repair_succeeded": repair_succeeded,
                "objective_summary": markdown_to_plain_summary(objective),
                **(audit_extra or {}),
            },
        )
        plan["repair_attempted"] = repair_attempted
        plan["repair_succeeded"] = repair_succeeded
        plan["context_summary"] = experiment_design_context_summary(context)
        yield {"type": "done", "data": {"plan": plan}}

    def _agent_generate_execution_plan_events(
        request,
        user,
        provider,
        provider_id,
        dataset,
        task_type,
        objective,
        language,
        max_tokens,
        context,
        manifest,
        strategy_preference="balanced",
        campaign_id=None,
        round_index=0,
        audit_extra=None,
    ):
        task_type = normalize_task_type(task_type)
        completed = False
        attempts = 0
        events = _agent_generate_execution_plan_event_core(
            request,
            user,
            provider,
            provider_id,
            dataset,
            task_type,
            objective,
            language,
            max_tokens,
            context,
            manifest,
            strategy_preference=strategy_preference,
            campaign_id=campaign_id,
            round_index=round_index,
            audit_extra=audit_extra,
        )
        try:
            for event in events:
                if event.get("type") == "stage":
                    attempts = max(attempts, int((event.get("data") or {}).get("attempt") or 0))
                if event.get("type") == "done":
                    completed = True
                yield event
        except GeneratorExit:
            if not completed:
                _audit(
                    "ai.agent_plan_created",
                    request=request,
                    actor=user,
                    metadata={
                        "dataset_id": dataset["id"],
                        "provider_id": provider_id,
                        "status": "stopped",
                        "attempts": attempts,
                        **(audit_extra or {}),
                    },
                )
            raise
        finally:
            close = getattr(events, "close", None)
            if callable(close):
                close()

    def _agent_generate_and_store_execution_plan(*args, **kwargs):
        plan = None
        for event in _agent_generate_execution_plan_events(*args, **kwargs):
            if event.get("type") == "done":
                plan = (event.get("data") or {}).get("plan")
        if not plan:
            raise HTTPException(
                status_code=502, detail="agent plan generation ended without a plan"
            )
        return plan

    @app.post("/api/agent/experiment-design")
    def agent_experiment_design(payload: dict, request: Request = None):
        user = _current_user(request)
        provider_id = str((payload or {}).get("provider_config_id") or "").strip()
        if not provider_id:
            raise HTTPException(status_code=400, detail="provider_config_id is required")
        provider = _require_ai_provider(provider_id, user, include_secret=True)
        if not provider["enabled"]:
            raise HTTPException(status_code=400, detail="AI provider config is disabled")
        dataset = _require_dataset((payload or {}).get("dataset_id"), user)
        task_type = (
            str((payload or {}).get("task_type") or dataset["task_type"]).strip()
            or dataset["task_type"]
        )
        objective = str((payload or {}).get("objective") or "").strip() or (
            "Recommend the next three experiment configurations for this dataset."
        )
        include = (payload or {}).get("include")
        include = include if isinstance(include, dict) else {}
        baseline_experiment_ids = (payload or {}).get("baseline_experiment_ids")
        baseline_experiment_ids = (
            baseline_experiment_ids if isinstance(baseline_experiment_ids, list) else []
        )
        language = str((payload or {}).get("language") or "zh").strip() or "zh"
        strategy_preference = _agent_strategy_preference(payload)
        try:
            max_tokens = max(512, min(int((payload or {}).get("max_tokens") or 2200), 6000))
        except (TypeError, ValueError):
            max_tokens = 2200

        context, manifest = build_experiment_design_context(
            store,
            dataset,
            task_type,
            user=user,
            include=include,
            baseline_experiment_ids=baseline_experiment_ids,
        )
        messages, warnings = build_experiment_design_messages(
            context,
            manifest,
            objective,
            language,
            strategy_preference,
        )
        repair_attempted = False
        repair_succeeded = False
        repair_issues = []
        attempts = 0
        try:
            api_key = decrypt_api_key(provider["api_key_ciphertext"])
            candidates = []
            candidate_warnings = []
            previous_answer = ""
            response = {}
            for attempt_index in range(3):
                if attempt_index:
                    messages, _repair_message_warnings = build_experiment_design_repair_messages(
                        context,
                        manifest,
                        objective,
                        language,
                        previous_answer,
                        repair_issues,
                        strategy_preference,
                    )
                response = _chat_ai(
                    provider,
                    api_key,
                    messages,
                    max_tokens=max_tokens,
                    temperature=0.2 if attempt_index == 0 else 0.1,
                    response_format={"type": "json_object"},
                )
                attempts += 1
                previous_answer = response.get("answer") or ""
                try:
                    candidates, candidate_warnings = parse_experiment_design_candidates(
                        previous_answer,
                        dataset,
                        task_type,
                    )
                    repair_issues = validate_design_candidates_for_repair(
                        candidates,
                        candidate_warnings,
                        strategy_preference,
                    )
                except AgentResponseError as exc:
                    repair_issues = [str(exc)]
                if not repair_issues:
                    break

            repair_attempted = attempts > 1
            repair_succeeded = repair_attempted and not repair_issues
            if repair_issues:
                _audit(
                    "ai.experiment_design",
                    request=request,
                    actor=user,
                    metadata={
                        "dataset_id": dataset["id"],
                        "provider_id": provider_id,
                        "status": "failed",
                        "reason": "preflight_exhausted",
                        "attempts": attempts,
                    },
                )
                raise HTTPException(
                    status_code=502,
                    detail=_agent_preflight_exhausted_detail(attempts, repair_issues),
                )
        except AiConfigurationError as exc:
            _audit(
                "ai.experiment_design",
                request=request,
                actor=user,
                metadata={
                    "dataset_id": dataset["id"],
                    "provider_id": provider_id,
                    "status": "failed",
                    "reason": "configuration",
                },
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except AiProviderError as exc:
            _audit(
                "ai.experiment_design",
                request=request,
                actor=user,
                metadata={
                    "dataset_id": dataset["id"],
                    "provider_id": provider_id,
                    "status": "failed",
                    "reason": "provider_error",
                },
            )
            raise HTTPException(status_code=502, detail=_sanitize_ai_error(exc)) from exc
        except AgentResponseError as exc:
            _audit(
                "ai.experiment_design",
                request=request,
                actor=user,
                metadata={
                    "dataset_id": dataset["id"],
                    "provider_id": provider_id,
                    "status": "failed",
                    "reason": "invalid_response",
                },
            )
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        warnings.extend(candidate_warnings)
        warnings = list(dict.fromkeys(str(warning) for warning in warnings if str(warning).strip()))
        _audit(
            "ai.experiment_design",
            request=request,
            actor=user,
            metadata={
                "dataset_id": dataset["id"],
                "task_type": task_type,
                "provider_id": provider_id,
                "status": "succeeded",
                "provider": provider["provider"],
                "model": response.get("model"),
                "candidate_count": len(candidates),
                "valid_candidate_count": sum(
                    1 for candidate in candidates if candidate.get("valid")
                ),
                "repair_attempted": repair_attempted,
                "repair_succeeded": repair_succeeded,
                "objective_summary": markdown_to_plain_summary(objective),
            },
        )
        return {
            "candidates": candidates,
            "provider": provider["provider"],
            "model": response.get("model") or provider["model"],
            "usage": response.get("usage") or {},
            "context_manifest": manifest,
            "context_summary": experiment_design_context_summary(context),
            "repair_attempted": repair_attempted,
            "repair_succeeded": repair_succeeded,
            "strategy_preference": strategy_preference,
            "attempts": attempts,
            "warnings": warnings,
        }

    @app.post("/api/agent/chat")
    def agent_chat(payload: dict, request: Request = None):
        user = _current_user(request)
        provider_id = str((payload or {}).get("provider_config_id") or "").strip()
        if not provider_id:
            raise HTTPException(status_code=400, detail="provider_config_id is required")
        provider = _require_ai_provider(provider_id, user, include_secret=True)
        if not provider["enabled"]:
            raise HTTPException(status_code=400, detail="AI provider config is disabled")
        messages = (payload or {}).get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(
                status_code=400, detail="messages is required and must be a non-empty list"
            )
        language = str((payload or {}).get("language") or "zh").strip() or "zh"
        try:
            max_tokens = max(256, min(int((payload or {}).get("max_tokens") or 2000), 8000))
        except (TypeError, ValueError):
            max_tokens = 2000
        try:
            temperature_val = float((payload or {}).get("temperature") or 0.3)
            temperature_val = max(0.0, min(2.0, temperature_val))
        except (TypeError, ValueError):
            temperature_val = 0.3

        system_prompt = build_agent_chat_system_prompt(language)
        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(_bounded_agent_chat_messages(messages))
        web_search = _disabled_web_search()
        if bool((payload or {}).get("web_search")):
            web_search = _perform_web_search(full_messages)
            full_messages.append(
                {
                    "role": "system",
                    "content": build_web_search_system_message(web_search, language=language),
                }
            )

        try:
            api_key = decrypt_api_key(provider["api_key_ciphertext"])
            response = _chat_ai(
                provider,
                api_key,
                full_messages,
                max_tokens=max_tokens,
                temperature=temperature_val,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"AI provider error: {exc}") from exc

        return {
            "reply": response.get("answer") or "",
            "finish_reason": response.get("finish_reason") or "stop",
            "provider": response.get("provider") or provider["provider"],
            "model": response.get("model") or provider["model"],
            "usage": response.get("usage") or {},
            "web_search": web_search,
        }

    @app.get("/api/agent/web-search/status")
    def agent_web_search_status(request: Request = None):
        _current_user(request)
        return app.state.web_search_client.status()

    @app.get("/api/agent/chat/conversations")
    def list_agent_chat_conversations(
        request: Request = None,
        q: str = "",
        limit: int = 50,
        offset: int = 0,
    ):
        user = _current_user(request)
        return store.list_agent_chat_conversations(user=user, query=q, limit=limit, offset=offset)

    @app.get("/api/agent/chat/conversations/{conversation_id}")
    def get_agent_chat_conversation(
        conversation_id: str,
        request: Request = None,
        limit: int = 100,
        before_ordinal: int = None,
    ):
        result = store.list_agent_chat_messages(
            conversation_id,
            user=_current_user(request),
            limit=limit,
            before_ordinal=before_ordinal,
        )
        if not result:
            raise HTTPException(status_code=404, detail="conversation not found")
        return result

    @app.patch("/api/agent/chat/conversations/{conversation_id}")
    def update_agent_chat_conversation(
        conversation_id: str, payload: dict, request: Request = None
    ):
        updates = {}
        if "title" in (payload or {}):
            updates["title"] = payload.get("title")
        if "pinned" in (payload or {}):
            updates["pinned"] = bool(payload.get("pinned"))
        if not updates:
            raise HTTPException(status_code=400, detail="title or pinned is required")
        try:
            conversation = store.update_agent_chat_conversation(
                conversation_id,
                updates,
                user=_current_user(request),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not conversation:
            raise HTTPException(status_code=404, detail="conversation not found")
        return conversation

    @app.delete("/api/agent/chat/conversations/{conversation_id}")
    def delete_agent_chat_conversation(conversation_id: str, request: Request = None):
        conversation = store.delete_agent_chat_conversation(
            conversation_id, user=_current_user(request)
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="conversation not found")
        return {"deleted": True, "conversation": conversation}

    @app.post("/api/agent/chat/stream")
    def stream_agent_chat(payload: dict, request: Request = None):
        user = _current_user(request)
        provider_id = str((payload or {}).get("provider_config_id") or "").strip()
        if not provider_id:
            raise HTTPException(status_code=400, detail="provider_config_id is required")
        provider = _require_ai_provider(provider_id, user, include_secret=True)
        if not provider["enabled"]:
            raise HTTPException(status_code=400, detail="AI provider config is disabled")
        content = str((payload or {}).get("content") or "").strip()
        if not content:
            raise HTTPException(status_code=400, detail="content is required")
        if len(content) > 120000:
            raise HTTPException(status_code=400, detail="content is too long")
        conversation_id = str((payload or {}).get("conversation_id") or "").strip() or None
        language = str((payload or {}).get("language") or "zh").strip() or "zh"
        web_search_requested = bool((payload or {}).get("web_search"))
        try:
            max_tokens = max(256, min(int((payload or {}).get("max_tokens") or 3000), 8000))
        except (TypeError, ValueError):
            max_tokens = 3000
        try:
            temperature = max(0.0, min(float((payload or {}).get("temperature") or 0.3), 2.0))
        except (TypeError, ValueError):
            temperature = 0.3
        try:
            api_key = decrypt_api_key(provider["api_key_ciphertext"])
            turn = store.begin_agent_chat_turn(
                user.get("id") if user else None,
                provider_id,
                provider["provider"],
                provider["model"],
                content,
                conversation_id=conversation_id,
                user=user,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=404 if conversation_id else 400, detail=str(exc)
            ) from exc
        except AiConfigurationError as exc:
            raise HTTPException(status_code=500, detail=_sanitize_ai_error(exc)) from exc

        assistant_message = turn["assistant_message"]
        context = (
            store.get_agent_chat_context(
                turn["conversation"]["id"],
                assistant_message["ordinal"],
                user=user,
                message_limit=40,
                char_limit=60000,
            )
            or []
        )
        messages = [
            {"role": "system", "content": build_agent_chat_system_prompt(language)},
            *context,
        ]

        def event_stream():
            answer = ""
            finalized = False
            finish_reason = ""
            usage = {}
            response_provider = provider["provider"]
            response_model = provider["model"]
            last_checkpoint = time.monotonic()
            web_search = _disabled_web_search()
            yield _sse_frame("meta", turn)
            try:
                if web_search_requested:
                    query = build_web_search_query(messages)
                    yield _sse_frame(
                        "search",
                        {
                            "message_id": assistant_message["id"],
                            "status": "checking",
                            "query": query,
                        },
                    )
                    status = app.state.web_search_client.status()
                    if not query:
                        web_search = _unavailable_web_search("", "empty_query")
                    elif not status.get("available"):
                        web_search = _unavailable_web_search(
                            query, status.get("reason") or "unavailable"
                        )
                    else:
                        yield _sse_frame(
                            "search",
                            {
                                "message_id": assistant_message["id"],
                                "status": "searching",
                                "query": query,
                            },
                        )
                        try:
                            web_search = app.state.web_search_client.search(query)
                        except WebSearchUnavailable as exc:
                            web_search = _unavailable_web_search(query, exc.reason)
                    store.update_agent_chat_message(
                        assistant_message["id"],
                        user=user,
                        web_search=web_search,
                    )
                    yield _sse_frame(
                        "search",
                        {
                            "message_id": assistant_message["id"],
                            "status": "complete"
                            if web_search.get("status") == "complete"
                            else "unavailable",
                            "web_search": web_search,
                        },
                    )
                    messages.append(
                        {
                            "role": "system",
                            "content": build_web_search_system_message(
                                web_search, language=language
                            ),
                        }
                    )
                for event in app.state.ai_client.stream_chat(
                    provider,
                    api_key,
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    event_type = event.get("type")
                    if event_type == "heartbeat":
                        yield ": ping\n\n"
                        continue
                    if event_type == "delta":
                        delta = str(event.get("content") or "")
                        if not delta:
                            continue
                        answer += delta
                        yield _sse_frame(
                            "delta",
                            {"message_id": assistant_message["id"], "content": delta},
                        )
                        now = time.monotonic()
                        if now - last_checkpoint >= 1.0:
                            store.update_agent_chat_message(
                                assistant_message["id"],
                                user=user,
                                content=answer,
                                status="streaming",
                            )
                            last_checkpoint = now
                        continue
                    if event_type == "done":
                        finish_reason = str(event.get("finish_reason") or "stop")
                        usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
                        response_provider = str(event.get("provider") or response_provider)
                        response_model = str(event.get("model") or response_model)
                message = store.update_agent_chat_message(
                    assistant_message["id"],
                    user=user,
                    content=answer,
                    status="complete",
                    finish_reason=finish_reason or "stop",
                    error="",
                    usage=usage,
                    web_search=web_search,
                    provider=response_provider,
                    model=response_model,
                )
                finalized = True
                yield _sse_frame("done", {"message": message})
            except GeneratorExit:
                raise
            except Exception as exc:
                interrupted = isinstance(exc, AiStreamInterruptedError)
                if interrupted:
                    if str(language).lower().startswith("zh"):
                        detail = (
                            "与 AI 服务的流式连接中断，已保留已生成内容，请重试。"
                            if answer
                            else "与 AI 服务的连接暂时中断，请稍后重试。"
                        )
                    else:
                        detail = (
                            "The AI stream was interrupted. The generated content has been preserved; please try again."
                            if answer
                            else "The connection to the AI service was interrupted. Please try again shortly."
                        )
                else:
                    detail = _sanitize_ai_error(exc)
                message = store.update_agent_chat_message(
                    assistant_message["id"],
                    user=user,
                    content=answer,
                    status="error",
                    finish_reason="interrupted" if interrupted else "error",
                    error=detail,
                    usage=usage,
                    web_search=web_search,
                )
                finalized = True
                yield _sse_frame(
                    "error",
                    {
                        "code": "ai_stream_interrupted" if interrupted else "ai_provider_error",
                        "detail": detail,
                        "message": message,
                    },
                )
            finally:
                if not finalized:
                    store.update_agent_chat_message(
                        assistant_message["id"],
                        user=user,
                        content=answer,
                        status="stopped",
                        finish_reason="stopped",
                        error="",
                        usage=usage,
                        web_search=web_search,
                    )

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/agent/experiment-batches", deprecated=True)
    def agent_experiment_batch(payload: dict, request: Request = None, response: Response = None):
        if response is not None:
            response.headers["Deprecation"] = "true"
            response.headers["Warning"] = (
                '299 - "Batch experiment creation is deprecated; use the Experiment Wizard."'
            )
        user = _current_user(request)
        dataset = _require_dataset((payload or {}).get("dataset_id"), user)
        task_type = (
            str((payload or {}).get("task_type") or dataset["task_type"]).strip()
            or dataset["task_type"]
        )
        objective = str((payload or {}).get("objective") or "").strip()
        candidates_payload = (payload or {}).get("candidates")
        if not isinstance(candidates_payload, list) or not 1 <= len(candidates_payload) <= 3:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "agent_batch_candidate_count_invalid",
                    "message": "candidates must contain between 1 and 3 items",
                },
            )

        prepared = []
        rejected = []
        response_warnings = []
        for index, raw_candidate in enumerate(candidates_payload):
            candidate = raw_candidate if isinstance(raw_candidate, dict) else {}
            lane = (
                str(candidate.get("lane") or f"candidate-{index + 1}").strip()
                or f"candidate-{index + 1}"
            )
            title = str(candidate.get("title") or lane).strip() or lane
            candidate_id = (
                str(candidate.get("id") or f"{lane}-{index + 1}").strip() or f"{lane}-{index + 1}"
            )
            raw_config = candidate.get("config")
            config, dropped_keys = _agent_candidate_config(candidate)
            if not isinstance(raw_config, dict):
                preflight = preflight_experiment(dataset, task_type, {})
                preflight["valid"] = False
                preflight.setdefault("error_codes", []).insert(0, "candidate_config_missing")
                preflight.setdefault("errors", []).insert(0, "candidate config must be an object")
            else:
                preflight = preflight_experiment(dataset, task_type, config)
            if dropped_keys:
                warning = (
                    f"{lane} candidate dropped unsupported config key(s): {', '.join(dropped_keys)}"
                )
                response_warnings.append(warning)
                preflight.setdefault("warning_codes", []).insert(0, "candidate_config_keys_dropped")
                preflight.setdefault("warnings", []).insert(0, warning)
            entry = {
                "id": candidate_id,
                "lane": lane,
                "title": title,
                "config": config,
                "preflight": preflight,
                "valid": bool(preflight.get("valid")),
            }
            if entry["valid"]:
                prepared.append(entry)
            else:
                rejected.append(entry)

        if rejected:
            _audit(
                "ai.experiment_batch_create",
                request=request,
                actor=user,
                metadata={
                    "dataset_id": dataset["id"],
                    "task_type": task_type,
                    "status": "failed",
                    "reason": "preflight",
                    "candidate_count": len(candidates_payload),
                    "invalid_candidate_count": len(rejected),
                    "objective_summary": markdown_to_plain_summary(objective),
                },
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "agent_batch_preflight_failed",
                    "message": "one or more agent candidates failed preflight",
                    "candidates": rejected,
                },
            )

        _check_experiment_quota(user, len(prepared))
        batch_id = uuid.uuid4().hex
        created_experiments = []
        for index, candidate in enumerate(prepared):
            experiment_id = uuid.uuid4().hex
            experiment_dir = paths.experiments / experiment_id
            experiment_dir.mkdir(parents=True, exist_ok=True)
            experiment = store.create_experiment(
                _agent_experiment_name(candidate, index),
                dataset["id"],
                task_type,
                candidate["preflight"]["normalized_config"],
                experiment_dir / "train.log",
                experiment_dir / "artifact",
                experiment_id=experiment_id,
                owner_user_id=user["id"] if user else None,
                name_kind="config",
                name_qualifier=(
                    candidate["lane"] if candidate["lane"] in GENERATED_NAME_QUALIFIERS else None
                ),
            )
            runner.submit(experiment["id"])
            created_experiments.append(
                {
                    "candidate_id": candidate["id"],
                    "lane": candidate["lane"],
                    "title": candidate["title"],
                    "experiment": experiment,
                }
            )
        if settings.enabled and user:
            store.increment_quota_usage(user["id"], experiments_created=len(created_experiments))
        _audit(
            "ai.experiment_batch_create",
            request=request,
            actor=user,
            metadata={
                "batch_id": batch_id,
                "dataset_id": dataset["id"],
                "task_type": task_type,
                "status": "succeeded",
                "candidate_count": len(prepared),
                "experiment_ids": [item["experiment"]["id"] for item in created_experiments],
                "objective_summary": markdown_to_plain_summary(objective),
            },
        )
        return {
            "batch_id": batch_id,
            "created_experiments": created_experiments,
            "warnings": list(dict.fromkeys(response_warnings)),
        }

    @app.post("/api/agent/campaigns")
    def create_agent_campaign(payload: dict, request: Request = None):
        user = _current_user(request)
        dataset = _require_dataset((payload or {}).get("dataset_id"), user)
        task_type = (
            str((payload or {}).get("task_type") or dataset["task_type"]).strip()
            or dataset["task_type"]
        )
        if normalize_task_type(dataset["task_type"]) != normalize_task_type(task_type):
            raise HTTPException(
                status_code=400, detail="dataset task_type does not match campaign task_type"
            )
        objective = str((payload or {}).get("objective") or "").strip() or (
            "Iteratively optimize experiment quality for this dataset."
        )
        try:
            max_rounds = max(1, min(int((payload or {}).get("max_rounds") or 3), 20))
        except (TypeError, ValueError):
            max_rounds = 3
        campaign = store.create_agent_campaign(
            user["id"] if user else None,
            task_type,
            dataset["id"],
            objective,
            max_rounds=max_rounds,
        )
        _audit(
            "ai.agent_campaign_created",
            request=request,
            actor=user,
            metadata={
                "campaign_id": campaign["id"],
                "dataset_id": dataset["id"],
                "task_type": task_type,
                "max_rounds": max_rounds,
                "objective_summary": markdown_to_plain_summary(objective),
            },
        )
        campaign["plans"] = []
        return campaign

    @app.get("/api/agent/campaigns")
    def list_agent_campaigns(
        request: Request = None,
        dataset_id: str = "",
        task_type: str = "",
        status: str = "",
        campaign_id: str = "",
        limit: int = 20,
        offset: int = 0,
    ):
        task_type = normalize_task_type(task_type)
        user = _current_user(request)
        campaigns, total = store.list_agent_campaigns(
            user=user,
            dataset_id=dataset_id or None,
            task_type=task_type or None,
            status=status or None,
            limit=limit,
            offset=offset,
        )
        return {
            "campaigns": campaigns,
            "total": total,
            "limit": max(1, min(int(limit), 100)),
            "offset": max(0, int(offset)),
        }

    @app.get("/api/agent/campaigns/{campaign_id}")
    def get_agent_campaign(campaign_id: str, request: Request = None):
        user = _current_user(request)
        return _require_agent_campaign(campaign_id, user, include_plans=True)

    def _agent_campaign_round_generation(campaign_id, payload, request):
        user = _current_user(request)
        campaign = _require_agent_campaign(campaign_id, user, include_plans=True)
        if campaign.get("status") != "active":
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "agent_campaign_not_active",
                    "message": "agent campaign is not active",
                },
            )
        existing_plans = campaign.get("plans") if isinstance(campaign.get("plans"), list) else []
        next_round_index = (
            max([int(plan.get("round_index") or 0) for plan in existing_plans] or [0]) + 1
        )
        if next_round_index > int(campaign.get("max_rounds") or 1):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "agent_campaign_max_rounds_reached",
                    "message": "agent campaign has reached its configured max_rounds",
                    "max_rounds": campaign.get("max_rounds"),
                },
            )
        provider_id = str((payload or {}).get("provider_config_id") or "").strip()
        if not provider_id and existing_plans:
            provider_id = str(existing_plans[-1].get("provider_config_id") or "").strip()
        if not provider_id:
            raise HTTPException(status_code=400, detail="provider_config_id is required")
        provider = _require_ai_provider(provider_id, user, include_secret=True)
        if not provider["enabled"]:
            raise HTTPException(status_code=400, detail="AI provider config is disabled")
        dataset = _require_dataset(campaign["dataset_id"], user)
        include = (payload or {}).get("include")
        include = include if isinstance(include, dict) else {}
        baseline_experiment_ids = (payload or {}).get("baseline_experiment_ids")
        baseline_experiment_ids = (
            baseline_experiment_ids if isinstance(baseline_experiment_ids, list) else []
        )
        language = str((payload or {}).get("language") or "zh").strip() or "zh"
        strategy_preference = _agent_strategy_preference(payload)
        objective = str((payload or {}).get("objective") or campaign.get("objective") or "").strip()
        try:
            max_tokens = max(512, min(int((payload or {}).get("max_tokens") or 2200), 6000))
        except (TypeError, ValueError):
            max_tokens = 2200
        previous_full_plans = [
            store.get_agent_execution_plan(plan["id"], user=user)
            for plan in existing_plans
            if isinstance(plan, dict) and plan.get("id")
        ]
        previous_full_plans = [
            _agent_refresh_plan(plan, user) for plan in previous_full_plans if plan
        ]
        previous_full_plans = sorted(
            previous_full_plans,
            key=lambda plan: (int(plan.get("round_index") or 0), plan.get("created_at") or ""),
        )
        if previous_full_plans:
            last_plan = previous_full_plans[-1]
            execution = (
                last_plan.get("execution") if isinstance(last_plan.get("execution"), dict) else {}
            )
            progress = (
                execution.get("progress") if isinstance(execution.get("progress"), dict) else {}
            )
            unfinished = (
                progress.get("unfinished") if isinstance(progress.get("unfinished"), list) else []
            )
            if unfinished:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "agent_campaign_previous_round_unfinished",
                        "message": "previous campaign round still has queued or running experiments",
                        "plan_id": last_plan.get("id"),
                        "unfinished": unfinished,
                    },
                )
            if not _agent_plan_has_review(last_plan):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "agent_campaign_previous_round_needs_review",
                        "message": "previous campaign round must be reviewed before generating the next round",
                        "plan_id": last_plan.get("id"),
                        "status": last_plan.get("status"),
                    },
                )
        if not str((payload or {}).get("objective") or "").strip():
            objective = _agent_campaign_round_objective(campaign, previous_full_plans)
        context, manifest = build_agent_campaign_round_context(
            store,
            campaign,
            dataset,
            campaign["task_type"],
            user=user,
            include=include,
            baseline_experiment_ids=baseline_experiment_ids,
            previous_plans=previous_full_plans[-3:],
        )
        return {
            "campaign": campaign,
            "next_round_index": next_round_index,
            "generation_args": {
                "request": request,
                "user": user,
                "provider": provider,
                "provider_id": provider_id,
                "dataset": dataset,
                "task_type": campaign["task_type"],
                "objective": objective,
                "language": language,
                "max_tokens": max_tokens,
                "context": context,
                "manifest": manifest,
                "strategy_preference": strategy_preference,
                "campaign_id": campaign["id"],
                "round_index": next_round_index,
                "audit_extra": {"campaign_id": campaign["id"], "round_index": next_round_index},
            },
        }

    def _agent_campaign_round_events(prepared):
        events = _agent_generate_execution_plan_events(**prepared["generation_args"])
        try:
            for event in events:
                if event.get("type") == "done":
                    store.update_agent_campaign(
                        prepared["campaign"]["id"],
                        current_round_index=prepared["next_round_index"],
                        status="active",
                    )
                yield event
        finally:
            close = getattr(events, "close", None)
            if callable(close):
                close()

    @app.post("/api/agent/campaigns/{campaign_id}/rounds")
    def create_agent_campaign_round(campaign_id: str, payload: dict, request: Request = None):
        plan = None
        for event in _agent_campaign_round_events(
            _agent_campaign_round_generation(campaign_id, payload, request)
        ):
            if event.get("type") == "done":
                plan = (event.get("data") or {}).get("plan")
        if not plan:
            raise HTTPException(
                status_code=502, detail="campaign round generation ended without a plan"
            )
        return plan

    @app.post("/api/agent/campaigns/{campaign_id}/rounds/stream")
    def stream_agent_campaign_round(campaign_id: str, payload: dict, request: Request = None):
        return _agent_generation_stream(
            _agent_campaign_round_events(
                _agent_campaign_round_generation(campaign_id, payload, request)
            )
        )

    def _agent_execution_plan_generation_args(payload, request):
        user = _current_user(request)
        provider_id = str((payload or {}).get("provider_config_id") or "").strip()
        if not provider_id:
            raise HTTPException(status_code=400, detail="provider_config_id is required")
        provider = _require_ai_provider(provider_id, user, include_secret=True)
        if not provider["enabled"]:
            raise HTTPException(status_code=400, detail="AI provider config is disabled")
        dataset = _require_dataset((payload or {}).get("dataset_id"), user)
        task_type = (
            str((payload or {}).get("task_type") or dataset["task_type"]).strip()
            or dataset["task_type"]
        )
        diagnostic_source = _agent_diagnostic_source(payload, user, dataset, task_type)
        objective = str((payload or {}).get("objective") or "").strip() or (
            "Use the attached diagnostic evidence to plan the next approved experiment actions."
            if diagnostic_source
            else "Plan the next approved experiment actions for this dataset."
        )
        include = (payload or {}).get("include")
        include = include if isinstance(include, dict) else {}
        baseline_experiment_ids = (payload or {}).get("baseline_experiment_ids")
        baseline_experiment_ids = (
            baseline_experiment_ids if isinstance(baseline_experiment_ids, list) else []
        )
        if diagnostic_source:
            baseline_experiment_ids = list(
                dict.fromkeys([diagnostic_source["experiment_id"], *baseline_experiment_ids])
            )
        language = str((payload or {}).get("language") or "zh").strip() or "zh"
        strategy_preference = _agent_strategy_preference(payload)
        try:
            max_tokens = max(512, min(int((payload or {}).get("max_tokens") or 2200), 6000))
        except (TypeError, ValueError):
            max_tokens = 2200

        context, manifest = build_experiment_design_context(
            store,
            dataset,
            task_type,
            user=user,
            include=include,
            baseline_experiment_ids=baseline_experiment_ids,
        )
        if diagnostic_source:
            context["diagnostic_source"] = {
                key: value
                for key, value in diagnostic_source.items()
                if key != "provider_config_id"
            }
            manifest["sections"].append(
                {
                    "id": "diagnostic_source",
                    "sent": True,
                    "detail": "one completed experiment diagnostic report",
                }
            )
        return {
            "request": request,
            "user": user,
            "provider": provider,
            "provider_id": provider_id,
            "dataset": dataset,
            "task_type": task_type,
            "objective": objective,
            "language": language,
            "max_tokens": max_tokens,
            "context": context,
            "manifest": manifest,
            "strategy_preference": strategy_preference,
        }

    @app.post("/api/agent/execution-plans")
    def agent_execution_plan(payload: dict, request: Request = None):
        return _agent_generate_and_store_execution_plan(
            **_agent_execution_plan_generation_args(payload, request)
        )

    @app.post("/api/agent/execution-plans/stream")
    def stream_agent_execution_plan(payload: dict, request: Request = None):
        return _agent_generation_stream(
            _agent_generate_execution_plan_events(
                **_agent_execution_plan_generation_args(payload, request)
            )
        )

    @app.get("/api/agent/execution-plans")
    def list_agent_execution_plans(
        request: Request = None,
        dataset_id: str = "",
        task_type: str = "",
        status: str = "",
        campaign_id: str = "",
        limit: int = 20,
        offset: int = 0,
    ):
        task_type = normalize_task_type(task_type)
        user = _current_user(request)
        try:
            plans, total = store.list_agent_execution_plans(
                user=user,
                dataset_id=dataset_id or None,
                task_type=task_type or None,
                status=status or None,
                campaign_id=campaign_id or None,
                limit=limit,
                offset=offset,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "plans": plans,
            "total": total,
            "limit": max(1, min(int(limit), 100)),
            "offset": max(0, int(offset)),
        }

    @app.delete("/api/agent/execution-plans/{plan_id}")
    def delete_agent_execution_plan(plan_id: str, request: Request = None):
        user = _current_user(request)
        plan = _require_agent_execution_plan(plan_id, user)
        execution = plan.get("execution") if isinstance(plan.get("execution"), dict) else {}
        progress = execution.get("progress") if isinstance(execution.get("progress"), dict) else {}
        unfinished = (
            progress.get("unfinished") if isinstance(progress.get("unfinished"), list) else []
        )
        if unfinished:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "agent_plan_has_unfinished_experiments",
                    "message": "cannot delete a plan with queued or running experiments",
                    "unfinished": unfinished,
                },
            )
        store.delete_agent_execution_plan(plan_id, user=user)
        _audit(
            "ai.agent_plan_deleted",
            request=request,
            actor=user,
            metadata={
                "plan_id": plan_id,
                "dataset_id": plan.get("dataset_id"),
                "task_type": plan.get("task_type"),
            },
        )
        return {"plan_id": plan_id, "deleted": True}

    @app.get("/api/agent/execution-plans/{plan_id}")
    def get_agent_execution_plan(plan_id: str, request: Request = None):
        user = _current_user(request)
        plan = _require_agent_execution_plan(plan_id, user)
        return _agent_refresh_plan(plan, user)

    @app.post("/api/agent/execution-plans/{plan_id}/refresh")
    def refresh_agent_execution_plan(plan_id: str, request: Request = None):
        user = _current_user(request)
        plan = _require_agent_execution_plan(plan_id, user)
        return _agent_refresh_plan(plan, user)

    @app.patch("/api/agent/execution-plans/{plan_id}/actions/{action_id}")
    def update_agent_execution_action(
        plan_id: str, action_id: str, payload: dict, request: Request = None
    ):
        user = _current_user(request)
        plan = _require_agent_execution_plan(plan_id, user)
        dataset = _require_dataset(plan["dataset_id"], user)
        actions = [
            dict(action) for action in (plan.get("actions") or []) if isinstance(action, dict)
        ]
        target_index = next(
            (
                index
                for index, action in enumerate(actions)
                if str(action.get("id") or "") == str(action_id)
            ),
            None,
        )
        if target_index is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "agent_plan_action_not_found", "message": "agent action not found"},
            )
        action = actions[target_index]
        if str(action.get("type") or "") != "create_experiment":
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "agent_action_not_editable",
                    "message": "only create_experiment actions can be edited",
                },
            )
        if str(action.get("status") or "pending") != "pending":
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "agent_action_already_executed",
                    "message": "executed actions cannot be edited",
                },
            )
        raw_config = (payload or {}).get("config")
        if not isinstance(raw_config, dict):
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "agent_action_config_invalid",
                    "message": "config must be an object",
                },
            )
        config, dropped_keys = _agent_candidate_config({"config": raw_config})
        preflight = preflight_experiment(dataset, plan["task_type"], config)
        if dropped_keys:
            warning = f"{action.get('lane') or action_id} action dropped unsupported config key(s): {', '.join(dropped_keys)}"
            preflight.setdefault("warning_codes", []).insert(0, "action_config_keys_dropped")
            preflight.setdefault("warnings", []).insert(0, warning)
        if "original_config" not in action:
            action["original_config"] = action.get("config") or {}
        if "original_preflight" not in action and action.get("preflight"):
            action["original_preflight"] = action.get("preflight")
        action["config"] = config
        action["preflight"] = preflight
        action["valid"] = bool(preflight.get("valid"))
        action["edited"] = True
        if (payload or {}).get("title"):
            action["title"] = str(
                (payload or {}).get("title") or action.get("title") or ""
            ).strip()[:180]
        actions[target_index] = action
        trace = dict(plan.get("trace") or {})
        trace["edited_action_count"] = sum(
            1 for item in actions if isinstance(item, dict) and item.get("edited")
        )
        trace["valid_action_count"] = sum(
            1 for item in actions if isinstance(item, dict) and item.get("valid")
        )
        updated = store.update_agent_execution_plan(plan_id, actions=actions, trace=trace)
        _audit(
            "ai.agent_action_updated",
            request=request,
            actor=user,
            metadata={
                "plan_id": plan_id,
                "action_id": action_id,
                "dataset_id": dataset["id"],
                "valid": action["valid"],
                "dropped_key_count": len(dropped_keys),
            },
        )
        return updated

    @app.post("/api/agent/execution-plans/{plan_id}/actions/{action_id}/confirm")
    def confirm_agent_execution_action(
        plan_id: str, action_id: str, payload: dict, request: Request = None
    ):
        user = _current_user(request)
        plan = _require_agent_execution_plan(plan_id, user)
        if str(plan.get("status") or "") in {"reported", "reviewed", "failed"}:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "agent_plan_closed",
                    "message": "this Agent plan can no longer accept experiments",
                },
            )

        actions = [
            dict(action) for action in (plan.get("actions") or []) if isinstance(action, dict)
        ]
        target_index = next(
            (
                index
                for index, action in enumerate(actions)
                if str(action.get("id") or "") == str(action_id)
            ),
            None,
        )
        if target_index is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "agent_plan_action_not_found", "message": "agent action not found"},
            )
        action = actions[target_index]
        if str(action.get("type") or "") != "create_experiment":
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "agent_action_not_confirmable",
                    "message": "only create_experiment actions can be confirmed",
                },
            )

        status = str(action.get("status") or "pending")
        if status == "executed":
            result = action.get("result") if isinstance(action.get("result"), dict) else {}
            experiment_id = str(result.get("experiment_id") or "").strip()
            experiment = store.get_experiment(experiment_id, user=user) if experiment_id else None
            if experiment:
                return experiment
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "agent_action_experiment_missing",
                    "message": "the confirmed Agent experiment is no longer available",
                    "experiment_id": experiment_id,
                },
            )
        if status != "pending":
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "agent_action_not_pending",
                    "message": "only pending Agent actions can be confirmed",
                    "status": status,
                },
            )

        requested_dataset_id = str(
            (payload or {}).get("dataset_id") or plan.get("dataset_id") or ""
        ).strip()
        requested_task_type = str(
            (payload or {}).get("task_type") or plan.get("task_type") or ""
        ).strip()
        if requested_dataset_id != str(plan.get("dataset_id") or "") or normalize_task_type(
            requested_task_type
        ) != str(plan.get("task_type") or ""):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "agent_action_context_mismatch",
                    "message": "the experiment task and dataset must match the Agent plan",
                    "expected_task_type": plan.get("task_type"),
                    "expected_dataset_id": plan.get("dataset_id"),
                },
            )
        raw_config = (payload or {}).get("config")
        if not isinstance(raw_config, dict):
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "agent_action_config_invalid",
                    "message": "config must be an object",
                },
            )

        dataset = _require_dataset(plan["dataset_id"], user)
        config, dropped_keys = _agent_candidate_config({"config": raw_config})
        preflight = preflight_experiment(dataset, plan["task_type"], config)
        if dropped_keys:
            warning = f"{action.get('lane') or action_id} action dropped unsupported config key(s): {', '.join(dropped_keys)}"
            preflight.setdefault("warning_codes", []).insert(0, "action_config_keys_dropped")
            preflight.setdefault("warnings", []).insert(0, warning)
        if not preflight.get("valid"):
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "agent_action_preflight_failed",
                    "message": "the final experiment configuration did not pass preflight",
                    "preflight": preflight,
                },
            )

        previous_preflight = (
            action.get("preflight") if isinstance(action.get("preflight"), dict) else {}
        )
        previous_normalized = (
            previous_preflight.get("normalized_config") or action.get("config") or {}
        )
        final_normalized = preflight.get("normalized_config") or config
        config_changed = json.dumps(previous_normalized, sort_keys=True, default=str) != json.dumps(
            final_normalized,
            sort_keys=True,
            default=str,
        )
        if config_changed and "original_config" not in action:
            action["original_config"] = action.get("config") or {}
        if config_changed and "original_preflight" not in action and previous_preflight:
            action["original_preflight"] = previous_preflight
        action["config"] = config
        action["preflight"] = preflight
        action["valid"] = True
        action["edited"] = bool(action.get("edited") or config_changed)

        deterministic_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"pepplatform:agent-plan:{plan_id}:action:{action_id}",
        ).hex
        experiment = store.get_experiment(deterministic_id, user=user)
        recovered_existing = experiment is not None
        if experiment:
            if experiment.get("dataset_id") != dataset["id"] or normalize_task_type(
                experiment.get("task_type")
            ) != normalize_task_type(plan["task_type"]):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "agent_action_experiment_conflict",
                        "message": "the deterministic Agent experiment id is already in use",
                    },
                )
            if experiment.get("status") == "queued":
                runner.submit(experiment["id"])
        else:
            _check_experiment_quota(user)
            generated_name_fields = _generated_experiment_name_fields(payload, user)
            if generated_name_fields.get("name_kind") == "config":
                lane = str(action.get("lane") or "").strip()
                generated_name_fields["name_qualifier"] = (
                    lane if lane in GENERATED_NAME_QUALIFIERS else None
                )
            experiment = _agent_create_experiment_for_action(
                plan,
                action,
                dataset,
                user,
                name=(payload or {}).get("name") or _agent_experiment_name(action, 0),
                experiment_id=deterministic_id,
                generated_name_fields=generated_name_fields,
            )

        action["status"] = "executed"
        action["result"] = {
            "experiment_id": experiment["id"],
            "status": experiment["status"],
            "confirmed_via": "experiment_wizard",
        }
        actions[target_index] = action
        if not any(
            isinstance(item, dict)
            and item.get("type") == "cancel_experiment"
            and str(item.get("source_action_id") or "") == str(action_id)
            for item in actions
        ):
            actions.append(_agent_cancel_action_for_experiment(plan_id, action, experiment))

        created_entry = {
            "action_id": action["id"],
            "experiment_id": experiment["id"],
            "experiment": experiment,
        }
        execution = dict(plan.get("execution") or {})
        existing_entries = (
            execution.get("created_experiments")
            if isinstance(execution.get("created_experiments"), list)
            else []
        )
        new_entries = (
            []
            if any(
                isinstance(item, dict)
                and str(item.get("action_id") or "") == str(action_id)
                and str(item.get("experiment_id") or "") == experiment["id"]
                for item in existing_entries
            )
            else [created_entry]
        )
        execution = _agent_merge_execution(plan, [action["id"]], new_entries, [])
        refreshed_execution, refreshed_status = _agent_refresh_execution(
            {**plan, "execution": execution, "status": "running"},
            user,
        )
        store.update_agent_execution_plan(
            plan_id,
            status=refreshed_status,
            actions=actions,
            execution=refreshed_execution,
            executed_at=store_utcnow(),
        )
        if settings.enabled and user and not recovered_existing:
            store.increment_quota_usage(user["id"], experiments_created=1)
        _audit(
            "ai.agent_action_confirmed",
            request=request,
            actor=user,
            metadata={
                "plan_id": plan_id,
                "action_id": action_id,
                "dataset_id": dataset["id"],
                "task_type": plan["task_type"],
                "experiment_id": experiment["id"],
                "recovered_existing": recovered_existing,
                "config_changed": config_changed,
                "status": "succeeded",
            },
        )
        return experiment

    @app.post("/api/agent/execution-plans/{plan_id}/execute", deprecated=True)
    def execute_agent_execution_plan(
        plan_id: str, payload: dict, request: Request = None, response: Response = None
    ):
        if response is not None:
            response.headers["Deprecation"] = "true"
            response.headers["Warning"] = (
                '299 - "Batch plan execution is deprecated; use the Experiment Wizard."'
            )
        user = _current_user(request)
        plan = _require_agent_execution_plan(plan_id, user)
        dataset = _require_dataset(plan["dataset_id"], user)
        approved_action_ids = (payload or {}).get("approved_action_ids")
        if not isinstance(approved_action_ids, list) or not approved_action_ids:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "agent_plan_approved_actions_required",
                    "message": "approved_action_ids must contain at least one action id",
                },
            )
        approved_action_ids = [
            str(action_id).strip() for action_id in approved_action_ids if str(action_id).strip()
        ]
        actions = [
            dict(action) for action in (plan.get("actions") or []) if isinstance(action, dict)
        ]
        actions_by_id = {str(action.get("id") or ""): action for action in actions}
        missing_ids = [
            action_id for action_id in approved_action_ids if action_id not in actions_by_id
        ]
        if missing_ids:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "agent_plan_action_not_found",
                    "message": "one or more approved actions were not found",
                    "action_ids": missing_ids,
                },
            )

        selected_actions = []
        rejected = []
        create_count = 0
        created_by_plan_ids = set(_agent_plan_created_experiment_ids(plan))
        for action_id in approved_action_ids:
            action = actions_by_id[action_id]
            action_type = str(action.get("type") or "").strip()
            status = str(action.get("status") or "pending").strip()
            if status != "pending":
                rejected.append(
                    {"id": action_id, "reason": "action is not pending", "status": status}
                )
                continue
            if action_type == "create_experiment":
                if not action.get("valid"):
                    rejected.append({"id": action_id, "reason": "action is not valid"})
                    continue
                preflight = preflight_experiment(
                    dataset, plan["task_type"], action.get("config") or {}
                )
                if not preflight.get("valid"):
                    rejected.append(
                        {"id": action_id, "reason": "preflight failed", "preflight": preflight}
                    )
                    continue
                action["preflight"] = preflight
                create_count += 1
            elif action_type == "cancel_experiment":
                target_experiment_id = str(action.get("target_experiment_id") or "").strip()
                if not target_experiment_id or target_experiment_id not in created_by_plan_ids:
                    rejected.append(
                        {"id": action_id, "reason": "cancel target was not created by this plan"}
                    )
                    continue
                experiment = _require_experiment(target_experiment_id, user)
                if experiment["status"] not in {"queued", "running"}:
                    rejected.append(
                        {
                            "id": action_id,
                            "reason": "cancel target is not queued or running",
                            "status": experiment["status"],
                        }
                    )
                    continue
                action["target_experiment"] = experiment
            else:
                rejected.append(
                    {"id": action_id, "reason": f"unsupported action type: {action_type}"}
                )
                continue
            selected_actions.append(action)

        if rejected:
            _audit(
                "ai.agent_action_rejected",
                request=request,
                actor=user,
                metadata={
                    "plan_id": plan_id,
                    "dataset_id": dataset["id"],
                    "status": "failed",
                    "rejected": rejected,
                },
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "agent_plan_action_validation_failed",
                    "message": "one or more approved actions cannot be executed",
                    "actions": rejected,
                },
            )
        _check_experiment_quota(user, create_count) if create_count else None

        created_experiments = []
        canceled_experiments = []
        appended_cancel_actions = []
        executed_action_ids = []
        for action in selected_actions:
            if action["type"] == "create_experiment":
                experiment = _agent_create_experiment_for_action(
                    plan,
                    action,
                    dataset,
                    user,
                    name=_agent_experiment_name(action, len(created_experiments)),
                    generated_name_fields={
                        "name_kind": "config",
                        "name_qualifier": (
                            action.get("lane")
                            if action.get("lane") in GENERATED_NAME_QUALIFIERS
                            else None
                        ),
                    },
                )
                action["status"] = "executed"
                action["result"] = {
                    "experiment_id": experiment["id"],
                    "status": experiment["status"],
                }
                created_entry = {
                    "action_id": action["id"],
                    "experiment_id": experiment["id"],
                    "experiment": experiment,
                }
                created_experiments.append(created_entry)
                appended_cancel_actions.append(
                    _agent_cancel_action_for_experiment(plan_id, action, experiment)
                )
            elif action["type"] == "cancel_experiment":
                updated = runner.cancel(action["target_experiment_id"])
                action["status"] = "executed"
                action["result"] = {"experiment_id": updated["id"], "status": updated["status"]}
                canceled_experiments.append(
                    {
                        "action_id": action["id"],
                        "experiment_id": updated["id"],
                        "experiment": updated,
                    }
                )
            executed_action_ids.append(action["id"])
            actions_by_id[action["id"]] = action

        updated_actions = [
            actions_by_id.get(str(action.get("id") or ""), action) for action in actions
        ]
        existing_action_ids = {str(action.get("id") or "") for action in updated_actions}
        for cancel_action in appended_cancel_actions:
            if cancel_action["id"] not in existing_action_ids:
                updated_actions.append(cancel_action)
                existing_action_ids.add(cancel_action["id"])
        execution = _agent_merge_execution(
            plan,
            executed_action_ids,
            created_experiments,
            canceled_experiments,
        )
        refreshed_execution, refreshed_status = _agent_refresh_execution(
            {
                **plan,
                "execution": execution,
                "status": "running" if create_count else plan.get("status"),
            },
            user,
        )
        updated_plan = store.update_agent_execution_plan(
            plan_id,
            status=refreshed_status,
            actions=updated_actions,
            execution=refreshed_execution,
            executed_at=store_utcnow(),
        )
        if settings.enabled and user and create_count:
            store.increment_quota_usage(user["id"], experiments_created=create_count)
        _audit(
            "ai.agent_plan_executed",
            request=request,
            actor=user,
            metadata={
                "plan_id": plan_id,
                "dataset_id": dataset["id"],
                "task_type": plan["task_type"],
                "status": "succeeded",
                "approved_action_ids": executed_action_ids,
                "created_experiment_ids": [item["experiment_id"] for item in created_experiments],
                "canceled_experiment_ids": [item["experiment_id"] for item in canceled_experiments],
            },
        )
        return updated_plan

    def _agent_plan_review_generation(plan_id, payload, request):
        user = _current_user(request)
        plan = _agent_refresh_plan(_require_agent_execution_plan(plan_id, user), user)
        dataset = _require_dataset(plan["dataset_id"], user)
        execution = plan.get("execution") if isinstance(plan.get("execution"), dict) else {}
        progress = execution.get("progress") if isinstance(execution.get("progress"), dict) else {}
        unfinished = (
            progress.get("unfinished") if isinstance(progress.get("unfinished"), list) else []
        )
        if unfinished:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "agent_plan_experiments_unfinished",
                    "message": "agent plan still has queued or running experiments",
                    "unfinished": unfinished,
                },
            )
        if not execution.get("report_ready"):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "agent_plan_report_not_ready",
                    "message": "agent plan has no completed experiment set to report",
                    "progress": progress,
                },
            )
        provider_id = str(
            (payload or {}).get("provider_config_id") or plan.get("provider_config_id") or ""
        ).strip()
        if not provider_id:
            raise HTTPException(status_code=400, detail="provider_config_id is required")
        provider = _require_ai_provider(provider_id, user, include_secret=True)
        if not provider["enabled"]:
            raise HTTPException(status_code=400, detail="AI provider config is disabled")
        include = (payload or {}).get("include")
        include = include if isinstance(include, dict) else {}
        language = str((payload or {}).get("language") or "zh").strip() or "zh"
        prompt = str((payload or {}).get("prompt") or "").strip() or (
            "Review the executed Agent plan and recommend the next approved experiment actions."
        )
        try:
            max_tokens = max(256, min(int((payload or {}).get("max_tokens") or 1600), 5000))
        except (TypeError, ValueError):
            max_tokens = 1600

        context, manifest = build_agent_plan_review_context(
            store, plan, dataset, user=user, include=include
        )
        messages, warnings = build_agent_plan_review_messages(context, manifest, prompt, language)
        return {
            "request": request,
            "user": user,
            "plan_id": plan_id,
            "plan": plan,
            "provider_id": provider_id,
            "provider": provider,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "manifest": manifest,
            "messages": messages,
            "warnings": warnings,
        }

    def _agent_plan_review_event_core(prepared):
        request = prepared["request"]
        user = prepared["user"]
        plan_id = prepared["plan_id"]
        plan = prepared["plan"]
        provider_id = prepared["provider_id"]
        provider = prepared["provider"]
        prompt = prepared["prompt"]
        manifest = prepared["manifest"]
        warnings = prepared["warnings"]
        try:
            api_key = decrypt_api_key(provider["api_key_ciphertext"])
            yield {
                "type": "stage",
                "data": {"stage": "generating", "attempt": 1, "max_attempts": 1},
            }
            response = {}
            for event in _ai_completion_events(
                provider,
                api_key,
                prepared["messages"],
                max_tokens=prepared["max_tokens"],
                temperature=0.2,
                response_format={"type": "json_object"},
            ):
                if event.get("type") == "completion":
                    response = event["data"]
                else:
                    yield event
            yield {
                "type": "stage",
                "data": {"stage": "validating", "attempt": 1, "max_attempts": 1},
            }
            review = parse_agent_plan_review(response["answer"])
        except AiConfigurationError as exc:
            _audit(
                "ai.agent_plan_reviewed",
                request=request,
                actor=user,
                metadata={
                    "plan_id": plan_id,
                    "provider_id": provider_id,
                    "status": "failed",
                    "reason": "configuration",
                },
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except AiProviderError as exc:
            _audit(
                "ai.agent_plan_reviewed",
                request=request,
                actor=user,
                metadata={
                    "plan_id": plan_id,
                    "provider_id": provider_id,
                    "status": "failed",
                    "reason": "provider_error",
                },
            )
            raise HTTPException(status_code=502, detail=_sanitize_ai_error(exc)) from exc
        except AgentResponseError as exc:
            _audit(
                "ai.agent_plan_reviewed",
                request=request,
                actor=user,
                metadata={
                    "plan_id": plan_id,
                    "provider_id": provider_id,
                    "status": "failed",
                    "reason": "invalid_response",
                },
            )
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        yield {"type": "stage", "data": {"stage": "saving", "attempt": 1, "max_attempts": 1}}
        updated_plan = store.update_agent_execution_plan(
            plan_id,
            status="reported",
            review=review,
            usage=response.get("usage") or {},
            context_manifest=manifest,
            warnings=list(dict.fromkeys([*(plan.get("warnings") or []), *warnings])),
            reviewed_at=store_utcnow(),
            provider=provider["provider"],
            model=response.get("model") or provider["model"],
        )
        _agent_maybe_complete_campaign_for_reviewed_plan(updated_plan, user)
        _audit(
            "ai.agent_plan_reviewed",
            request=request,
            actor=user,
            metadata={
                "plan_id": plan_id,
                "provider_id": provider_id,
                "status": "succeeded",
                "provider": provider["provider"],
                "model": response.get("model"),
                "prompt_summary": markdown_to_plain_summary(prompt),
            },
        )
        yield {"type": "done", "data": {"plan": updated_plan}}

    def _agent_plan_review_events(prepared):
        completed = False
        events = _agent_plan_review_event_core(prepared)
        try:
            for event in events:
                if event.get("type") == "done":
                    completed = True
                yield event
        except GeneratorExit:
            if not completed:
                _audit(
                    "ai.agent_plan_reviewed",
                    request=prepared["request"],
                    actor=prepared["user"],
                    metadata={
                        "plan_id": prepared["plan_id"],
                        "provider_id": prepared["provider_id"],
                        "status": "stopped",
                    },
                )
            raise
        finally:
            close = getattr(events, "close", None)
            if callable(close):
                close()

    @app.post("/api/agent/execution-plans/{plan_id}/review")
    def review_agent_execution_plan(plan_id: str, payload: dict, request: Request = None):
        plan = None
        for event in _agent_plan_review_events(
            _agent_plan_review_generation(plan_id, payload, request)
        ):
            if event.get("type") == "done":
                plan = (event.get("data") or {}).get("plan")
        if not plan:
            raise HTTPException(status_code=502, detail="agent plan review ended without a plan")
        return plan

    @app.post("/api/agent/execution-plans/{plan_id}/review/stream")
    def stream_agent_execution_plan_review(plan_id: str, payload: dict, request: Request = None):
        return _agent_generation_stream(
            _agent_plan_review_events(_agent_plan_review_generation(plan_id, payload, request))
        )

    @app.post("/api/auth/password-reset/request")
    def password_reset_request(payload: dict):
        email = normalize_email(payload.get("email"))
        user = store.get_user_by_email(email)
        if not user:
            return {"ok": True, "reset_sent": False}
        code, sent = _issue_email_code(user, CODE_PURPOSE_PASSWORD_RESET)
        response = {"ok": True, "reset_sent": sent, "email": user["email"]}
        if not sent:
            _attach_debug_code(response, code)
        return response

    @app.post("/api/auth/password-reset/confirm")
    def password_reset_confirm(payload: dict):
        email = normalize_email(payload.get("email"))
        code = str(payload.get("code") or "").strip()
        password = str(payload.get("password") or "")
        if len(password) < 8:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "password_too_short",
                    "message": "password must be at least 8 characters",
                },
            )
        user = store.get_user_by_email(email)
        if not user:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "verification_code_invalid",
                    "message": "verification code is invalid or expired",
                },
            )
        _verify_email_code(user, CODE_PURPOSE_PASSWORD_RESET, code)
        updated = store.update_user_password(user["id"], hash_password(password))
        return _auth_response(updated)

    @app.get("/api/admin/overview")
    def admin_overview(request: Request = None):
        _require_admin(request)
        return store.get_admin_overview()

    @app.get("/api/admin/users")
    def admin_list_users(
        request: Request = None,
        search: str = "",
        role: str = "",
        status: str = "",
        email_verified: bool | None = None,
        sort: str = "created_at",
        order: str = "desc",
        limit: int = 50,
        offset: int = 0,
    ):
        _require_admin(request)
        if role and role not in ADMIN_ROLES:
            raise HTTPException(status_code=400, detail="unsupported role")
        if status and status not in ADMIN_STATUSES:
            raise HTTPException(status_code=400, detail="unsupported status")
        if sort not in ADMIN_USER_SORTS:
            raise HTTPException(status_code=400, detail="unsupported user sort")
        if order not in ADMIN_SORT_ORDERS:
            raise HTTPException(status_code=400, detail="unsupported sort order")
        return store.list_admin_users(
            search=search,
            role=role,
            status=status,
            email_verified=email_verified,
            sort_by=sort,
            sort_order=order,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/admin/users/{user_id}")
    def admin_get_user(user_id: str, request: Request = None):
        _require_admin(request)
        user = store.get_admin_user_detail(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="user not found")
        return user

    @app.patch("/api/admin/users/{user_id}")
    def admin_update_user(user_id: str, payload: dict, request: Request = None):
        actor = _require_admin(request)
        target = store.get_user(user_id)
        if not target:
            raise HTTPException(status_code=404, detail="user not found")
        role = payload.get("role") if "role" in payload else None
        status = payload.get("status") if "status" in payload else None
        if role is not None:
            role = str(role).strip()
            if role not in ADMIN_ROLES:
                raise HTTPException(status_code=400, detail="unsupported role")
        if status is not None:
            status = str(status).strip()
            if status not in ADMIN_STATUSES:
                raise HTTPException(status_code=400, detail="unsupported status")
        if target["id"] == actor["id"] and (
            (role is not None and role != "admin") or (status is not None and status != "active")
        ):
            raise HTTPException(status_code=400, detail="cannot remove your own admin access")

        updated = store.update_user_admin_fields(user_id, role=role, status=status)
        if not updated:
            raise HTTPException(status_code=404, detail="user not found")
        revoked_sessions = 0
        if status == "disabled" and target["status"] != "disabled":
            revoked_sessions = store.revoke_sessions_for_user(user_id)
        changes = {}
        if role is not None and role != target["role"]:
            changes["role"] = {"from": target["role"], "to": role}
        if status is not None and status != target["status"]:
            changes["status"] = {"from": target["status"], "to": status}
        if changes:
            _audit(
                "admin.user_updated",
                request=request,
                actor=actor,
                target_user_id=user_id,
                metadata={"changes": changes, "revoked_sessions": revoked_sessions},
            )
        return store.get_admin_user_summary(user_id)

    @app.post("/api/admin/users/{user_id}/revoke-sessions")
    def admin_revoke_user_sessions(user_id: str, request: Request = None):
        actor = _require_admin(request)
        target = store.get_user(user_id)
        if not target:
            raise HTTPException(status_code=404, detail="user not found")
        if target["id"] == actor["id"]:
            raise HTTPException(
                status_code=400, detail="cannot revoke your own sessions from this panel"
            )
        revoked_sessions = store.revoke_sessions_for_user(user_id)
        _audit(
            "admin.sessions_revoked",
            request=request,
            actor=actor,
            target_user_id=user_id,
            metadata={"revoked_sessions": revoked_sessions},
        )
        return {"user": store.get_admin_user_detail(user_id), "revoked_sessions": revoked_sessions}

    @app.post("/api/admin/users/{user_id}/resend-verification")
    def admin_resend_verification(user_id: str, request: Request = None):
        actor = _require_admin(request)
        target = store.get_user(user_id)
        if not target:
            raise HTTPException(status_code=404, detail="user not found")
        if target["email_verified"]:
            _audit(
                "admin.verification_resend",
                request=request,
                actor=actor,
                target_user_id=user_id,
                metadata={"sent": False, "reason": "already_verified"},
            )
            return {"user": store.get_admin_user_summary(user_id), "verification_sent": False}
        code, sent = _issue_email_code(target, CODE_PURPOSE_EMAIL_VERIFY)
        _audit(
            "admin.verification_resend",
            request=request,
            actor=actor,
            target_user_id=user_id,
            metadata={"sent": sent},
        )
        response = {"user": store.get_admin_user_summary(user_id), "verification_sent": sent}
        if not sent:
            _attach_debug_code(response, code)
        return response

    @app.delete("/api/admin/users/{user_id}")
    def admin_delete_user(user_id: str, request: Request = None):
        actor = _require_admin(request)
        target = store.get_user(user_id)
        if not target:
            raise HTTPException(status_code=404, detail="user not found")
        if target["id"] == actor["id"]:
            raise HTTPException(status_code=400, detail="cannot delete your own account")
        deleted, decision = store.delete_user_without_resources(user_id)
        if not deleted:
            reason = decision.get("reason") or "user cannot be deleted"
            status_code = 409 if reason in {"admin_user", "user_has_resources"} else 400
            raise HTTPException(status_code=status_code, detail=reason)
        _audit(
            "admin.user_deleted",
            request=request,
            actor=actor,
            target_user_id=None,
            metadata={
                "target_user_id": target["id"],
                "target_email": target["email"],
                "deleted_counts": deleted["deleted_counts"],
            },
        )
        return {
            "user_id": target["id"],
            "email": target["email"],
            "deleted": True,
            "deleted_counts": deleted["deleted_counts"],
        }

    @app.patch("/api/admin/users/{user_id}/quotas")
    def admin_update_user_quotas(user_id: str, payload: dict, request: Request = None):
        actor = _require_admin(request)
        with store.atomic():
            before = store.get_user_quota(user_id)
            try:
                quota = store.resources.set_quota(user_id, payload or {}, actor_user_id=actor["id"])
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if not quota:
                raise HTTPException(status_code=404, detail="user not found")
            changes = {
                key: {
                    "from": before.get("configured", {}).get(key, before.get(key)),
                    "to": quota.get("configured", {}).get(key, quota.get(key)),
                }
                for key in [*RESOURCE_QUOTAS, "allow_global_override", "overrides"]
                if before.get("configured", {}).get(key, before.get(key))
                != quota.get("configured", {}).get(key, quota.get(key))
            }
            effective_changes = {
                key: {"from": before[key], "to": quota[key]}
                for key in RESOURCE_QUOTAS
                if before[key] != quota[key]
            }
            if changes or effective_changes:
                _audit(
                    "admin.user_quota_updated",
                    request=request,
                    actor=actor,
                    target_user_id=user_id,
                    metadata={
                        "changes": changes,
                        "effective_changes": effective_changes,
                        "pending": quota["pending"],
                        "daily_reduction_mode": payload.get("daily_reduction_mode", "next_reset"),
                        "publish_announcements": payload.get("publish_announcements", False),
                    },
                )
            return quota

    @app.get("/api/admin/audit-logs")
    def admin_list_audit_logs(
        request: Request = None,
        event_type: str = "",
        actor_user_id: str = "",
        target_user_id: str = "",
        user_id: str = "",
        search: str = "",
        created_from: str = "",
        created_to: str = "",
        limit: int = 50,
        offset: int = 0,
    ):
        _require_admin(request)
        return store.list_audit_logs(
            event_type=event_type,
            actor_user_id=actor_user_id,
            target_user_id=target_user_id,
            user_id=user_id,
            search=search,
            created_from=created_from,
            created_to=created_to,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/admin/audit-event-types")
    def admin_list_audit_event_types(request: Request = None):
        _require_admin(request)
        return store.list_audit_event_types()

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "dependencies": {
                "profile_inference": profile_environment_readiness(),
            },
        }

    @app.get("/api/capabilities")
    def capabilities():
        return build_capabilities()

    @app.get("/api/datasets")
    def list_datasets(request: Request = None):
        user = _current_user(request)
        return store.list_datasets(user=user)

    @app.post("/api/datasets/upload")
    async def upload_dataset(
        task_type: str = Form(...),
        name: str = Form(...),
        archive: UploadFile = File(...),
        request: Request = None,
    ):
        task_type = normalize_task_type(task_type)
        try:
            user = _current_user(request)
            def upload_with_resources():
                with governor.reserve_write(user, upload_limits.max_extracted_bytes * 2 + upload_limits.max_archive_bytes, kind="upload"):
                    result = datasets.upload_archive(
                        task_type, name, archive.filename or "dataset.zip", archive.file,
                        owner_user_id=user["id"] if user else None,
                    )
                governor.scan()
                return result
            return await run_in_threadpool(
                upload_with_resources,
            )
        except ResourceLimitError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        except DatasetUploadLimitError as exc:
            raise HTTPException(status_code=413, detail=exc.detail) from exc
        except DatasetArchiveError as exc:
            raise HTTPException(status_code=400, detail=exc.detail) from exc
        except DatasetValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            await archive.close()

    @app.delete("/api/datasets/{dataset_id}")
    def delete_dataset(dataset_id: str, request: Request = None):
        user = _current_user(request)
        dataset = _require_dataset(dataset_id, user, write=True)
        if dataset["source"] != "upload":
            raise HTTPException(status_code=409, detail="only uploaded datasets can be deleted")
        if store.count_experiments_for_dataset(dataset_id, user=user) > 0:
            raise HTTPException(status_code=409, detail="dataset is still used by experiments")
        try:
            artifact_deleted = delete_dataset_artifact(dataset, paths)
        except OSError as exc:
            raise HTTPException(
                status_code=500, detail=f"failed to delete dataset artifact: {exc}"
            ) from exc
        store.delete_dataset(dataset_id, user=user)
        return {"dataset_id": dataset_id, "artifact_deleted": artifact_deleted}

    @app.get("/api/datasets/{dataset_id}/report")
    def dataset_report(dataset_id: str, limit: int = 5, request: Request = None):
        user = _current_user(request)
        dataset = _require_dataset(dataset_id, user)
        return build_dataset_report(dataset, preview_limit=limit)

    @app.get("/api/ensembles/capabilities")
    def ensemble_capabilities(request: Request = None):
        _current_user(request)
        return EnsembleService(store, paths).capabilities()

    @app.post("/api/ensembles/preflight")
    def ensemble_preflight(payload: dict, request: Request = None):
        user = _current_user(request)
        return EnsembleService(store, paths).preflight(payload, user=user)

    @app.post("/api/ensembles")
    def create_ensemble(payload: dict, request: Request = None):
        user = _current_user(request)
        if not creation_enabled():
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "ensemble_creation_disabled",
                    "message": "New ensemble creation is disabled",
                },
            )
        try:
            prepared = EnsembleService(store, paths).prepare(payload, user=user)
        except EnsembleError as exc:
            raise HTTPException(status_code=400, detail=exc.detail()) from exc
        _check_experiment_quota(user)
        name = str(payload.get("name") or "Model ensemble").strip()
        if not name or len(name) > 120:
            raise HTTPException(
                status_code=400, detail="Ensemble name must contain 1–120 characters"
            )
        experiment_id = uuid.uuid4().hex
        directory = paths.experiments / experiment_id
        directory.mkdir(parents=True)
        definition = prepared["definition"]
        experiment = store.create_experiment(
            name,
            prepared["dataset"]["id"],
            definition["task_type"],
            {
                "web_ensemble": definition,
                "model": "ensemble",
                "encoder": definition["method"],
                "tokenizer": "ensemble",
            },
            directory / "train.log",
            directory / "artifact",
            experiment_id=experiment_id,
            owner_user_id=user["id"] if user else None,
        )
        runner.submit(experiment_id)
        if settings.enabled and user:
            store.increment_quota_usage(user["id"], experiments_created=1)
        return _with_effective_experiment_config(experiment)

    @app.get("/api/experiments")
    def list_experiments(request: Request = None):
        user = _current_user(request)
        return [
            _with_effective_experiment_config(item) for item in store.list_experiments(user=user)
        ]

    @app.post("/api/experiments/preflight")
    def preflight(payload: dict, request: Request = None):
        user = _current_user(request)
        dataset = _require_dataset(payload.get("dataset_id"), user)
        task_type = payload.get("task_type") or dataset["task_type"]
        return preflight_experiment(dataset, task_type, payload.get("config") or {})

    @app.post("/api/experiments")
    def create_experiment(payload: dict, request: Request = None):
        user = _current_user(request)
        if (payload.get("config") or {}).get("web_ensemble"):
            raise HTTPException(
                status_code=400, detail="Use /api/ensembles to create a versioned ensemble"
            )
        dataset = _require_dataset(payload.get("dataset_id"), user)
        task_type = payload.get("task_type") or dataset["task_type"]
        report = preflight_experiment(dataset, task_type, payload.get("config") or {})
        if not report["valid"]:
            raise HTTPException(status_code=400, detail=report)
        config = report["normalized_config"]
        _check_experiment_quota(user)
        experiment_name = payload.get("name") or f"{task_type} experiment"
        generated_name_fields = _generated_experiment_name_fields(payload, user)
        if not payload.get("name") and not generated_name_fields:
            generated_name_fields = {"name_kind": "config"}
        experiment_id = validated_experiment_id(payload.get("id") or uuid.uuid4().hex)
        experiment_dir = paths.experiments / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        experiment = store.create_experiment(
            experiment_name,
            dataset["id"],
            task_type,
            config,
            experiment_dir / "train.log",
            experiment_dir / "artifact",
            experiment_id=experiment_id,
            owner_user_id=user["id"] if user else None,
            **generated_name_fields,
        )
        runner.submit(experiment["id"])
        if settings.enabled and user:
            store.increment_quota_usage(user["id"], experiments_created=1)
        return _with_effective_experiment_config(experiment)

    @app.post("/api/experiments/{experiment_id}/resume-downstream")
    def resume_downstream_svm(
        experiment_id: str,
        payload: dict,
        request: Request = None,
    ):
        user = _current_user(request)
        source = _require_experiment(experiment_id, user)
        if source.get("status") != "failed":
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "svm_resume_not_available",
                    "message": "only failed experiments can resume downstream training",
                },
            )
        source_config = dict(source.get("config") or {})
        source_classifier = (
            str(source_config.get("classifier") or source_config.get("model") or "").strip().lower()
        )
        source_calibration = (
            str(
                source_config.get("svm_calibration")
                or ("sigmoid" if source_classifier == "calibrated_svm" else "none")
            )
            .strip()
            .lower()
        )
        if source_classifier == "calibrated_svm":
            source_classifier = "svm"
        if source_classifier != "svm":
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "svm_resume_not_available",
                    "message": "downstream resume requires a failed SVM experiment",
                },
            )
        resume_bundle = inspect_svm_resume_bundle(source["artifact_dir"])
        if not resume_bundle:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "svm_resume_bundle_invalid",
                    "message": "the downstream SVM resume bundle is missing or invalid",
                },
            )
        overrides = (payload or {}).get("overrides") or {}
        if not isinstance(overrides, dict):
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "svm_resume_overrides_invalid",
                    "message": "resume overrides must be an object",
                },
            )
        unsupported = sorted(set(overrides) - SVM_DOWNSTREAM_OVERRIDE_FIELDS)
        if unsupported:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "svm_resume_override_unsupported",
                    "message": "unsupported downstream override(s): " + ", ".join(unsupported),
                },
            )
        config = source_config
        config.update(overrides)
        resumed_reduction = str(config.get("ml_reduction") or "none").strip().lower()
        resumed_kpca_kernel = str(config.get("kpca_kernel") or "rbf").strip().lower()
        if resumed_reduction != "kpca":
            config.pop("kpca_kernel", None)
            config.pop("kpca_gamma", None)
        elif resumed_kpca_kernel not in {"rbf", "poly"}:
            config.pop("kpca_gamma", None)
        resumed_classifier = (
            str(config.get("classifier") or config.get("model") or "").strip().lower()
        )
        if resumed_classifier == "calibrated_svm":
            resumed_classifier = "svm"
        resumed_calibration = (
            str(
                config.get("svm_calibration")
                or ("sigmoid" if source_calibration == "sigmoid" else "none")
            )
            .strip()
            .lower()
        )
        if resumed_classifier != "svm":
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "svm_resume_classifier_required",
                    "message": "downstream resume requires classifier=svm",
                },
            )
        if resumed_calibration != source_calibration:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "svm_resume_calibration_mismatch",
                    "message": "downstream resume must preserve the source SVM calibration mode",
                },
            )
        dataset = _require_dataset(source["dataset_id"], user)
        report = preflight_experiment(dataset, source["task_type"], config)
        if not report["valid"]:
            raise HTTPException(status_code=400, detail=report)
        _check_experiment_quota(user)
        resumed_id = uuid.uuid4().hex
        resumed_dir = paths.experiments / resumed_id
        resumed_dir.mkdir(parents=True, exist_ok=True)
        try:
            materialized = materialize_svm_resume_bundle(
                source["artifact_dir"],
                resumed_dir / "artifact",
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "svm_resume_bundle_invalid",
                    "message": str(exc),
                },
            ) from exc
        generated_name_fields = _generated_experiment_name_fields(
            {
                "generated_name": {
                    "kind": "rerun",
                    "source_experiment_id": source["id"],
                }
            },
            user,
        )
        resume = {
            "source_experiment_id": source["id"],
            "stage": "downstream_svm",
            "bundle_schema_version": materialized["schema_version"],
            "bundle_digest": materialized["tensor_sha256"],
        }
        experiment = store.create_experiment(
            source["name"],
            dataset["id"],
            source["task_type"],
            report["normalized_config"],
            resumed_dir / "train.log",
            resumed_dir / "artifact",
            experiment_id=resumed_id,
            owner_user_id=user["id"] if user else None,
            resume=resume,
            **generated_name_fields,
        )
        runner.submit(experiment["id"])
        if settings.enabled and user:
            store.increment_quota_usage(user["id"], experiments_created=1)
        return _with_effective_experiment_config(experiment)

    @app.patch("/api/experiments/{experiment_id}")
    def update_experiment_name(experiment_id: str, payload: dict, request: Request = None):
        user = _current_user(request)
        _require_experiment(experiment_id, user)
        raw_name = (payload or {}).get("name")
        name = raw_name.strip() if isinstance(raw_name, str) else ""
        if not name:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "experiment_name_required",
                    "message": "experiment name is required",
                },
            )
        if len(name) > 120:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "experiment_name_too_long",
                    "message": "experiment name must be 120 characters or fewer",
                },
            )
        if any(ord(character) < 32 or ord(character) == 127 for character in name):
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "experiment_name_invalid",
                    "message": "experiment name contains unsupported characters",
                },
            )
        updated = store.update_experiment_name(experiment_id, name, user=user)
        if not updated:
            raise HTTPException(status_code=404, detail="experiment not found")
        return updated

    @app.get("/api/experiments/{experiment_id}")
    def get_experiment(experiment_id: str, request: Request = None):
        user = _current_user(request)
        return _with_effective_experiment_config(_require_experiment(experiment_id, user))

    @app.delete("/api/experiments/{experiment_id}")
    def delete_experiment(experiment_id: str, request: Request = None):
        user = _current_user(request)
        experiment = _require_experiment(experiment_id, user)
        if experiment["status"] not in {"succeeded", "failed", "canceled"}:
            raise HTTPException(
                status_code=409,
                detail="only completed, failed, or stopped experiments can be deleted",
            )
        model_artifacts_deleted = 0
        try:
            for model in store.list_models_for_experiment(experiment_id, user=user):
                if delete_model_artifact(model["bundle_path"], paths):
                    model_artifacts_deleted += 1
            artifact_deleted = delete_experiment_artifact(experiment_id, paths)
        except OSError as exc:
            raise HTTPException(
                status_code=500, detail=f"failed to delete experiment artifact: {exc}"
            ) from exc
        store.delete_experiment(experiment_id, user=user)
        return {
            "experiment_id": experiment_id,
            "artifact_deleted": artifact_deleted,
            "model_artifacts_deleted": model_artifacts_deleted,
        }

    @app.post("/api/experiments/{experiment_id}/cancel")
    def cancel_experiment(experiment_id: str, request: Request = None):
        user = _current_user(request)
        _require_experiment(experiment_id, user)
        try:
            updated = runner.cancel(experiment_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not updated:
            raise HTTPException(status_code=404, detail="experiment not found")
        return updated

    @app.get("/api/experiments/{experiment_id}/logs")
    def get_logs(experiment_id: str, request: Request = None):
        user = _current_user(request)
        experiment = _require_experiment(experiment_id, user)
        path = Path(experiment["log_path"])
        return {"content": path.read_text(encoding="utf-8") if path.exists() else ""}

    @app.get("/api/experiments/{experiment_id}/analysis")
    def get_analysis(experiment_id: str, request: Request = None):
        user = _current_user(request)
        experiment = _require_experiment(experiment_id, user)
        try:
            return load_analysis(experiment["artifact_dir"])
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            payload = empty_analysis()
            payload["warnings"].append(f"analysis artifact could not be read: {exc}")
            return payload

    @app.post("/api/experiments/{experiment_id}/ai-analysis")
    def experiment_ai_analysis(experiment_id: str, payload: dict, request: Request = None):
        user = _current_user(request)
        experiment = _require_experiment(experiment_id, user)
        dataset = _require_dataset(experiment["dataset_id"], user)
        provider_id = str((payload or {}).get("provider_config_id") or "").strip()
        if not provider_id:
            raise HTTPException(status_code=400, detail="provider_config_id is required")
        provider = _require_ai_provider(provider_id, user, include_secret=True)
        if not provider["enabled"]:
            raise HTTPException(status_code=400, detail="AI provider config is disabled")
        prompt = str((payload or {}).get("prompt") or "").strip() or (
            "Summarize this experiment, explain the strongest evidence in the metrics, "
            "and recommend the next experiment to run."
        )
        include = (payload or {}).get("include")
        include = include if isinstance(include, dict) else {}
        language = str((payload or {}).get("language") or "zh").strip() or "zh"
        try:
            max_tokens = max(256, min(int((payload or {}).get("max_tokens") or 1200), 4000))
        except (TypeError, ValueError):
            max_tokens = 1200

        context, manifest = build_experiment_context(store, experiment, dataset, include=include)
        messages, warnings = build_analysis_messages(context, manifest, prompt, language)
        try:
            api_key = decrypt_api_key(provider["api_key_ciphertext"])
            response = app.state.ai_client.chat(
                provider,
                api_key,
                messages,
                max_tokens=max_tokens,
                temperature=0.2,
            )
        except AiConfigurationError as exc:
            _audit(
                "ai.experiment_analysis",
                request=request,
                actor=user,
                metadata={
                    "experiment_id": experiment_id,
                    "provider_id": provider_id,
                    "status": "failed",
                    "reason": "configuration",
                },
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except AiProviderError as exc:
            _audit(
                "ai.experiment_analysis",
                request=request,
                actor=user,
                metadata={
                    "experiment_id": experiment_id,
                    "provider_id": provider_id,
                    "status": "failed",
                    "reason": "provider_error",
                },
            )
            raise HTTPException(status_code=502, detail=_sanitize_ai_error(exc)) from exc

        _audit(
            "ai.experiment_analysis",
            request=request,
            actor=user,
            metadata={
                "experiment_id": experiment_id,
                "provider_id": provider_id,
                "status": "succeeded",
                "provider": provider["provider"],
                "model": response.get("model"),
                "prompt_summary": markdown_to_plain_summary(prompt),
                "context_sections": [
                    section["id"] for section in manifest.get("sections", []) if section.get("sent")
                ],
            },
        )
        return {
            "answer": response["answer"],
            "provider": provider["provider"],
            "model": response.get("model") or provider["model"],
            "usage": response.get("usage") or {},
            "context_manifest": manifest,
            "warnings": warnings,
        }

    @app.get("/api/experiments/{experiment_id}/analysis/conversations")
    def list_experiment_analysis_conversations(
        experiment_id: str,
        request: Request = None,
        q: str = "",
        limit: int = 50,
        offset: int = 0,
    ):
        user = _current_user(request)
        _require_experiment(experiment_id, user)
        return store.list_agent_chat_conversations(
            user=user,
            query=q,
            limit=limit,
            offset=offset,
            scope="experiment_analysis",
            experiment_id=experiment_id,
        )

    @app.get("/api/experiments/{experiment_id}/analysis/conversations/{conversation_id}")
    def get_experiment_analysis_conversation(
        experiment_id: str,
        conversation_id: str,
        request: Request = None,
        limit: int = 100,
        before_ordinal: int = None,
    ):
        user = _current_user(request)
        _require_experiment(experiment_id, user)
        result = store.list_agent_chat_messages(
            conversation_id,
            user=user,
            limit=limit,
            before_ordinal=before_ordinal,
            scope="experiment_analysis",
            experiment_id=experiment_id,
        )
        if not result:
            raise HTTPException(status_code=404, detail="analysis conversation not found")
        return result

    @app.get("/api/experiments/{experiment_id}/analysis/messages/{message_id}")
    def get_experiment_analysis_message(
        experiment_id: str,
        message_id: str,
        request: Request = None,
    ):
        user = _current_user(request)
        _require_experiment(experiment_id, user)
        message = store.get_agent_chat_message(
            message_id,
            user=user,
            scope="experiment_analysis",
            experiment_id=experiment_id,
        )
        if not message:
            raise HTTPException(status_code=404, detail="analysis message not found")
        return message

    @app.patch("/api/experiments/{experiment_id}/analysis/conversations/{conversation_id}")
    def update_experiment_analysis_conversation(
        experiment_id: str,
        conversation_id: str,
        payload: dict,
        request: Request = None,
    ):
        user = _current_user(request)
        _require_experiment(experiment_id, user)
        updates = {}
        if "title" in (payload or {}):
            updates["title"] = payload.get("title")
        if "pinned" in (payload or {}):
            updates["pinned"] = bool(payload.get("pinned"))
        if not updates:
            raise HTTPException(status_code=400, detail="title or pinned is required")
        try:
            conversation = store.update_agent_chat_conversation(
                conversation_id,
                updates,
                user=user,
                scope="experiment_analysis",
                experiment_id=experiment_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not conversation:
            raise HTTPException(status_code=404, detail="analysis conversation not found")
        return conversation

    @app.delete("/api/experiments/{experiment_id}/analysis/conversations/{conversation_id}")
    def delete_experiment_analysis_conversation(
        experiment_id: str,
        conversation_id: str,
        request: Request = None,
    ):
        user = _current_user(request)
        _require_experiment(experiment_id, user)
        conversation = store.delete_agent_chat_conversation(
            conversation_id,
            user=user,
            scope="experiment_analysis",
            experiment_id=experiment_id,
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="analysis conversation not found")
        return {"deleted": True, "conversation": conversation}

    @app.post("/api/experiments/{experiment_id}/analysis/stream")
    def stream_experiment_analysis(experiment_id: str, payload: dict, request: Request = None):
        user = _current_user(request)
        experiment = _require_experiment(experiment_id, user)
        if experiment.get("status") == "queued":
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "experiment_analysis_not_ready",
                    "message": "queued experiments do not have enough evidence for analysis",
                },
            )
        if experiment.get("status") not in {"running", "succeeded", "failed", "canceled"}:
            raise HTTPException(status_code=409, detail="experiment is not ready for analysis")
        dataset = _require_dataset(experiment["dataset_id"], user)
        provider_id = str((payload or {}).get("provider_config_id") or "").strip()
        if not provider_id:
            raise HTTPException(status_code=400, detail="provider_config_id is required")
        provider = _require_ai_provider(provider_id, user, include_secret=True)
        if not provider["enabled"]:
            raise HTTPException(status_code=400, detail="AI provider config is disabled")
        content = str((payload or {}).get("content") or "").strip()
        if not content:
            raise HTTPException(status_code=400, detail="content is required")
        if len(content) > 120000:
            raise HTTPException(status_code=400, detail="content is too long")
        conversation_id = str((payload or {}).get("conversation_id") or "").strip() or None
        if conversation_id and not store.get_agent_chat_conversation(
            conversation_id,
            user=user,
            scope="experiment_analysis",
            experiment_id=experiment_id,
        ):
            raise HTTPException(status_code=404, detail="analysis conversation not found")
        language = str((payload or {}).get("language") or "zh").strip() or "zh"
        include = (payload or {}).get("include")
        include = include if isinstance(include, dict) else {}
        try:
            max_tokens = max(512, min(int((payload or {}).get("max_tokens") or 4000), 8000))
        except (TypeError, ValueError):
            max_tokens = 4000
        try:
            api_key = decrypt_api_key(provider["api_key_ciphertext"])
            turn = store.begin_agent_chat_turn(
                user.get("id") if user else None,
                provider_id,
                provider["provider"],
                provider["model"],
                content,
                conversation_id=conversation_id,
                user=user,
                scope="experiment_analysis",
                experiment_id=experiment_id,
                assistant_kind="report" if conversation_id is None else "chat",
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=404 if conversation_id else 400, detail=str(exc)
            ) from exc
        except AiConfigurationError as exc:
            raise HTTPException(status_code=500, detail=_sanitize_ai_error(exc)) from exc

        assistant_message = turn["assistant_message"]
        is_report = assistant_message.get("kind") == "report"
        experiment_context, manifest = build_experiment_context(
            store, experiment, dataset, include=include
        )
        experiment_context["allowed_config_keys"] = sorted(COMMON_FIELDS)
        experiment_context["capabilities"] = build_capabilities()
        history = []
        if not is_report:
            history = (
                store.get_agent_chat_context(
                    turn["conversation"]["id"],
                    assistant_message["ordinal"],
                    user=user,
                    message_limit=40,
                    char_limit=30000,
                    scope="experiment_analysis",
                    experiment_id=experiment_id,
                )
                or []
            )
            messages, context_json = build_experiment_followup_messages(
                experiment_context,
                manifest,
                history,
                language,
            )
        else:
            messages, context_json = build_experiment_report_messages(
                experiment_context,
                manifest,
                content,
                language,
            )
        manifest = enrich_context_manifest(
            manifest,
            experiment,
            context_json,
            store_utcnow(),
        )

        def event_stream():
            answer = ""
            finalized = False
            usage = {}
            response_provider = provider["provider"]
            response_model = provider["model"]
            yield _sse_frame("meta", turn)
            yield _sse_frame(
                "stage",
                {"message_id": assistant_message["id"], "stage": "context_ready"},
            )
            try:
                if is_report:
                    yield _sse_frame(
                        "stage",
                        {"message_id": assistant_message["id"], "stage": "analyzing_evidence"},
                    )
                    response = app.state.ai_client.chat(
                        provider,
                        api_key,
                        messages,
                        max_tokens=max_tokens,
                        temperature=0.2,
                        response_format={"type": "json_object"},
                    )
                    answer = response.get("answer") or ""
                    usage = response.get("usage") or {}
                    response_provider = response.get("provider") or response_provider
                    response_model = response.get("model") or response_model
                    yield _sse_frame(
                        "stage",
                        {"message_id": assistant_message["id"], "stage": "validating_report"},
                    )
                    repair_attempted = False
                    try:
                        report, warnings = parse_experiment_diagnostic_report(
                            answer,
                            dataset,
                            experiment,
                            experiment.get("status"),
                        )
                    except ExperimentAnalysisResponseError as exc:
                        repair_attempted = True
                        yield _sse_frame(
                            "stage",
                            {"message_id": assistant_message["id"], "stage": "repairing_report"},
                        )
                        repair_response = app.state.ai_client.chat(
                            provider,
                            api_key,
                            build_experiment_report_repair_messages(messages, answer, exc),
                            max_tokens=max_tokens,
                            temperature=0,
                            response_format={"type": "json_object"},
                        )
                        answer = repair_response.get("answer") or ""
                        usage = repair_response.get("usage") or usage
                        response_provider = repair_response.get("provider") or response_provider
                        response_model = repair_response.get("model") or response_model
                        report, warnings = parse_experiment_diagnostic_report(
                            answer,
                            dataset,
                            experiment,
                            experiment.get("status"),
                        )
                    normalized_content = report_as_markdown(report)
                    message = store.update_agent_chat_message(
                        assistant_message["id"],
                        user=user,
                        content=normalized_content,
                        status="complete",
                        finish_reason="stop",
                        error="",
                        usage=usage,
                        provider=response_provider,
                        model=response_model,
                        kind="report",
                        payload={
                            "report": report,
                            "warnings": warnings,
                            "repair_attempted": repair_attempted,
                        },
                        context_manifest=manifest,
                        scope="experiment_analysis",
                        experiment_id=experiment_id,
                    )
                else:
                    yield _sse_frame(
                        "stage",
                        {"message_id": assistant_message["id"], "stage": "answering_follow_up"},
                    )
                    finish_reason = ""
                    last_checkpoint = time.monotonic()
                    for event in app.state.ai_client.stream_chat(
                        provider,
                        api_key,
                        messages,
                        max_tokens=max_tokens,
                        temperature=0.2,
                    ):
                        event_type = event.get("type")
                        if event_type == "heartbeat":
                            yield ": ping\n\n"
                            continue
                        if event_type == "delta":
                            delta = str(event.get("content") or "")
                            if not delta:
                                continue
                            answer += delta
                            yield _sse_frame(
                                "delta",
                                {"message_id": assistant_message["id"], "content": delta},
                            )
                            now = time.monotonic()
                            if now - last_checkpoint >= 1.0:
                                store.update_agent_chat_message(
                                    assistant_message["id"],
                                    user=user,
                                    content=answer,
                                    status="streaming",
                                    context_manifest=manifest,
                                    scope="experiment_analysis",
                                    experiment_id=experiment_id,
                                )
                                last_checkpoint = now
                            continue
                        if event_type == "done":
                            finish_reason = str(event.get("finish_reason") or "stop")
                            usage = (
                                event.get("usage") if isinstance(event.get("usage"), dict) else {}
                            )
                            response_provider = str(event.get("provider") or response_provider)
                            response_model = str(event.get("model") or response_model)
                    message = store.update_agent_chat_message(
                        assistant_message["id"],
                        user=user,
                        content=answer,
                        status="complete",
                        finish_reason=finish_reason or "stop",
                        error="",
                        usage=usage,
                        provider=response_provider,
                        model=response_model,
                        context_manifest=manifest,
                        scope="experiment_analysis",
                        experiment_id=experiment_id,
                    )
                finalized = True
                _audit(
                    "ai.experiment_analysis_conversation",
                    request=request,
                    actor=user,
                    metadata={
                        "experiment_id": experiment_id,
                        "conversation_id": turn["conversation"]["id"],
                        "message_id": assistant_message["id"],
                        "kind": assistant_message.get("kind"),
                        "provider_id": provider_id,
                        "status": "succeeded",
                    },
                )
                yield _sse_frame("done", {"message": message})
            except GeneratorExit:
                raise
            except Exception as exc:
                detail = _sanitize_ai_error(exc)
                message = store.update_agent_chat_message(
                    assistant_message["id"],
                    user=user,
                    content=answer if not is_report else "",
                    status="error",
                    finish_reason="error",
                    error=detail,
                    usage=usage,
                    context_manifest=manifest,
                    scope="experiment_analysis",
                    experiment_id=experiment_id,
                )
                finalized = True
                yield _sse_frame(
                    "error",
                    {"code": "experiment_analysis_failed", "detail": detail, "message": message},
                )
            finally:
                if not finalized:
                    store.update_agent_chat_message(
                        assistant_message["id"],
                        user=user,
                        content=answer if not is_report else "",
                        status="stopped",
                        finish_reason="stopped",
                        error="",
                        usage=usage,
                        context_manifest=manifest,
                        scope="experiment_analysis",
                        experiment_id=experiment_id,
                    )

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/leaderboard")
    def leaderboard(
        request: Request = None,
        task_type: str = "",
        dataset_id: str = "",
        metric: str = "f1",
        split: str = "auto",
        supports_inference: bool | None = None,
        status: str = "succeeded",
        limit: int = 200,
        global_view: bool = False,
    ):
        task_type = normalize_task_type(task_type)
        user = _current_user(request)
        global_view_enabled = bool(settings.enabled and _is_admin(user) and global_view)
        try:
            rows = store.list_leaderboard(
                task_type=task_type or None,
                dataset_id=dataset_id or None,
                metric=metric,
                split=split,
                supports_inference=supports_inference,
                status=status,
                limit=limit,
                user=None if global_view_enabled else user,
                owner_only=bool(settings.enabled and _is_admin(user) and not global_view_enabled),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "rows": rows,
            "total": len(rows),
            "metric": str(metric or "f1").strip().lower(),
            "split": str(split or "auto").strip().lower(),
            "filters": {
                "task_type": task_type or "",
                "dataset_id": dataset_id or "",
                "supports_inference": supports_inference,
                "status": status,
                "limit": limit,
                "global_view": global_view_enabled,
            },
        }

    @app.get("/api/models")
    def list_models(request: Request = None):
        user = _current_user(request)
        return store.list_models(user=user)

    @app.get("/api/models/{model_id}")
    def get_model(model_id: str, request: Request = None):
        user = _current_user(request)
        return _require_model(model_id, user)

    @app.get("/api/models/{model_id}/inference-readiness")
    def inference_readiness(model_id: str, request: Request = None):
        user = _current_user(request)
        _require_model(model_id, user)
        if app.state.models is None:
            app.state.models = ModelService(store)
        try:
            return app.state.models.readiness(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model not found") from exc

    @app.patch("/api/models/{model_id}")
    def update_model(model_id: str, payload: dict, request: Request = None):
        user = _current_user(request)
        raw_name = payload.get("name")
        name = raw_name.strip() if isinstance(raw_name, str) else ""
        if not name:
            raise HTTPException(status_code=400, detail="model name is required")
        if len(name) > 120:
            raise HTTPException(
                status_code=400, detail="model name must be 120 characters or fewer"
            )
        model = store.update_model_name(model_id, name, user=user)
        if not model:
            raise HTTPException(status_code=404, detail="model not found")
        return model

    @app.delete("/api/models/{model_id}")
    def delete_model(model_id: str, request: Request = None):
        user = _current_user(request)
        model = _require_model(model_id, user)
        try:
            artifact_deleted = delete_model_artifact(model["bundle_path"], paths)
        except OSError as exc:
            raise HTTPException(
                status_code=500, detail=f"failed to delete model artifact: {exc}"
            ) from exc
        store.delete_model(model_id, user=user)
        return {"model_id": model_id, "artifact_deleted": artifact_deleted}

    @app.get("/api/models/{model_id}/inference-runs")
    def list_inference_runs(model_id: str, request: Request = None):
        user = _current_user(request)
        _require_model(model_id, user, allow_removed=True)
        return store.list_inference_runs(model_id, user=user)

    @app.get("/api/inference-runs/{run_id}/csv")
    def download_inference_run(run_id: str, request: Request = None):
        user = _current_user(request)
        run = _require_inference_run(run_id, user)
        content = _prediction_csv(run["response"])
        headers = {"Content-Disposition": f'attachment; filename="inference-{run_id}.csv"'}
        return Response(content, media_type="text/csv; charset=utf-8", headers=headers)

    @app.delete("/api/inference-runs/{run_id}")
    def delete_inference_run(run_id: str, request: Request = None):
        user = _current_user(request)
        _require_inference_run(run_id, user)
        store.delete_inference_run(run_id, user=user)
        return {"run_id": run_id, "deleted": True}

    def _prediction_request(payload):
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="request body must be an object")
        sequences = payload.get("sequences")
        pairs = payload.get("pairs")
        if isinstance(sequences, str):
            sequences = [sequences]
        if sequences is not None and not isinstance(sequences, list):
            raise HTTPException(status_code=400, detail="sequences must be a list")
        if pairs is not None and not isinstance(pairs, list):
            raise HTTPException(status_code=400, detail="pairs must be a list")
        if sequences is None and pairs is None:
            raise HTTPException(
                status_code=400, detail="at least one sequence or peptide-protein pair is required"
            )
        input_count = len(pairs) if pairs is not None else len(sequences)
        request_payload = {}
        if sequences is not None:
            request_payload["sequences"] = sequences
        if pairs is not None:
            request_payload["pairs"] = pairs
        return sequences, pairs, input_count, request_payload

    def _save_prediction(model_id, request_payload, response, user, input_count, reservation):
        with store.atomic():
            run = store.create_inference_run(
                model_id,
                request_payload,
                response,
                owner_user_id=user["id"] if user else None,
            )
            governor.settle_inference(reservation, success=bool(run))
        if run:
            return {**response, "run_id": run["id"]}
        return response

    def _prediction_error(exc):
        from .inference import InferenceUnsupportedError

        if isinstance(exc, (ResourceLimitError, RemoteInferenceError)):
            return exc.status_code, exc.detail
        if isinstance(exc, EnsembleError):
            return 409, exc.detail()
        if isinstance(exc, KeyError):
            return 404, {"message": "model not found"}
        if isinstance(exc, ValueError):
            return 400, {"message": str(exc)}
        if isinstance(exc, InferenceUnsupportedError):
            return 409, exc.to_detail()
        return 500, {"code": "inference_failed", "stage": "inference", "message": str(exc)}

    @app.post("/api/models/{model_id}/predict")
    def predict(model_id: str, payload: dict, request: Request = None):
        user = _current_user(request)
        model = _require_model(model_id, user)
        sequences, pairs, input_count, request_payload = _prediction_request(payload)
        reservation = _check_inference_quota(user, input_count, model_id)
        try:
            if app.state.models is None:
                app.state.models = ManagedModelService(store, versions, governor)
            with governor.gpu() if model_uses_gpu(model) else nullcontext():
                governor.touch_model(model_id)
                extra = {"resource_limits": governor.inference_policy(reservation)} if isinstance(app.state.models, ManagedModelService) else {}
                response = app.state.models.predict(model_id, sequences=sequences, pairs=pairs, **extra)
                return _save_prediction(model_id, request_payload, response, user, input_count, reservation)
        except Exception as exc:
            status_code, detail = _prediction_error(exc)
            raise HTTPException(status_code=status_code, detail=detail) from exc
        finally:
            governor.settle_inference(reservation)

    @app.post("/api/models/{model_id}/predict/stream")
    def predict_stream(model_id: str, payload: dict, request: Request = None):
        user = _current_user(request)
        model = _require_model(model_id, user)
        sequences, pairs, input_count, request_payload = _prediction_request(payload)
        reservation = _check_inference_quota(user, input_count, model_id)
        from .inference import InferenceUnsupportedError

        gpu_lease = governor.gpu() if model_uses_gpu(model) else nullcontext()
        gpu_entered = False
        try:
            gpu_lease.__enter__()
            gpu_entered = True
            governor.touch_model(model_id)
        except Exception as exc:
            if gpu_entered:
                gpu_lease.__exit__(None, None, None)
            governor.settle_inference(reservation)
            status_code, detail = _prediction_error(exc)
            raise HTTPException(status_code=status_code, detail=detail) from exc

        events = queue.Queue()
        worker_finished = threading.Event()

        def push(event_name, data):
            events.put((event_name, dict(data or {})))

        def worker():
            try:
                if app.state.models is None:
                    app.state.models = ManagedModelService(store, versions, governor)
                response = app.state.models.predict(
                    model_id,
                    sequences=sequences,
                    pairs=pairs,
                    progress_callback=lambda event: push("stage", event),
                    **({"resource_limits": governor.inference_policy(reservation)} if isinstance(app.state.models, ManagedModelService) else {}),
                )
                push("stage", {"stage": "saving_result"})
                response = _save_prediction(model_id, request_payload, response, user, input_count, reservation)
                push("done", response)
            except Exception as exc:
                if isinstance(exc, (ResourceLimitError, RemoteInferenceError)):
                    detail = exc.detail
                elif isinstance(exc, EnsembleError):
                    detail = exc.detail()
                elif isinstance(exc, InferenceUnsupportedError):
                    detail = exc.to_detail()
                elif isinstance(exc, KeyError):
                    detail = {
                        "code": "model_not_found",
                        "stage": "validating_bundle",
                        "message": "model not found",
                    }
                elif isinstance(exc, ValueError):
                    detail = {
                        "code": "invalid_prediction",
                        "stage": "validating_bundle",
                        "message": str(exc),
                    }
                else:
                    detail = {"code": "inference_failed", "stage": "inference", "message": str(exc)}
                push("error", {**detail, "status_code": _prediction_error(exc)[0]})
            finally:
                governor.settle_inference(reservation)
                gpu_lease.__exit__(None, None, None)
                worker_finished.set()
                events.put(None)

        threading.Thread(
            target=worker, name=f"pepplatform-inference-{model_id}", daemon=True
        ).start()

        def event_stream():
            try:
                while True:
                    try:
                        item = events.get(timeout=15.0)
                    except queue.Empty:
                        yield ": ping\n\n"
                        continue
                    if item is None:
                        return
                    event_name, data = item
                    yield f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=True, separators=(',', ':'))}\n\n"
            finally:
                if not worker_finished.is_set() and isinstance(app.state.models, ManagedModelService):
                    app.state.models.cancel_prediction(model_id)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    register_resource_routes(app, governor, _current_user, _require_admin, _require_model, _audit)
    serialize_resource_mutations(app, store)
    return app


app = None
if (
    FastAPI is not None and "pytest" not in sys.modules
):  # pragma: no cover - depends on optional install state
    app = create_app()
