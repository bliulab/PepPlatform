from pepplatform.task_compat import normalize_task_type

import json
import re

from pepplatform.analysis import empty_analysis, load_analysis

from .ai import _compact_analysis_payload, _compact_dataset_report, _env_int, _redacted_experiment
from .capabilities import build_capabilities
from .preflight import preflight_experiment
from .runner import COMMON_FIELDS


LANES = ["conservative", "balanced", "exploratory"]
RISK_LEVELS = {"low", "medium", "high"}
STRATEGY_PREFERENCES = {"conservative", "balanced", "aggressive"}
STRATEGY_RISK_BANDS = {
    "conservative": {"conservative": "low", "balanced": "low", "exploratory": "medium"},
    "balanced": {"conservative": "low", "balanced": "medium", "exploratory": "high"},
    "aggressive": {"conservative": "medium", "balanced": "high", "exploratory": "high"},
}
ACTION_TYPES = {"create_experiment", "cancel_experiment"}


class AgentResponseError(ValueError):
    pass


def _safe_list(value):
    return value if isinstance(value, list) else []


def _compact_experiment_for_design(experiment):
    compact = _redacted_experiment(experiment)
    compact["error"] = str(compact.get("error") or "")[:500]
    return compact


def _compact_analysis_for_experiment(experiment):
    try:
        analysis = load_analysis(experiment.get("artifact_dir"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        analysis = empty_analysis()
    return _compact_analysis_payload(analysis)


def _compact_experiment_for_report(experiment):
    return {
        "experiment": _compact_experiment_for_design(experiment),
        "analysis": _compact_analysis_for_experiment(experiment),
    }


def _compact_leaderboard_row(row):
    row = row if isinstance(row, dict) else {}
    experiment = row.get("experiment") if isinstance(row.get("experiment"), dict) else {}
    dataset = row.get("dataset") if isinstance(row.get("dataset"), dict) else {}
    model = row.get("model") if isinstance(row.get("model"), dict) else {}
    return {
        "rank": row.get("rank"),
        "score": row.get("score"),
        "score_source": row.get("score_source"),
        "duration_seconds": row.get("duration_seconds"),
        "experiment": {
            "id": experiment.get("id"),
            "name": experiment.get("name"),
            "status": experiment.get("status"),
            "created_at": experiment.get("created_at"),
        },
        "dataset": {
            "id": dataset.get("id"),
            "name": dataset.get("name"),
        }
        if dataset
        else None,
        "model": {
            "id": model.get("id"),
            "name": model.get("name"),
            "supports_inference": model.get("supports_inference"),
        }
        if model
        else None,
        "metrics": row.get("metrics") or {},
        "config_summary": row.get("config_summary") or {},
    }


def _leaderboard_context(store, task_type, dataset_id, user):
    task_type = normalize_task_type(task_type)
    try:
        rows = store.list_leaderboard(
            task_type=task_type,
            dataset_id=dataset_id,
            metric="f1",
            split="auto",
            limit=5,
            user=user,
        )
    except (TypeError, ValueError):
        return []
    return [_compact_leaderboard_row(row) for row in rows[:5]]


def build_experiment_design_context(
    store,
    dataset,
    task_type,
    user=None,
    include=None,
    baseline_experiment_ids=None,
):
    task_type = normalize_task_type(task_type)
    include = include if isinstance(include, dict) else {}
    baseline_experiment_ids = _safe_list(baseline_experiment_ids)
    include_dataset_preview = bool(include.get("dataset_preview"))
    all_visible_experiments = store.list_experiments(user=user)
    recent_experiments = [
        _compact_experiment_for_design(experiment)
        for experiment in all_visible_experiments
        if normalize_task_type(experiment.get("task_type")) == normalize_task_type(task_type)
        and experiment.get("dataset_id") == dataset.get("id")
    ][:8]

    baseline_experiments = []
    seen_baselines = set()
    for raw_id in baseline_experiment_ids[:6]:
        experiment_id = str(raw_id or "").strip()
        if not experiment_id or experiment_id in seen_baselines:
            continue
        seen_baselines.add(experiment_id)
        experiment = store.get_experiment(experiment_id, user=user)
        if not experiment or normalize_task_type(
            experiment.get("task_type")
        ) != normalize_task_type(task_type):
            continue
        baseline_experiments.append(_compact_experiment_for_design(experiment))

    leaderboard = _leaderboard_context(store, task_type, dataset.get("id"), user)
    context = {
        "task_type": task_type,
        "allowed_config_keys": sorted(COMMON_FIELDS),
        "capabilities": build_capabilities(),
        "dataset": {
            "id": dataset.get("id"),
            "name": dataset.get("name"),
            "task_type": dataset.get("task_type"),
            "source": dataset.get("source"),
            "format": dataset.get("format"),
            "metadata": dataset.get("metadata") or {},
            "report": _compact_dataset_report(dataset, include_dataset_preview),
        },
        "recent_experiments": recent_experiments,
        "baseline_experiments": baseline_experiments,
        "leaderboard": leaderboard,
    }
    manifest = {
        "sections": [
            {
                "id": "capabilities",
                "sent": True,
                "detail": "supported tasks, routes, encoders, models, and classifiers",
            },
            {
                "id": "dataset",
                "sent": True,
                "detail": "dataset report with preview"
                if include_dataset_preview
                else "dataset report without sample preview",
            },
            {
                "id": "recent_experiments",
                "sent": bool(recent_experiments),
                "detail": f"{len(recent_experiments)} recent matching experiment summaries",
            },
            {
                "id": "baseline_experiments",
                "sent": bool(baseline_experiments),
                "detail": f"{len(baseline_experiments)} selected baseline experiment summaries",
            },
            {
                "id": "leaderboard",
                "sent": bool(leaderboard),
                "detail": f"{len(leaderboard)} top matching experiment summaries by auto F1",
            },
        ],
        "include": {"dataset_preview": include_dataset_preview},
        "redactions": [
            "API keys are never sent",
            "full logs are not sent",
            "raw prediction rows are not sent",
            "dataset sample preview is sent only when explicitly enabled",
        ],
    }
    return context, manifest


def experiment_design_context_summary(context):
    context = context if isinstance(context, dict) else {}
    campaign = context.get("campaign") if isinstance(context.get("campaign"), dict) else {}
    learning_summary = (
        campaign.get("learning_summary")
        if isinstance(campaign.get("learning_summary"), dict)
        else {}
    )
    summary = {
        "recent_experiment_count": len(_safe_list(context.get("recent_experiments"))),
        "baseline_experiment_count": len(_safe_list(context.get("baseline_experiments"))),
        "leaderboard_count": len(_safe_list(context.get("leaderboard"))),
    }
    previous_round_count = len(_safe_list(campaign.get("previous_rounds")))
    if previous_round_count:
        summary["previous_round_count"] = previous_round_count
    review_count = int(learning_summary.get("review_count") or 0)
    if review_count:
        summary["review_count"] = review_count
    recommendation_count = int(learning_summary.get("recommendation_count") or 0)
    if recommendation_count:
        summary["recommendation_count"] = recommendation_count
    avoided_config_count = len(_safe_list(campaign.get("do_not_repeat_configs")))
    if avoided_config_count:
        summary["avoided_config_count"] = avoided_config_count
    return summary


def _compact_created_experiments_for_campaign(plan):
    execution = plan.get("execution") if isinstance(plan.get("execution"), dict) else {}
    created = []
    for entry in _safe_list(execution.get("created_experiments"))[:6]:
        if not isinstance(entry, dict):
            continue
        experiment = entry.get("experiment") if isinstance(entry.get("experiment"), dict) else {}
        if not experiment:
            continue
        created.append(
            {
                "action_id": entry.get("action_id"),
                "experiment": _compact_experiment_for_design(experiment),
            }
        )
    return created


def _compact_plan_for_campaign_round(plan):
    plan = plan if isinstance(plan, dict) else {}
    execution = plan.get("execution") if isinstance(plan.get("execution"), dict) else {}
    progress = execution.get("progress") if isinstance(execution.get("progress"), dict) else {}
    return {
        "id": plan.get("id"),
        "round_index": plan.get("round_index") or 0,
        "status": plan.get("status"),
        "objective": plan.get("objective"),
        "actions": [
            {
                key: action.get(key)
                for key in [
                    "id",
                    "type",
                    "lane",
                    "title",
                    "status",
                    "valid",
                    "risk_level",
                    "objective_alignment",
                    "baseline_changes",
                    "verification_steps",
                    "result",
                ]
                if key in action
            }
            for action in _safe_list(plan.get("actions"))
            if isinstance(action, dict)
        ],
        "progress": {
            key: progress.get(key)
            for key in [
                "total",
                "queued",
                "running",
                "succeeded",
                "failed",
                "canceled",
                "missing",
                "completed",
            ]
            if key in progress
        },
        "created_experiments": _compact_created_experiments_for_campaign(plan),
        "review": plan.get("review") if isinstance(plan.get("review"), dict) else {},
        "warnings": _safe_list(plan.get("warnings")),
    }


def _config_fingerprint(config):
    return json.dumps(config or {}, ensure_ascii=False, sort_keys=True, default=str)


def _action_normalized_config(action):
    action = action if isinstance(action, dict) else {}
    preflight = action.get("preflight") if isinstance(action.get("preflight"), dict) else {}
    config = (
        preflight.get("normalized_config")
        if isinstance(preflight.get("normalized_config"), dict)
        else None
    )
    if config is None:
        config = action.get("config") if isinstance(action.get("config"), dict) else None
    return config or {}


def _do_not_repeat_configs(previous_plans):
    configs = []
    seen = set()
    for plan in _safe_list(previous_plans):
        if not isinstance(plan, dict):
            continue
        for action in _safe_list(plan.get("actions")):
            if not isinstance(action, dict) or action.get("type") != "create_experiment":
                continue
            config = _action_normalized_config(action)
            if not config:
                continue
            fingerprint = _config_fingerprint(config)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            configs.append(
                {
                    "plan_id": plan.get("id"),
                    "round_index": plan.get("round_index") or 0,
                    "action_id": action.get("id"),
                    "lane": action.get("lane"),
                    "config": config,
                }
            )
    return configs[:12]


def _review_recommendations(review):
    review = review if isinstance(review, dict) else {}
    recommendations = _string_list(review.get("recommendations"))
    if not recommendations:
        recommendations = _string_list(review.get("next_recommendations"))
    return recommendations


def _campaign_review_summary(previous_plans):
    reviews = []
    recommendations = []
    risks = []
    best_experiments = []
    failed_or_incomplete = []
    analysis_findings = []
    for plan in _safe_list(previous_plans):
        if not isinstance(plan, dict):
            continue
        review = plan.get("review") if isinstance(plan.get("review"), dict) else {}
        if not review or not str(review.get("summary") or "").strip():
            continue
        recs = _review_recommendations(review)
        review_entry = {
            "plan_id": plan.get("id"),
            "round_index": plan.get("round_index") or 0,
            "summary": str(review.get("summary") or "").strip(),
            "best_experiment": review.get("best_experiment")
            if isinstance(review.get("best_experiment"), dict)
            else {},
            "metric_evidence": _string_list(
                review.get("metric_evidence") or review.get("evidence")
            ),
            "recommendations": recs,
            "risks": _string_list(review.get("risks")),
            "failed_or_incomplete": [
                item
                for item in _safe_list(review.get("failed_or_incomplete"))
                if isinstance(item, dict)
            ],
            "analysis_findings": _string_list(review.get("analysis_findings")),
        }
        reviews.append(review_entry)
        recommendations.extend(recs)
        risks.extend(review_entry["risks"])
        failed_or_incomplete.extend(review_entry["failed_or_incomplete"])
        analysis_findings.extend(review_entry["analysis_findings"])
        if review_entry["best_experiment"]:
            best_experiments.append(
                {
                    "plan_id": plan.get("id"),
                    "round_index": plan.get("round_index") or 0,
                    **review_entry["best_experiment"],
                }
            )

    last_review = reviews[-1] if reviews else {}
    return last_review, {
        "review_count": len(reviews),
        "recommendation_count": len([item for item in recommendations if str(item).strip()]),
        "recommendations": recommendations[-8:],
        "risks": risks[-8:],
        "best_experiments": best_experiments[-5:],
        "failed_or_incomplete": failed_or_incomplete[-8:],
        "analysis_findings": analysis_findings[-8:],
    }


def build_agent_campaign_round_context(
    store,
    campaign,
    dataset,
    task_type,
    user=None,
    include=None,
    baseline_experiment_ids=None,
    previous_plans=None,
):
    task_type = normalize_task_type(task_type)
    context, manifest = build_experiment_design_context(
        store,
        dataset,
        task_type,
        user=user,
        include=include,
        baseline_experiment_ids=baseline_experiment_ids,
    )
    previous_rounds = [
        _compact_plan_for_campaign_round(plan)
        for plan in _safe_list(previous_plans)
        if isinstance(plan, dict)
    ]
    last_review, learning_summary = _campaign_review_summary(previous_plans)
    do_not_repeat_configs = _do_not_repeat_configs(previous_plans)
    context["campaign"] = {
        "id": campaign.get("id"),
        "objective": campaign.get("objective"),
        "status": campaign.get("status"),
        "max_rounds": campaign.get("max_rounds"),
        "current_round_index": campaign.get("current_round_index"),
        "previous_rounds": previous_rounds,
        "last_review": last_review,
        "learning_summary": learning_summary,
        "do_not_repeat_configs": do_not_repeat_configs,
    }
    manifest["sections"].append(
        {
            "id": "campaign",
            "sent": True,
            "detail": (
                f"{len(previous_rounds)} previous campaign round summaries, "
                f"{learning_summary['review_count']} reviews, "
                f"{len(do_not_repeat_configs)} configs to avoid repeating"
            ),
        }
    )
    return context, manifest


def build_experiment_design_messages(
    context, manifest, objective, language, strategy_preference="balanced"
):
    context_json = json.dumps(context, ensure_ascii=False, sort_keys=True)
    max_context_chars = max(8_000, _env_int("PEP_LLM_MAX_CONTEXT_CHARS", 60_000))
    warnings = []
    if len(context_json) > max_context_chars:
        warnings.append("context was truncated to fit the configured maximum size")
        context_json = context_json[:max_context_chars]
    manifest["context_chars"] = len(context_json)
    language_hint = "Chinese" if str(language or "").lower().startswith("zh") else "English"
    strategy_preference = (
        strategy_preference if strategy_preference in STRATEGY_PREFERENCES else "balanced"
    )
    risk_bands = STRATEGY_RISK_BANDS[strategy_preference]
    system_prompt = (
        "You are the PepPlatform experiment design agent. "
        "Use only the provided JSON context. Design exactly three candidate experiments: "
        "conservative, balanced, and exploratory. "
        "Return only strict JSON. Do not wrap it in Markdown. "
        "Every candidate config must use only allowed_config_keys and valid capability values. "
        "Every neural candidate must explicitly include batch_size, lr, and epochs. "
        "For classical candidates, include the primary classifier hyperparameters when they are intentionally chosen. "
        "The three configs should be meaningfully different after normalization. "
        f"The requested research strategy is {strategy_preference}. Use exactly these research-risk levels: "
        f"conservative={risk_bands['conservative']}, balanced={risk_bands['balanced']}, "
        f"exploratory={risk_bands['exploratory']}. "
        "Do not include API keys, shell commands, raw datasets, or training execution instructions."
    )
    schema_hint = {
        "candidates": [
            {
                "lane": "conservative",
                "title": "short title",
                "rationale": "why this experiment should be tried",
                "expected_tradeoffs": ["tradeoff 1", "tradeoff 2"],
                "objective_alignment": "how this matches the stated objective",
                "baseline_changes": ["what changes versus selected baselines or recent/top runs"],
                "risk_level": risk_bands["conservative"],
                "verification_steps": ["what to check after training"],
                "config": {
                    "tokenizer": "kmer",
                    "encoder": "index",
                    "model": "cnn",
                    "batch_size": 32,
                    "lr": 0.001,
                    "epochs": 30,
                },
            }
        ]
    }
    user_message = (
        f"Answer every title and explanatory field in {language_hint}, regardless of the objective language.\n"
        f"Research objective:\n{str(objective or '').strip()}\n\n"
        f"Strategy preference:\n{strategy_preference}\n\n"
        f"Required JSON shape:\n{json.dumps(schema_hint, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Context manifest:\n{json.dumps(manifest, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Experiment design context JSON:\n{context_json}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ], warnings


def build_experiment_design_repair_messages(
    context,
    manifest,
    objective,
    language,
    previous_answer,
    validation_issues,
    strategy_preference="balanced",
):
    messages, warnings = build_experiment_design_messages(
        context,
        manifest,
        objective,
        language,
        strategy_preference,
    )
    repair_payload = {
        "previous_answer": _strip_json_fence(previous_answer),
        "validation_issues": _safe_list(validation_issues),
    }
    repair_message = (
        "The previous answer failed validation. Rewrite the entire response as strict JSON only. "
        "Keep exactly three candidates with lanes conservative, balanced, and exploratory. "
        "Every candidate must pass preflight and the normalized configs must be distinct.\n\n"
        f"Validation report:\n{json.dumps(repair_payload, ensure_ascii=False, sort_keys=True)}"
    )
    return messages + [{"role": "user", "content": repair_message}], warnings


def _strip_json_fence(text):
    stripped = str(text or "").strip()
    fence_match = re.match(
        r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.IGNORECASE | re.DOTALL
    )
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def _load_agent_json(answer):
    text = _strip_json_fence(answer)
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except (TypeError, ValueError):
                pass
    raise AgentResponseError("AI provider did not return valid experiment-design JSON")


def _string_list(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value in [None, ""]:
        return []
    return [str(value).strip()]


def _risk_level(value):
    risk = str(value or "").strip().lower()
    return risk if risk in RISK_LEVELS else "medium"


def _clean_config(config):
    if not isinstance(config, dict):
        return {}, []
    cleaned = {key: value for key, value in config.items() if key in COMMON_FIELDS}
    dropped = sorted(str(key) for key in config if key not in COMMON_FIELDS)
    return cleaned, dropped


def _invalid_candidate_report(dataset, task_type, config, code, message):
    task_type = normalize_task_type(task_type)
    report = preflight_experiment(dataset, task_type, config)
    report["valid"] = False
    report.setdefault("error_codes", []).insert(0, code)
    report.setdefault("errors", []).insert(0, message)
    return report


def parse_experiment_design_candidates(answer, dataset, task_type, allow_partial=False):
    task_type = normalize_task_type(task_type)
    payload = _load_agent_json(answer)
    candidates_payload = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(candidates_payload, list):
        raise AgentResponseError("AI provider did not return a candidates list")
    warnings = []
    if len(candidates_payload) < 3:
        message = "AI provider returned fewer than three experiment candidates"
        if not allow_partial:
            raise AgentResponseError(message)
        warnings.append(message)
    if len(candidates_payload) > 3:
        warnings.append("AI provider returned more than three experiment candidates")

    if not candidates_payload:
        return [], warnings

    candidates = []
    for index, raw_candidate in enumerate(candidates_payload[:3]):
        candidate = raw_candidate if isinstance(raw_candidate, dict) else {}
        raw_lane = str(candidate.get("lane") or "").strip().lower()
        lane = raw_lane or LANES[index]
        if lane not in LANES:
            warnings.append(f"candidate {index + 1} used unsupported lane: {lane}")
            lane = LANES[index]
        raw_config = candidate.get("config")
        config, dropped_keys = _clean_config(raw_config)
        if dropped_keys:
            warnings.append(
                f"{lane} candidate dropped unsupported config key(s): {', '.join(dropped_keys)}"
            )

        if not isinstance(raw_config, dict):
            preflight = _invalid_candidate_report(
                dataset,
                task_type,
                {},
                "candidate_config_missing",
                "candidate config must be an object",
            )
        else:
            preflight = preflight_experiment(dataset, task_type, config)

        raw_risk_level = str(candidate.get("risk_level") or "").strip().lower()
        if raw_risk_level not in RISK_LEVELS:
            warnings.append(
                f"{lane} candidate used unsupported risk_level: {raw_risk_level or 'missing'}"
            )
        candidates.append(
            {
                "id": f"{lane}-{index + 1}",
                "lane": lane,
                "title": str(candidate.get("title") or lane.replace("_", " ").title()).strip(),
                "rationale": str(candidate.get("rationale") or "").strip(),
                "expected_tradeoffs": _string_list(candidate.get("expected_tradeoffs")),
                "objective_alignment": str(candidate.get("objective_alignment") or "").strip(),
                "baseline_changes": _string_list(candidate.get("baseline_changes")),
                "risk_level": _risk_level(raw_risk_level),
                "verification_steps": _string_list(candidate.get("verification_steps")),
                "config": config,
                "preflight": preflight,
                "valid": bool(preflight.get("valid")),
            }
        )
    return candidates, warnings


def validate_design_candidates_for_repair(candidates, warnings, strategy_preference="balanced"):
    issues = []
    if len(candidates or []) != 3:
        issues.append("exactly three candidates are required")
    if warnings:
        issues.extend(str(warning) for warning in warnings if str(warning).strip())

    strategy_preference = (
        strategy_preference if strategy_preference in STRATEGY_PREFERENCES else "balanced"
    )
    expected_risks = STRATEGY_RISK_BANDS[strategy_preference]
    lane_counts = {lane: 0 for lane in LANES}
    seen_configs = {}
    for candidate in candidates or []:
        lane = candidate.get("lane") or candidate.get("id") or "candidate"
        if lane in lane_counts:
            lane_counts[lane] += 1
            expected_risk = expected_risks[lane]
            if candidate.get("risk_level") != expected_risk:
                issues.append(
                    f"{lane} candidate risk_level must be {expected_risk} for {strategy_preference} strategy"
                )
        preflight = candidate.get("preflight") or {}
        normalized_config = preflight.get("normalized_config") or candidate.get("config") or {}
        if not candidate.get("valid"):
            details = []
            details.extend(preflight.get("error_codes") or [])
            details.extend(preflight.get("errors") or [])
            issue = "; ".join(str(item) for item in details if str(item).strip())
            issues.append(
                f"{lane} candidate failed preflight: {issue or 'unknown validation error'}"
            )

        fingerprint = json.dumps(normalized_config, ensure_ascii=False, sort_keys=True, default=str)
        if fingerprint in seen_configs:
            issues.append(
                f"{lane} candidate duplicates normalized config from {seen_configs[fingerprint]} candidate"
            )
        else:
            seen_configs[fingerprint] = lane

    if any(count != 1 for count in lane_counts.values()):
        issues.append(
            "candidate lanes must contain conservative, balanced, and exploratory exactly once"
        )

    return issues


def _language_hint(language):
    return "Chinese" if str(language or "").lower().startswith("zh") else "English"


def _context_json_and_warnings(context, manifest):
    context_json = json.dumps(context, ensure_ascii=False, sort_keys=True)
    max_context_chars = max(8_000, _env_int("PEP_LLM_MAX_CONTEXT_CHARS", 60_000))
    warnings = []
    if len(context_json) > max_context_chars:
        warnings.append("context was truncated to fit the configured maximum size")
        context_json = context_json[:max_context_chars]
    manifest["context_chars"] = len(context_json)
    return context_json, warnings


def _action_id(action_type, lane, index):
    prefix = "create" if action_type == "create_experiment" else "cancel"
    suffix = re.sub(r"[^a-z0-9_-]+", "-", str(lane or f"action-{index + 1}").strip().lower()).strip(
        "-"
    )
    return f"{prefix}-{suffix or index + 1}-{index + 1}"


def build_agent_execution_plan_messages(
    context, manifest, objective, language, strategy_preference="balanced"
):
    context_json, warnings = _context_json_and_warnings(context, manifest)
    language_hint = _language_hint(language)
    is_campaign_round = isinstance(context, dict) and isinstance(context.get("campaign"), dict)
    strategy_preference = (
        strategy_preference if strategy_preference in STRATEGY_PREFERENCES else "balanced"
    )
    risk_bands = STRATEGY_RISK_BANDS[strategy_preference]
    action_count_instruction = (
        "Draft one to three user-approvable tool actions. "
        if is_campaign_round
        else "Draft exactly three user-approvable tool actions with conservative, balanced, and exploratory lanes. "
    )
    system_prompt = (
        "You are the PepPlatform tool-execution planning agent. "
        "Use only the provided JSON context. Return only strict JSON. Do not wrap it in Markdown. "
        f"{action_count_instruction}"
        "For this draft, use only create_experiment actions. "
        "Every create_experiment config must use only allowed_config_keys and valid capability values. "
        "Every neural create_experiment config must explicitly include batch_size, lr, and epochs. "
        "For classical configs, include the primary classifier hyperparameters when they are intentionally chosen. "
        "Actions should be meaningfully different after normalization. "
        f"The requested research strategy is {strategy_preference}. For each returned lane, use exactly these "
        f"research-risk levels: conservative={risk_bands['conservative']}, balanced={risk_bands['balanced']}, "
        f"exploratory={risk_bands['exploratory']}. "
        "When campaign context is present, use last_review and learning_summary to address the latest "
        "recommendations, explain how each action uses prior evidence, and avoid configs listed in "
        "do_not_repeat_configs unless a deliberate minimal-control repeat is justified in the rationale. "
        "Do not include API keys, shell commands, raw datasets, or direct runner instructions."
    )
    schema_hint = {
        "actions": [
            {
                "type": "create_experiment",
                "lane": "conservative",
                "title": "short action title",
                "rationale": "why this tool action should be approved",
                "expected_tradeoffs": ["tradeoff 1", "tradeoff 2"],
                "objective_alignment": "how this action matches the stated objective",
                "baseline_changes": ["what changes versus selected baselines or recent/top runs"],
                "risk_level": risk_bands["conservative"],
                "verification_steps": ["what to check after training"],
                "config": {
                    "tokenizer": "kmer",
                    "encoder": "index",
                    "model": "cnn",
                    "batch_size": 32,
                    "lr": 0.001,
                    "epochs": 30,
                },
            }
        ]
    }
    user_message = (
        f"Answer every title and explanatory field in {language_hint}, regardless of the objective language.\n"
        f"Research objective:\n{str(objective or '').strip()}\n\n"
        f"Strategy preference:\n{strategy_preference}\n\n"
        f"Planning mode:\n{'campaign follow-up round' if is_campaign_round else 'standalone execution plan'}\n\n"
        f"Required JSON shape:\n{json.dumps(schema_hint, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Context manifest:\n{json.dumps(manifest, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Experiment design context JSON:\n{context_json}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ], warnings


def build_agent_execution_plan_repair_messages(
    context,
    manifest,
    objective,
    language,
    previous_answer,
    validation_issues,
    strategy_preference="balanced",
):
    messages, warnings = build_agent_execution_plan_messages(
        context,
        manifest,
        objective,
        language,
        strategy_preference,
    )
    repair_payload = {
        "previous_answer": _strip_json_fence(previous_answer),
        "validation_issues": _safe_list(validation_issues),
    }
    repair_message = (
        "The previous answer failed validation. Rewrite the entire response as strict JSON only. "
        "Keep one to three create_experiment actions. "
        "Every action must pass preflight and normalized create_experiment configs must be distinct.\n\n"
        f"Validation report:\n{json.dumps(repair_payload, ensure_ascii=False, sort_keys=True)}"
    )
    return messages + [{"role": "user", "content": repair_message}], warnings


def _candidate_to_create_action(candidate):
    candidate = candidate if isinstance(candidate, dict) else {}
    return {
        "type": "create_experiment",
        "lane": candidate.get("lane"),
        "title": candidate.get("title"),
        "rationale": candidate.get("rationale"),
        "expected_tradeoffs": candidate.get("expected_tradeoffs"),
        "objective_alignment": candidate.get("objective_alignment"),
        "baseline_changes": candidate.get("baseline_changes"),
        "risk_level": candidate.get("risk_level"),
        "verification_steps": candidate.get("verification_steps"),
        "config": candidate.get("config"),
    }


def _invalid_action_report(dataset, task_type, code, message):
    task_type = normalize_task_type(task_type)
    report = preflight_experiment(dataset, task_type, {})
    report["valid"] = False
    report.setdefault("error_codes", []).insert(0, code)
    report.setdefault("errors", []).insert(0, message)
    return report


def parse_agent_execution_actions(answer, dataset, task_type, allow_partial=False):
    task_type = normalize_task_type(task_type)
    payload = _load_agent_json(answer)
    if not isinstance(payload, dict):
        raise AgentResponseError("AI provider did not return an execution-plan object")
    actions_payload = payload.get("actions")
    if actions_payload is None and isinstance(payload.get("candidates"), list):
        actions_payload = [
            _candidate_to_create_action(candidate) for candidate in payload.get("candidates")
        ]
    if not isinstance(actions_payload, list):
        raise AgentResponseError("AI provider did not return an actions list")
    warnings = []
    if not actions_payload:
        message = "AI provider returned no execution actions"
        if not allow_partial:
            raise AgentResponseError(message)
        warnings.append(message)
    if len(actions_payload) > 3:
        warnings.append("AI provider returned more than three execution actions")

    actions = []
    seen_ids = set()
    for index, raw_action in enumerate(actions_payload[:3]):
        action = raw_action if isinstance(raw_action, dict) else {}
        action_type = str(action.get("type") or "create_experiment").strip().lower()
        fallback_lane = LANES[index] if index < len(LANES) else f"action-{index + 1}"
        raw_lane = str(action.get("lane") or "").strip().lower()
        lane = raw_lane or fallback_lane
        if lane not in LANES:
            warnings.append(f"action {index + 1} used unsupported lane: {lane}")
            lane = fallback_lane
        action_id = str(action.get("id") or _action_id(action_type, lane, index)).strip()
        if not action_id or action_id in seen_ids:
            action_id = _action_id(action_type, lane, index)
        seen_ids.add(action_id)
        title = str(action.get("title") or lane.replace("_", " ").title()).strip()
        raw_risk_level = str(action.get("risk_level") or "").strip().lower()
        if raw_risk_level not in RISK_LEVELS:
            warnings.append(
                f"{lane} action used unsupported risk_level: {raw_risk_level or 'missing'}"
            )
        base = {
            "id": action_id,
            "type": action_type,
            "lane": lane,
            "title": title,
            "rationale": str(action.get("rationale") or "").strip(),
            "expected_tradeoffs": _string_list(action.get("expected_tradeoffs")),
            "objective_alignment": str(action.get("objective_alignment") or "").strip(),
            "baseline_changes": _string_list(action.get("baseline_changes")),
            "risk_level": _risk_level(raw_risk_level),
            "verification_steps": _string_list(action.get("verification_steps")),
            "requires_approval": True,
            "status": "pending",
            "result": {},
            "valid": False,
        }
        if action_type not in ACTION_TYPES:
            base["preflight"] = _invalid_action_report(
                dataset,
                task_type,
                "agent_action_type_unsupported",
                f"unsupported agent action type: {action_type}",
            )
            actions.append(base)
            continue
        if action_type == "cancel_experiment":
            target_experiment_id = str(action.get("target_experiment_id") or "").strip()
            base["target_experiment_id"] = target_experiment_id
            base["valid"] = bool(target_experiment_id)
            if not target_experiment_id:
                base["preflight"] = _invalid_action_report(
                    dataset,
                    task_type,
                    "agent_cancel_target_missing",
                    "cancel_experiment action requires target_experiment_id",
                )
            actions.append(base)
            continue

        raw_config = action.get("config")
        config, dropped_keys = _clean_config(raw_config)
        if dropped_keys:
            warnings.append(
                f"{lane} action dropped unsupported config key(s): {', '.join(dropped_keys)}"
            )
        if not isinstance(raw_config, dict):
            preflight = _invalid_action_report(
                dataset,
                task_type,
                "action_config_missing",
                "create_experiment action config must be an object",
            )
        else:
            preflight = preflight_experiment(dataset, task_type, config)
        base.update(
            {
                "config": config,
                "preflight": preflight,
                "valid": bool(preflight.get("valid")),
            }
        )
        actions.append(base)
    return actions, warnings


def validate_execution_actions_for_repair(
    actions,
    warnings,
    strategy_preference="balanced",
    require_three=True,
):
    issues = []
    if not actions:
        issues.append("no actions were returned")
    if require_three and len(actions or []) != 3:
        issues.append("standalone execution plans require exactly three actions")
    if not require_three and len(actions or []) > 3:
        issues.append("campaign rounds support at most three actions")
    if warnings:
        issues.extend(str(warning) for warning in warnings if str(warning).strip())

    strategy_preference = (
        strategy_preference if strategy_preference in STRATEGY_PREFERENCES else "balanced"
    )
    expected_risks = STRATEGY_RISK_BANDS[strategy_preference]
    lane_counts = {lane: 0 for lane in LANES}
    seen_configs = {}
    for action in actions or []:
        label = action.get("lane") or action.get("id") or "action"
        if label in lane_counts:
            lane_counts[label] += 1
            expected_risk = expected_risks[label]
            if action.get("risk_level") != expected_risk:
                issues.append(
                    f"{label} action risk_level must be {expected_risk} for {strategy_preference} strategy"
                )
        action_type = action.get("type")
        if action_type != "create_experiment":
            issues.append(f"{label} action uses unsupported draft action type: {action_type}")
            continue
        preflight = action.get("preflight") or {}
        normalized_config = preflight.get("normalized_config") or action.get("config") or {}
        if not action.get("valid"):
            details = []
            details.extend(preflight.get("error_codes") or [])
            details.extend(preflight.get("errors") or [])
            issue = "; ".join(str(item) for item in details if str(item).strip())
            issues.append(f"{label} action failed preflight: {issue or 'unknown validation error'}")
        fingerprint = json.dumps(normalized_config, ensure_ascii=False, sort_keys=True, default=str)
        if fingerprint in seen_configs:
            issues.append(
                f"{label} action duplicates normalized config from {seen_configs[fingerprint]} action"
            )
        else:
            seen_configs[fingerprint] = label
    if require_three and any(count != 1 for count in lane_counts.values()):
        issues.append(
            "standalone action lanes must contain conservative, balanced, and exploratory exactly once"
        )
    elif not require_three and any(count > 1 for count in lane_counts.values()):
        issues.append("campaign action lanes must be unique")
    return issues


def build_agent_plan_review_context(store, plan, dataset, user=None, include=None):
    include = include if isinstance(include, dict) else {}
    include_dataset_preview = bool(include.get("dataset_preview"))
    execution = plan.get("execution") if isinstance(plan.get("execution"), dict) else {}
    created_entries = _safe_list(execution.get("created_experiments"))
    created_ids = []
    for entry in created_entries:
        if not isinstance(entry, dict):
            continue
        experiment = entry.get("experiment") if isinstance(entry.get("experiment"), dict) else {}
        experiment_id = str(entry.get("experiment_id") or experiment.get("id") or "").strip()
        if experiment_id and experiment_id not in created_ids:
            created_ids.append(experiment_id)
    experiments = []
    for experiment_id in created_ids[:12]:
        experiment = store.get_experiment(experiment_id, user=user)
        if experiment:
            experiments.append(_compact_experiment_for_report(experiment))
    context = {
        "plan": {
            "id": plan.get("id"),
            "status": plan.get("status"),
            "objective": plan.get("objective"),
            "task_type": plan.get("task_type"),
            "dataset_id": plan.get("dataset_id"),
            "actions": [
                {
                    key: action.get(key)
                    for key in [
                        "id",
                        "type",
                        "lane",
                        "title",
                        "status",
                        "risk_level",
                        "objective_alignment",
                        "baseline_changes",
                        "verification_steps",
                        "target_experiment_id",
                    ]
                    if key in action
                }
                for action in _safe_list(plan.get("actions"))
                if isinstance(action, dict)
            ],
            "execution": {
                "executed_action_ids": _safe_list(execution.get("executed_action_ids")),
                "created_count": len(created_entries),
                "canceled_count": len(_safe_list(execution.get("canceled_experiments"))),
                "warnings": _safe_list(execution.get("warnings")),
            },
        },
        "dataset": {
            "id": dataset.get("id"),
            "name": dataset.get("name"),
            "task_type": dataset.get("task_type"),
            "source": dataset.get("source"),
            "format": dataset.get("format"),
            "report": _compact_dataset_report(dataset, include_dataset_preview),
        },
        "created_experiments": experiments,
        "leaderboard": _leaderboard_context(
            store, plan.get("task_type"), plan.get("dataset_id"), user
        ),
    }
    manifest = {
        "sections": [
            {"id": "plan", "sent": True, "detail": "agent actions, execution state, and objective"},
            {
                "id": "dataset",
                "sent": True,
                "detail": "dataset report with preview"
                if include_dataset_preview
                else "dataset report without sample preview",
            },
            {
                "id": "created_experiments",
                "sent": bool(experiments),
                "detail": f"{len(experiments)} current experiment and analysis summaries created by this plan",
            },
            {
                "id": "leaderboard",
                "sent": bool(context["leaderboard"]),
                "detail": f"{len(context['leaderboard'])} top matching experiment summaries by auto F1",
            },
        ],
        "include": {"dataset_preview": include_dataset_preview},
        "redactions": [
            "API keys are never sent",
            "full logs are not sent",
            "raw prediction rows are not sent",
            "dataset sample preview is sent only when explicitly enabled",
        ],
    }
    return context, manifest


def build_agent_plan_review_messages(context, manifest, prompt, language):
    context_json, warnings = _context_json_and_warnings(context, manifest)
    language_hint = _language_hint(language)
    system_prompt = (
        "You are the PepPlatform experiment execution review agent. "
        "Use only the provided JSON context. Return only strict JSON. Do not wrap it in Markdown. "
        "Write a formal report for the completed approved tool plan. "
        "Evaluate experiment outcomes from the available metrics, statuses, errors, and analysis artifacts, "
        "then recommend what to do next. Do not invent metrics, unseen raw data, or hidden log details."
    )
    schema_hint = {
        "summary": "concise outcome summary",
        "best_experiment": {
            "experiment_id": "experiment id or empty string",
            "name": "experiment name or empty string",
            "reason": "why this is best based on provided evidence",
        },
        "metric_evidence": ["metric or status evidence"],
        "failed_or_incomplete": [
            {
                "experiment_id": "experiment id",
                "status": "failed",
                "reason": "failure or missing evidence",
            }
        ],
        "analysis_findings": ["finding from analysis artifacts"],
        "action_outcomes": [
            {
                "action_id": "create-conservative-1",
                "experiment_id": "experiment id",
                "assessment": "outcome assessment",
            }
        ],
        "recommendations": ["next step 1", "next step 2"],
        "risks": ["remaining risk or missing evidence"],
    }
    user_message = (
        f"Answer explanatory text in {language_hint} unless the prompt strongly requests another language.\n"
        f"Review prompt:\n{str(prompt or '').strip()}\n\n"
        f"Required JSON shape:\n{json.dumps(schema_hint, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Context manifest:\n{json.dumps(manifest, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Agent execution context JSON:\n{context_json}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ], warnings


def parse_agent_plan_review(answer):
    payload = _load_agent_json(answer)
    if not isinstance(payload, dict):
        raise AgentResponseError("AI provider did not return a review object")
    metric_evidence = _string_list(payload.get("metric_evidence"))
    evidence = _string_list(payload.get("evidence")) or metric_evidence
    recommendations = _string_list(payload.get("recommendations")) or _string_list(
        payload.get("next_recommendations")
    )
    best_experiment = (
        payload.get("best_experiment") if isinstance(payload.get("best_experiment"), dict) else {}
    )
    return {
        "summary": str(payload.get("summary") or "").strip(),
        "best_experiment": best_experiment,
        "metric_evidence": metric_evidence,
        "failed_or_incomplete": [
            item
            for item in _safe_list(payload.get("failed_or_incomplete"))
            if isinstance(item, dict)
        ],
        "analysis_findings": _string_list(payload.get("analysis_findings")),
        "evidence": evidence,
        "action_outcomes": [
            item for item in _safe_list(payload.get("action_outcomes")) if isinstance(item, dict)
        ],
        "recommendations": recommendations,
        "next_recommendations": recommendations,
        "risks": _string_list(payload.get("risks")),
    }


def build_agent_chat_system_prompt(language="zh"):
    base = """You are a helpful AI assistant embedded in PepPlatform, a peptide machine-learning research platform.

PepPlatform enables researchers to train, evaluate, and compare ML models for peptide-related biological tasks. Users can design experiments, compare model performance on leaderboards, and run inference.

## Available Task Types

| Task ID | Type | Description |
|---------|------|-------------|
| pep_function | Multi-label | Predict functional labels for peptides (e.g., antimicrobial, anticancer, etc.) |
| pep_protein_interaction | Binary | Predict whether a peptide binds to a target protein |
| pep_hlaii | Binary | Predict peptide binding to HLA class II molecules |
| pep_hlai | Binary | Predict peptide binding to HLA class I molecules |

## Metrics Reference

- **F1 (Macro F1 for pep_function)**: Harmonic mean of precision and recall. Default sort metric for multi-label tasks. Higher is better (0-1).
- **AUC**: Area under the ROC curve. Threshold-independent measure of ranking quality. Default sort metric for binary tasks. Higher is better (0-1).
- **AUPR**: Area under the Precision-Recall curve. Better for imbalanced datasets than AUC. Higher is better (0-1).
- **SAcc**: Subset accuracy for multi-label tasks. Exact-match accuracy across all labels.
- **ACC**: Standard accuracy. Higher is better (0-1).
- **Precision**: TP / (TP + FP). Higher means fewer false positives.
- **Recall**: TP / (TP + FN). Higher means fewer false negatives.
- **Specificity**: TN / (TN + FP). Higher means fewer false positives on negatives.
- **MCC**: Matthews Correlation Coefficient. Balanced measure for binary classification. Range -1 to 1, 0 is random.
- **Balanced Accuracy**: Average of recall and specificity.
- **Micro AUC / Micro AUPR**: Micro-averaged variants for multi-label tasks.

## Available Configurable Routes

Users can configure experiments with:
- Feature routes: `pretrained` (PLM embeddings like ESM2, ProtBERT) or `classic` (AAC, PSSM, etc.)
- Tokenizers / Encoders for each route
- Classifiers: MLP, GBDT, etc.
- Training modes: `grid`, `kfold`, etc.
- For pair tasks: interaction levels, fusion strategies, graph predictors

## Your Role

- Answer questions about PepPlatform usage, experiment design, and metric interpretation
- Explain biological concepts related to peptides, proteins, and immune interactions
- Help users understand what metrics mean and how to interpret their results
- Suggest experiment configurations when asked (but direct users to the Experiment Design mode for formal proposals)
- Be concise and accurate. If you do not know an answer, say so.
- Do NOT fabricate experiment results, metrics, or user data you have not been shown.

Respond in the user's language. Most users communicate in Chinese or English."""
    if language.startswith("zh"):
        return "你是 PepPlatform 肽平台的内置 AI 助手。请用中文回答。\n\n" + base
    return base
