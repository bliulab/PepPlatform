import hashlib
import json

from .ai import _env_int


REPORT_SOURCE_SECTIONS = {
    "experiment",
    "dataset",
    "analysis",
    "model",
    "logs",
    "inference_runs",
}
REPORT_SEVERITIES = {"info", "warning", "critical"}
REPORT_PRIORITIES = {"high", "medium", "low"}


class ExperimentAnalysisResponseError(ValueError):
    pass


def _strip_json_fence(value):
    text = str(value or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _load_report_json(answer):
    text = _strip_json_fence(answer)
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ExperimentAnalysisResponseError(
                "AI provider did not return valid diagnostic JSON"
            )
        try:
            payload = json.loads(text[start : end + 1])
        except (TypeError, ValueError) as exc:
            raise ExperimentAnalysisResponseError(
                "AI provider did not return valid diagnostic JSON"
            ) from exc
    if not isinstance(payload, dict):
        raise ExperimentAnalysisResponseError("AI provider diagnostic response must be an object")
    return payload


def _clean_text(value, limit=4000):
    return str(value or "").strip()[:limit]


def _string_list(value, limit=8):
    if not isinstance(value, list):
        value = [] if value in [None, ""] else [value]
    return [_clean_text(item, 1000) for item in value if _clean_text(item, 1000)][:limit]


def _source(value, fallback="experiment"):
    source = str(value or "").strip().lower()
    return source if source in REPORT_SOURCE_SECTIONS else fallback


def _source_list(value):
    values = value if isinstance(value, list) else [value]
    sources = []
    for item in values:
        source = _source(item, "")
        if source and source not in sources:
            sources.append(source)
    return sources or ["experiment"]


def parse_experiment_diagnostic_report(answer, dataset, experiment, status):
    del dataset
    payload = _load_report_json(answer)
    summary = _clean_text(payload.get("summary"), 8000)
    if not summary:
        raise ExperimentAnalysisResponseError(
            "AI provider diagnostic response is missing a summary"
        )

    evidence = []
    for raw in (payload.get("evidence") if isinstance(payload.get("evidence"), list) else [])[:10]:
        item = raw if isinstance(raw, dict) else {}
        label = _clean_text(item.get("label"), 300)
        interpretation = _clean_text(item.get("interpretation"), 2000)
        if not label or not interpretation:
            continue
        evidence.append(
            {
                "label": label,
                "value": _clean_text(item.get("value"), 500),
                "interpretation": interpretation,
                "source_section": _source(item.get("source_section")),
            }
        )

    findings = []
    for raw in (payload.get("findings") if isinstance(payload.get("findings"), list) else [])[:10]:
        item = raw if isinstance(raw, dict) else {}
        title = _clean_text(item.get("title"), 400)
        detail = _clean_text(item.get("detail"), 3000)
        if not title or not detail:
            continue
        severity = str(item.get("severity") or "info").strip().lower()
        findings.append(
            {
                "severity": severity if severity in REPORT_SEVERITIES else "info",
                "title": title,
                "detail": detail,
                "source_sections": _source_list(item.get("source_sections")),
            }
        )

    warnings = []
    recommendations = []
    raw_recommendations = payload.get("recommendations")
    if not isinstance(raw_recommendations, list):
        raw_recommendations = []
    for index, raw in enumerate(raw_recommendations[:3]):
        item = raw if isinstance(raw, dict) else {}
        title = _clean_text(item.get("title"), 400)
        rationale = _clean_text(item.get("rationale"), 3000)
        if not title or not rationale:
            continue
        priority = str(item.get("priority") or "medium").strip().lower()
        recommendation = {
            "id": f"recommendation-{index + 1}",
            "title": title,
            "rationale": rationale,
            "priority": priority if priority in REPORT_PRIORITIES else "medium",
            "success_criteria": _string_list(item.get("success_criteria"), 6),
        }
        recommendations.append(recommendation)

    report = {
        "status": "provisional" if status == "running" else "final",
        "summary": summary,
        "evidence": evidence,
        "findings": findings,
        "recommendations": recommendations,
        "limitations": _string_list(payload.get("limitations"), 10),
    }
    return report, warnings


def report_as_markdown(report):
    lines = ["# Experiment diagnostic report", "", str(report.get("summary") or "").strip()]
    evidence = report.get("evidence") if isinstance(report.get("evidence"), list) else []
    if evidence:
        lines.extend(["", "## Evidence"])
        for item in evidence:
            lines.append(
                f"- **{item.get('label')}**: {item.get('value')} — {item.get('interpretation')} "
                f"(source: {item.get('source_section')})"
            )
    findings = report.get("findings") if isinstance(report.get("findings"), list) else []
    if findings:
        lines.extend(["", "## Findings"])
        for item in findings:
            lines.append(f"- **{item.get('title')}**: {item.get('detail')}")
    recommendations = (
        report.get("recommendations") if isinstance(report.get("recommendations"), list) else []
    )
    if recommendations:
        lines.extend(["", "## Recommendations"])
        for item in recommendations:
            lines.append(f"- **{item.get('title')}**: {item.get('rationale')}")
    limitations = report.get("limitations") if isinstance(report.get("limitations"), list) else []
    if limitations:
        lines.extend(["", "## Limitations"])
        lines.extend(f"- {item}" for item in limitations)
    return "\n".join(lines).strip()


def enrich_context_manifest(manifest, experiment, context_json, captured_at):
    enriched = dict(manifest or {})
    enriched.update(
        {
            "captured_at": captured_at,
            "experiment_status": experiment.get("status"),
            "experiment_updated_at": experiment.get("updated_at"),
            "context_chars": len(context_json),
            "context_hash": hashlib.sha256(context_json.encode("utf-8")).hexdigest()[:20],
        }
    )
    return enriched


def _bounded_context_json(context):
    context_json = json.dumps(context, ensure_ascii=False, sort_keys=True, default=str)
    max_context_chars = max(8000, _env_int("PEP_LLM_MAX_CONTEXT_CHARS", 60000))
    return context_json[:max_context_chars]


def build_experiment_report_messages(context, manifest, prompt, language):
    language_hint = "Chinese" if str(language or "").lower().startswith("zh") else "English"
    context_json = _bounded_context_json(context)
    schema = {
        "summary": "concise synthesis",
        "evidence": [
            {
                "label": "metric or observation",
                "value": "observed value",
                "interpretation": "what the evidence means",
                "source_section": "experiment|dataset|analysis|model|logs|inference_runs",
            }
        ],
        "findings": [
            {
                "severity": "info|warning|critical",
                "title": "finding title",
                "detail": "evidence-grounded explanation",
                "source_sections": ["experiment"],
            }
        ],
        "recommendations": [
            {
                "title": "research direction",
                "rationale": "why this direction should be investigated",
                "priority": "high|medium|low",
                "success_criteria": ["what evidence would support or reject this direction"],
            }
        ],
        "limitations": ["missing evidence or uncertainty"],
    }
    system = (
        "You are the PepPlatform experiment diagnostic assistant. Use only the supplied experiment context. "
        "Do not invent metrics, samples, model behavior, or configuration fields. Distinguish observation from inference. "
        "Return strict JSON only, matching the requested schema. Provide one to three research directions. "
        "Do not provide executable experiment configs, parameter diffs, or Preflight results. "
        "Describe what should be investigated and what evidence would validate the direction; formal experiment "
        "configurations are generated later in the AI experiment planning workspace."
    )
    user = (
        f"Write all human-facing text in {language_hint}.\n"
        f"Diagnostic objective:\n{str(prompt or '').strip()}\n\n"
        f"Required JSON shape:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Context manifest:\n{json.dumps(manifest, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Experiment context:\n{context_json}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], context_json


def build_experiment_report_repair_messages(messages, answer, reason):
    return [
        *messages,
        {"role": "assistant", "content": str(answer or "")},
        {
            "role": "user",
            "content": (
                "The previous response failed validation. Rewrite the complete response as strict JSON only. "
                f"Do not add markdown fences or commentary. Validation failure: {str(reason)[:500]}"
            ),
        },
    ]


def build_experiment_followup_messages(context, manifest, history, language):
    language_hint = "Chinese" if str(language or "").lower().startswith("zh") else "English"
    context_json = _bounded_context_json(context)
    system = (
        "You are the PepPlatform experiment diagnostic assistant. Answer the follow-up using only the current "
        "experiment context and conversation history. Be concise, evidence-grounded, and explicit when evidence is "
        f"missing. Answer in {language_hint}. Use Markdown, not JSON."
    )
    context_message = (
        f"Current context manifest:\n{json.dumps(manifest, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Current experiment context:\n{context_json}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "system", "content": context_message},
        *(history or []),
    ], context_json
