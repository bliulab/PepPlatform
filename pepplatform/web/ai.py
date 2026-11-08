import base64
import errno
import hashlib
import hmac
import http.client
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from pepplatform.analysis import empty_analysis, load_analysis

from .dataset_reports import build_dataset_report


DEFAULT_PROVIDER = "deepseek"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"
DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_COMPATIBLE_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
ANTHROPIC_VERSION = "2023-06-01"
BASE_OPENAI_COMPATIBLE_CAPABILITIES = {
    "supports_json_object": True,
    "supports_reasoning_toggle": False,
    "reasoning_control": "none",
    "default_max_tokens": 2200,
}
PROVIDER_ORDER = [
    "deepseek",
    "qwen_dashscope",
    "kimi",
    "zhipu_glm",
    "xiaomi_mimo",
    "minimax",
    "siliconflow",
    "openai",
    "google_gemini",
    "anthropic_claude",
    "openrouter",
    "together",
    "deepinfra",
    "openai_compatible",
]
PROVIDER_DEFAULTS = {
    "deepseek": {
        "display_name": "DeepSeek",
        "base_url": DEFAULT_DEEPSEEK_BASE_URL,
        "model": DEFAULT_DEEPSEEK_MODEL,
        "description": "DeepSeek hosted chat models.",
        "capabilities": {
            **BASE_OPENAI_COMPATIBLE_CAPABILITIES,
            "supports_json_object": True,
            "supports_reasoning_toggle": True,
            "reasoning_control": "thinking_disabled",
            "default_max_tokens": 2200,
        },
    },
    "qwen_dashscope": {
        "display_name": "Qwen / DashScope",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "description": "Alibaba Cloud Model Studio Qwen models through the OpenAI-compatible endpoint.",
        "capabilities": {**BASE_OPENAI_COMPATIBLE_CAPABILITIES},
    },
    "kimi": {
        "display_name": "Kimi",
        "base_url": "https://api.moonshot.ai/v1",
        "model": "kimi-k2.6",
        "description": "Moonshot Kimi models through the OpenAI-compatible endpoint.",
        "capabilities": {**BASE_OPENAI_COMPATIBLE_CAPABILITIES},
    },
    "zhipu_glm": {
        "display_name": "Zhipu GLM",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "GLM-4.7-Flash",
        "description": "Zhipu / Z.AI GLM models through the OpenAI-compatible endpoint.",
        "capabilities": {**BASE_OPENAI_COMPATIBLE_CAPABILITIES},
    },
    "xiaomi_mimo": {
        "display_name": "Xiaomi MiMo",
        "base_url": "https://api.xiaomimimo.com/v1",
        "model": "mimo-v2.5-pro",
        "description": "Xiaomi MiMo models through the OpenAI-compatible endpoint.",
        "capabilities": {**BASE_OPENAI_COMPATIBLE_CAPABILITIES},
    },
    "minimax": {
        "display_name": "MiniMax",
        "base_url": "https://api.minimax.io/v1",
        "model": "MiniMax-M3",
        "description": "MiniMax text models through the OpenAI-compatible endpoint.",
        "capabilities": {**BASE_OPENAI_COMPATIBLE_CAPABILITIES},
    },
    "siliconflow": {
        "display_name": "SiliconFlow",
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Pro/zai-org/GLM-4.7",
        "description": "SiliconFlow model platform through the OpenAI-compatible endpoint.",
        "capabilities": {**BASE_OPENAI_COMPATIBLE_CAPABILITIES},
    },
    "openai": {
        "display_name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "model": DEFAULT_OPENAI_COMPATIBLE_MODEL,
        "description": "OpenAI chat models.",
        "capabilities": {**BASE_OPENAI_COMPATIBLE_CAPABILITIES},
    },
    "google_gemini": {
        "display_name": "Google Gemini",
        "base_url": DEFAULT_GEMINI_BASE_URL,
        "model": DEFAULT_GEMINI_MODEL,
        "description": "Google Gemini models through the official OpenAI-compatible endpoint.",
        "capabilities": {**BASE_OPENAI_COMPATIBLE_CAPABILITIES},
    },
    "anthropic_claude": {
        "display_name": "Anthropic Claude",
        "base_url": DEFAULT_ANTHROPIC_BASE_URL,
        "model": DEFAULT_ANTHROPIC_MODEL,
        "description": "Anthropic Claude models through the native Messages API.",
        "capabilities": {**BASE_OPENAI_COMPATIBLE_CAPABILITIES},
    },
    "openrouter": {
        "display_name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4o-mini",
        "description": "OpenRouter unified model gateway. Edit the model id to match your account/catalog.",
        "capabilities": {**BASE_OPENAI_COMPATIBLE_CAPABILITIES},
    },
    "together": {
        "display_name": "Together AI",
        "base_url": "https://api.together.ai/v1",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "description": "Together AI hosted open-source models. Edit the model id to match your account/catalog.",
        "capabilities": {**BASE_OPENAI_COMPATIBLE_CAPABILITIES},
    },
    "deepinfra": {
        "display_name": "DeepInfra",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "description": "DeepInfra hosted open-source models. Edit the model id to match your account/catalog.",
        "capabilities": {**BASE_OPENAI_COMPATIBLE_CAPABILITIES},
    },
    "openai_compatible": {
        "display_name": "OpenAI-compatible",
        "base_url": DEFAULT_OPENAI_COMPATIBLE_BASE_URL,
        "model": DEFAULT_OPENAI_COMPATIBLE_MODEL,
        "description": "Custom endpoint compatible with OpenAI Chat Completions.",
        "capabilities": {**BASE_OPENAI_COMPATIBLE_CAPABILITIES},
    },
}
SUPPORTED_PROVIDERS = set(PROVIDER_DEFAULTS)
ENCRYPTION_ENV = "PEP_LLM_KEY_ENCRYPTION_SECRET"
LOGGER = logging.getLogger(__name__)


class AiConfigurationError(RuntimeError):
    pass


class AiProviderError(RuntimeError):
    pass


class AiStreamInterruptedError(AiProviderError):
    def __init__(self, partial=False, retries=0):
        super().__init__("AI provider stream connection was interrupted")
        self.partial = bool(partial)
        self.retries = max(0, int(retries))


_TRANSIENT_STREAM_ERRNOS = {
    errno.EPIPE,
    errno.ECONNABORTED,
    errno.ECONNRESET,
    errno.ETIMEDOUT,
}


def _is_transient_stream_error(exc):
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if reason is not exc and _is_transient_stream_error(reason):
        return True
    if isinstance(
        reason,
        (
            BrokenPipeError,
            ConnectionAbortedError,
            ConnectionResetError,
            TimeoutError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
        ),
    ):
        return True
    return isinstance(reason, OSError) and reason.errno in _TRANSIENT_STREAM_ERRNOS


def _normalize_provider_capabilities(capabilities):
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    normalized = dict(capabilities)
    for key in ["supports_json_object", "supports_reasoning_toggle"]:
        normalized[key] = bool(normalized.get(key))
    reasoning_control = str(normalized.get("reasoning_control") or "none").strip().lower()
    if reasoning_control not in {"none", "thinking_disabled"}:
        reasoning_control = "none"
    normalized["reasoning_control"] = reasoning_control
    try:
        normalized["default_max_tokens"] = max(
            256, min(int(normalized.get("default_max_tokens") or 2200), 6000)
        )
    except (TypeError, ValueError):
        normalized["default_max_tokens"] = 2200
    return normalized


def provider_presets():
    presets = []
    for provider in PROVIDER_ORDER:
        defaults = PROVIDER_DEFAULTS[provider]
        presets.append(
            {
                "provider": provider,
                "display_name": defaults["display_name"],
                "base_url": defaults["base_url"],
                "model": defaults["model"],
                "description": defaults.get("description") or "",
                "capabilities": _normalize_provider_capabilities(defaults.get("capabilities")),
            }
        )
    return presets


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _encryption_secret():
    secret = os.environ.get(ENCRYPTION_ENV, "").strip()
    if not secret:
        raise AiConfigurationError(f"{ENCRYPTION_ENV} is required to store AI provider API keys")
    if len(secret) < 16:
        raise AiConfigurationError(f"{ENCRYPTION_ENV} must be at least 16 characters")
    return secret.encode("utf-8")


def _derive_keys(secret, salt):
    material = hashlib.pbkdf2_hmac("sha256", secret, salt, 200_000, dklen=64)
    return material[:32], material[32:]


def _keystream(key, nonce, length):
    chunks = []
    counter = 0
    while sum(len(chunk) for chunk in chunks) < length:
        counter_bytes = counter.to_bytes(8, "big")
        chunks.append(hmac.new(key, nonce + counter_bytes, hashlib.sha256).digest())
        counter += 1
    return b"".join(chunks)[:length]


def _xor_bytes(left, right):
    return bytes(a ^ b for a, b in zip(left, right))


def encrypt_api_key(api_key):
    api_key = str(api_key or "").strip()
    if not api_key:
        raise ValueError("api_key is required")
    salt = os.urandom(16)
    nonce = os.urandom(16)
    enc_key, mac_key = _derive_keys(_encryption_secret(), salt)
    plaintext = api_key.encode("utf-8")
    ciphertext = _xor_bytes(plaintext, _keystream(enc_key, nonce, len(plaintext)))
    authenticated = b"v1" + salt + nonce + ciphertext
    tag = hmac.new(mac_key, authenticated, hashlib.sha256).digest()
    return "v1:" + base64.urlsafe_b64encode(salt + nonce + tag + ciphertext).decode("ascii")


def decrypt_api_key(ciphertext_value):
    text = str(ciphertext_value or "")
    if not text.startswith("v1:"):
        raise AiConfigurationError("AI provider API key ciphertext is invalid")
    try:
        raw = base64.urlsafe_b64decode(text.split(":", 1)[1].encode("ascii"))
    except (ValueError, TypeError) as exc:
        raise AiConfigurationError("AI provider API key ciphertext is invalid") from exc
    if len(raw) < 16 + 16 + 32:
        raise AiConfigurationError("AI provider API key ciphertext is invalid")
    salt = raw[:16]
    nonce = raw[16:32]
    tag = raw[32:64]
    encrypted = raw[64:]
    enc_key, mac_key = _derive_keys(_encryption_secret(), salt)
    expected = hmac.new(mac_key, b"v1" + salt + nonce + encrypted, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected):
        raise AiConfigurationError("AI provider API key could not be decrypted")
    plaintext = _xor_bytes(encrypted, _keystream(enc_key, nonce, len(encrypted)))
    return plaintext.decode("utf-8")


def normalize_ai_provider_payload(payload, existing=None, require_api_key=False):
    payload = payload or {}
    existing = existing or {}
    provider = (
        str(payload.get("provider", existing.get("provider") or DEFAULT_PROVIDER)).strip().lower()
    )
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"unsupported AI provider: {provider}")
    defaults = PROVIDER_DEFAULTS[provider]
    display_name = str(
        payload.get("display_name", existing.get("display_name") or defaults["display_name"])
    ).strip()
    if not display_name:
        raise ValueError("display_name is required")
    if len(display_name) > 80:
        raise ValueError("display_name must be at most 80 characters")
    base_url = str(
        payload.get("base_url", existing.get("base_url") or defaults["base_url"])
    ).strip()
    if not base_url.startswith(("https://", "http://")):
        raise ValueError("base_url must be an HTTP(S) URL")
    model = str(payload.get("model", existing.get("model") or defaults["model"])).strip()
    if not model:
        raise ValueError("model is required")
    api_key = str(payload.get("api_key") or "").strip()
    if require_api_key and not api_key:
        raise ValueError("api_key is required")
    enabled = payload.get("enabled", existing.get("enabled", True))
    raw_capabilities = (
        existing.get("capabilities") if isinstance(existing.get("capabilities"), dict) else {}
    )
    payload_capabilities = (
        payload.get("capabilities") if isinstance(payload.get("capabilities"), dict) else {}
    )
    capabilities = {**defaults["capabilities"], **raw_capabilities, **payload_capabilities}
    capabilities = _normalize_provider_capabilities(capabilities)
    return {
        "provider": provider,
        "display_name": display_name,
        "base_url": base_url.rstrip("/"),
        "model": model,
        "api_key": api_key,
        "capabilities": capabilities,
        "enabled": bool(enabled),
    }


def api_key_last4(api_key):
    text = str(api_key or "")
    return text[-4:] if text else ""


def _chat_endpoint(base_url):
    base = str(base_url or DEFAULT_DEEPSEEK_BASE_URL).rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _anthropic_messages_endpoint(base_url):
    base = str(base_url or DEFAULT_ANTHROPIC_BASE_URL).rstrip("/")
    if base.endswith("/messages"):
        return base
    return f"{base}/messages"


def _is_deepseek_v4_model(model):
    return str(model or "").strip().lower().startswith("deepseek-v4")


def _supports_thinking_parameter(provider, model):
    return provider == "deepseek" and _is_deepseek_v4_model(model)


def _message_text(message):
    content = (message or {}).get("content")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") if item.get("type") == "text" else item.get("content")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _anthropic_payload_messages(messages, force_json_prompt=False):
    system_parts = []
    converted = []
    for message in messages or []:
        role = str((message or {}).get("role") or "").strip().lower()
        content = _message_text(message)
        if not content:
            continue
        if role in {"system", "developer"}:
            system_parts.append(content)
            continue
        if role not in {"user", "assistant"}:
            role = "user"
        if converted and converted[-1]["role"] == role:
            converted[-1]["content"] = f"{converted[-1]['content']}\n\n{content}"
        else:
            converted.append({"role": role, "content": content})

    if not converted:
        converted.append({"role": "user", "content": "Reply with a short OK message."})
    if converted[0]["role"] == "assistant":
        converted.insert(0, {"role": "user", "content": "Continue."})
    if force_json_prompt:
        system_parts.append("Return only a valid JSON object. Do not wrap it in Markdown.")
    return "\n\n".join(system_parts).strip(), converted


def _anthropic_json_object_output_config():
    return {
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
            },
        }
    }


def _iter_sse_events(response):
    """Parse SSE frames without assuming transport chunks align to lines."""
    buffer = ""
    event_name = "message"
    data_lines = []

    def dispatch():
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = "message"
            return None
        event = {"event": event_name, "data": "\n".join(data_lines)}
        event_name = "message"
        data_lines = []
        return event

    for raw_chunk in response:
        if isinstance(raw_chunk, bytes):
            chunk = raw_chunk.decode("utf-8", errors="replace")
        else:
            chunk = str(raw_chunk)
        buffer += chunk.replace("\r\n", "\n").replace("\r", "\n")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            if line == "":
                event = dispatch()
                if event:
                    yield event
                continue
            if line.startswith(":"):
                yield {"event": "heartbeat", "data": ""}
                continue
            field, separator, value = line.partition(":")
            if separator and value.startswith(" "):
                value = value[1:]
            if field == "event":
                event_name = value or "message"
            elif field == "data":
                data_lines.append(value)
    if buffer:
        field, separator, value = buffer.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value or "message"
        elif field == "data":
            data_lines.append(value)
    event = dispatch()
    if event:
        yield event


class AiChatClient:
    def __init__(self, timeout_seconds=None):
        self.timeout_seconds = timeout_seconds or max(
            1, _env_int("PEP_LLM_HTTP_TIMEOUT_SECONDS", 60)
        )

    def chat(
        self,
        provider_config,
        api_key,
        messages,
        max_tokens=1200,
        temperature=0.2,
        response_format=None,
    ):
        provider = str(provider_config.get("provider") or "").lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise AiProviderError(f"unsupported AI provider: {provider}")
        model = (
            provider_config.get("model")
            or PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS[DEFAULT_PROVIDER])["model"]
        )
        raw_capabilities = (
            provider_config.get("capabilities")
            if isinstance(provider_config.get("capabilities"), dict)
            else {}
        )
        capabilities = _normalize_provider_capabilities(
            {
                **PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS[DEFAULT_PROVIDER])[
                    "capabilities"
                ],
                **raw_capabilities,
            }
        )
        if provider == "anthropic_claude":
            return self._chat_anthropic(
                provider_config,
                api_key,
                messages,
                model=model,
                capabilities=capabilities,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )
        return self._chat_openai_compatible(
            provider_config,
            api_key,
            messages,
            model=model,
            capabilities=capabilities,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )

    def stream_chat(
        self,
        provider_config,
        api_key,
        messages,
        max_tokens=1200,
        temperature=0.2,
        response_format=None,
    ):
        provider = str(provider_config.get("provider") or "").lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise AiProviderError(f"unsupported AI provider: {provider}")
        model = (
            provider_config.get("model")
            or PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS[DEFAULT_PROVIDER])["model"]
        )
        raw_capabilities = (
            provider_config.get("capabilities")
            if isinstance(provider_config.get("capabilities"), dict)
            else {}
        )
        capabilities = _normalize_provider_capabilities(
            {
                **PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS[DEFAULT_PROVIDER])[
                    "capabilities"
                ],
                **raw_capabilities,
            }
        )
        if provider == "anthropic_claude":
            yield from self._stream_anthropic(
                provider_config,
                api_key,
                messages,
                model=model,
                capabilities=capabilities,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )
            return
        yield from self._stream_openai_compatible(
            provider_config,
            api_key,
            messages,
            model=model,
            capabilities=capabilities,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )

    def _stream_openai_compatible(
        self,
        provider_config,
        api_key,
        messages,
        model,
        capabilities,
        max_tokens=1200,
        temperature=0.2,
        response_format=None,
    ):
        provider = str(provider_config.get("provider") or "").lower()
        request_payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": True,
        }
        if isinstance(response_format, dict) and capabilities.get("supports_json_object"):
            request_payload["response_format"] = response_format
        if (
            capabilities.get("reasoning_control") == "thinking_disabled"
            and capabilities.get("supports_reasoning_toggle")
            and _supports_thinking_parameter(provider, model)
        ):
            request_payload["thinking"] = {"type": "disabled"}
        answer_parts = []
        reasoning_parts = []
        finish_reason = ""
        usage = {}
        raw_id = None
        response_model = model
        started_at = time.monotonic()
        first_delta_at = None
        retries = 0
        for attempt in range(2):
            request = urllib.request.Request(
                _chat_endpoint(provider_config.get("base_url")),
                data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
                method="POST",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    for event in _iter_sse_events(response):
                        if event["event"] == "heartbeat":
                            yield {"type": "heartbeat"}
                            continue
                        raw = event["data"].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        try:
                            payload = json.loads(raw)
                        except (TypeError, ValueError) as exc:
                            raise AiProviderError(
                                "AI provider returned an invalid stream event"
                            ) from exc
                        if payload.get("error"):
                            detail = payload.get("error")
                            if isinstance(detail, dict):
                                detail = detail.get("message") or json.dumps(
                                    detail, ensure_ascii=False
                                )
                            raise AiProviderError(f"AI provider stream failed: {detail}")
                        raw_id = payload.get("id") or raw_id
                        response_model = payload.get("model") or response_model
                        if isinstance(payload.get("usage"), dict):
                            usage = payload["usage"]
                        choice = (payload.get("choices") or [{}])[0]
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if isinstance(content, list):
                            content = "".join(
                                str(item.get("text") or "")
                                for item in content
                                if isinstance(item, dict)
                            )
                        content = str(content or "")
                        reasoning = str(delta.get("reasoning_content") or "")
                        if reasoning:
                            reasoning_parts.append(reasoning)
                            yield {"type": "heartbeat"}
                        if content:
                            if first_delta_at is None:
                                first_delta_at = time.monotonic()
                            answer_parts.append(content)
                            yield {"type": "delta", "content": content}
                        if choice.get("finish_reason") is not None:
                            finish_reason = str(choice.get("finish_reason") or "")
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:600]
                raise AiProviderError(f"AI provider returned HTTP {exc.code}: {detail}") from exc
            except Exception as exc:
                if not _is_transient_stream_error(exc):
                    if isinstance(exc, urllib.error.URLError):
                        raise AiProviderError(f"AI provider request failed: {exc.reason}") from exc
                    raise
                partial = bool("".join(answer_parts).strip())
                if not partial and attempt == 0:
                    retries = 1
                    LOGGER.warning(
                        "AI stream transport retry provider=%s model=%s error=%s",
                        provider,
                        model,
                        type(exc).__name__,
                    )
                    reasoning_parts.clear()
                    finish_reason = ""
                    usage = {}
                    raw_id = None
                    response_model = model
                    time.sleep(0.25)
                    continue
                LOGGER.warning(
                    "AI stream interrupted provider=%s model=%s partial=%s chars=%s retries=%s error=%s",
                    provider,
                    model,
                    partial,
                    len("".join(answer_parts)),
                    retries,
                    type(exc).__name__,
                )
                raise AiStreamInterruptedError(partial=partial, retries=retries) from exc
        if not "".join(answer_parts).strip():
            if "".join(reasoning_parts).strip():
                raise AiProviderError(
                    "AI provider returned reasoning content but no final answer; disable thinking or increase max_tokens"
                )
            raise AiProviderError("AI provider returned an empty answer")
        finished_at = time.monotonic()
        LOGGER.info(
            "AI stream completed provider=%s model=%s first_delta_ms=%s duration_ms=%s chars=%s retries=%s",
            provider,
            response_model,
            round(((first_delta_at or finished_at) - started_at) * 1000),
            round((finished_at - started_at) * 1000),
            len("".join(answer_parts)),
            retries,
        )
        yield {
            "type": "done",
            "provider": provider,
            "model": response_model,
            "finish_reason": finish_reason,
            "usage": usage,
            "raw_id": raw_id,
        }

    def _stream_anthropic(
        self,
        provider_config,
        api_key,
        messages,
        model,
        capabilities,
        max_tokens=1200,
        temperature=0.2,
        response_format=None,
    ):
        request_payload = self._anthropic_request_payload(
            model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            supports_json_object=capabilities.get("supports_json_object"),
        )
        request_payload["stream"] = True
        request = urllib.request.Request(
            _anthropic_messages_endpoint(provider_config.get("base_url")),
            data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
        )
        answer_parts = []
        finish_reason = ""
        usage = {}
        raw_id = None
        response_model = model
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                for event in _iter_sse_events(response):
                    if event["event"] == "heartbeat":
                        yield {"type": "heartbeat"}
                        continue
                    raw = event["data"].strip()
                    if not raw:
                        continue
                    try:
                        payload = json.loads(raw)
                    except (TypeError, ValueError) as exc:
                        raise AiProviderError(
                            "AI provider returned an invalid stream event"
                        ) from exc
                    event_type = str(payload.get("type") or event["event"] or "")
                    if event_type == "error":
                        detail = payload.get("error") or {}
                        if isinstance(detail, dict):
                            detail = detail.get("message") or json.dumps(detail, ensure_ascii=False)
                        raise AiProviderError(f"AI provider stream failed: {detail}")
                    if event_type == "message_start":
                        message = payload.get("message") or {}
                        raw_id = message.get("id") or raw_id
                        response_model = message.get("model") or response_model
                        if isinstance(message.get("usage"), dict):
                            usage.update(message["usage"])
                    elif event_type == "content_block_delta":
                        delta = payload.get("delta") or {}
                        if delta.get("type") == "text_delta":
                            content = str(delta.get("text") or "")
                            if content:
                                answer_parts.append(content)
                                yield {"type": "delta", "content": content}
                    elif event_type == "message_delta":
                        delta = payload.get("delta") or {}
                        finish_reason = str(delta.get("stop_reason") or finish_reason)
                        if isinstance(payload.get("usage"), dict):
                            usage.update(payload["usage"])
                    elif event_type == "ping":
                        yield {"type": "heartbeat"}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:600]
            raise AiProviderError(f"AI provider returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise AiProviderError(f"AI provider request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise AiProviderError("AI provider request timed out") from exc
        if not "".join(answer_parts).strip():
            raise AiProviderError("AI provider returned an empty answer")
        yield {
            "type": "done",
            "provider": "anthropic_claude",
            "model": response_model,
            "finish_reason": finish_reason,
            "usage": usage,
            "raw_id": raw_id,
        }

    def _chat_openai_compatible(
        self,
        provider_config,
        api_key,
        messages,
        model,
        capabilities,
        max_tokens=1200,
        temperature=0.2,
        response_format=None,
    ):
        provider = str(provider_config.get("provider") or "").lower()
        request_payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        if isinstance(response_format, dict) and capabilities.get("supports_json_object"):
            request_payload["response_format"] = response_format
        if (
            capabilities.get("reasoning_control") == "thinking_disabled"
            and capabilities.get("supports_reasoning_toggle")
            and _supports_thinking_parameter(provider, model)
        ):
            request_payload["thinking"] = {"type": "disabled"}
        body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            _chat_endpoint(provider_config.get("base_url")),
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:600]
            raise AiProviderError(f"AI provider returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise AiProviderError(f"AI provider request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise AiProviderError("AI provider request timed out") from exc

        try:
            payload = json.loads(raw)
            choice = (payload.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            answer = str(message.get("content") or "").strip()
            reasoning = str(message.get("reasoning_content") or "").strip()
        except (TypeError, ValueError, KeyError) as exc:
            raise AiProviderError("AI provider returned an invalid response") from exc
        if not answer:
            if reasoning:
                raise AiProviderError(
                    "AI provider returned reasoning content but no final answer; disable thinking or increase max_tokens"
                )
            raise AiProviderError("AI provider returned an empty answer")
        return {
            "answer": answer,
            "provider": provider,
            "model": request_payload["model"],
            "usage": payload.get("usage") or {},
            "raw_id": payload.get("id"),
        }

    def _anthropic_request_payload(
        self,
        model,
        messages,
        max_tokens=1200,
        temperature=0.2,
        response_format=None,
        supports_json_object=True,
        force_json_prompt=False,
    ):
        system, converted_messages = _anthropic_payload_messages(
            messages, force_json_prompt=force_json_prompt
        )
        request_payload = {
            "model": model,
            "messages": converted_messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if system:
            request_payload["system"] = system
        if isinstance(response_format, dict) and supports_json_object and not force_json_prompt:
            request_payload["output_config"] = _anthropic_json_object_output_config()
        return request_payload

    def _post_anthropic_messages(self, provider_config, api_key, request_payload):
        body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            _anthropic_messages_endpoint(provider_config.get("base_url")),
            data=body,
            method="POST",
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw)

    def _chat_anthropic(
        self,
        provider_config,
        api_key,
        messages,
        model,
        capabilities,
        max_tokens=1200,
        temperature=0.2,
        response_format=None,
    ):
        request_payload = self._anthropic_request_payload(
            model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            supports_json_object=capabilities.get("supports_json_object"),
        )
        try:
            try:
                payload = self._post_anthropic_messages(provider_config, api_key, request_payload)
            except urllib.error.HTTPError as exc:
                if not (exc.code == 400 and "output_config" in request_payload):
                    raise
                request_payload = self._anthropic_request_payload(
                    model,
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=None,
                    supports_json_object=False,
                    force_json_prompt=True,
                )
                payload = self._post_anthropic_messages(provider_config, api_key, request_payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:600]
            raise AiProviderError(f"AI provider returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise AiProviderError(f"AI provider request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise AiProviderError("AI provider request timed out") from exc
        except (TypeError, ValueError) as exc:
            raise AiProviderError("AI provider returned an invalid response") from exc

        try:
            content = payload.get("content") or []
            if isinstance(content, str):
                answer = content.strip()
            else:
                answer = "\n".join(
                    str(item.get("text") or "").strip()
                    for item in content
                    if isinstance(item, dict)
                    and item.get("type") == "text"
                    and str(item.get("text") or "").strip()
                ).strip()
        except (TypeError, ValueError, KeyError) as exc:
            raise AiProviderError("AI provider returned an invalid response") from exc
        if not answer:
            raise AiProviderError("AI provider returned an empty answer")
        return {
            "answer": answer,
            "provider": "anthropic_claude",
            "model": payload.get("model") or request_payload["model"],
            "usage": payload.get("usage") or {},
            "raw_id": payload.get("id"),
        }


def _trim_text(value, limit):
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[-limit:]


def _redact_sequence_like_text(value):
    def replace(match):
        token = match.group(0)
        return f"<redacted_sequence len={len(token)}>"

    return re.sub(
        r"(?<![A-Za-z])([ACDEFGHIKLMNPQRSTVWYXBZJUO]{8,})(?![A-Za-z])", replace, str(value or "")
    )


def _compact_metric_events(events, limit=12):
    events = list(events or [])
    compact = []
    for event in events[-limit:]:
        if not isinstance(event, dict):
            continue
        compact.append(
            {
                key: value
                for key, value in event.items()
                if key
                in {
                    "kind",
                    "epoch",
                    "total",
                    "train_loss",
                    "val_loss",
                    "val_acc",
                    "val_sacc",
                    "val_f1",
                    "val_auc",
                    "val_aupr",
                    "best_val_f1",
                    "best_val_auc",
                    "best_epoch",
                    "test_acc",
                    "test_sacc",
                    "test_f1",
                    "test_auc",
                    "test_aupr",
                    "test_balanced_accuracy",
                    "test_precision",
                    "test_recall",
                    "test_specificity",
                    "test_mcc",
                }
            }
        )
    return compact


def _compact_dataset_report(dataset, include_preview):
    report = build_dataset_report(dataset, preview_limit=5 if include_preview else 1)
    compact_splits = []
    for split in report.get("splits") or []:
        compact = {
            key: split.get(key)
            for key in [
                "name",
                "sample_count",
                "files",
                "columns",
                "label_mode",
                "label_counts",
                "sequence_count",
                "sequence_lengths",
                "peptide_lengths",
                "partner_lengths",
            ]
            if key in split
        }
        if include_preview:
            compact["preview"] = split.get("preview") or []
        compact_splits.append(compact)
    return {
        "dataset_id": report.get("dataset_id"),
        "status": report.get("status"),
        "valid": report.get("valid"),
        "task_type": report.get("task_type"),
        "format": report.get("format"),
        "summary": report.get("summary"),
        "splits": compact_splits,
        "issues": report.get("issues") or [],
    }


def _compact_curve_payload(curve):
    curve = curve or {}
    return {key: curve.get(key) for key in ["available", "reason", "auc", "aupr"] if key in curve}


def _compact_analysis_payload(analysis):
    analysis = analysis or empty_analysis()
    curves = {}
    for split, payload in (analysis.get("curves") or {}).items():
        if not isinstance(payload, dict):
            continue
        curves[split] = {
            "average": payload.get("average"),
            "positive_count": payload.get("positive_count"),
            "sample_count": payload.get("sample_count"),
            "roc": _compact_curve_payload(payload.get("roc")),
            "pr": _compact_curve_payload(payload.get("pr")),
            "confusion_matrix": payload.get("confusion_matrix"),
            "score_distribution": {
                key: (payload.get("score_distribution") or {}).get(key)
                for key in [
                    "available",
                    "reason",
                    "bin_count",
                    "score_min",
                    "score_max",
                    "positive",
                    "negative",
                ]
                if key in (payload.get("score_distribution") or {})
            },
            "threshold_sweep": {
                key: (payload.get("threshold_sweep") or {}).get(key)
                for key in ["available", "reason", "selected_threshold", "threshold_source"]
                if key in (payload.get("threshold_sweep") or {})
            },
        }

    def compact_importance(payload):
        if not isinstance(payload, dict):
            return payload
        items = list(payload.get("items") or [])[:20]
        return {**{key: value for key, value in payload.items() if key != "items"}, "items": items}

    return {
        "schema_version": analysis.get("schema_version"),
        "curves": curves,
        "projection_2d": {
            key: (analysis.get("projection_2d") or {}).get(key)
            for key in ["available", "reason", "method"]
            if key in (analysis.get("projection_2d") or {})
        },
        "projection_3d": {
            key: (analysis.get("projection_3d") or {}).get(key)
            for key in ["available", "reason", "method"]
            if key in (analysis.get("projection_3d") or {})
        },
        "clusters": analysis.get("clusters"),
        "feature_importance": compact_importance(analysis.get("feature_importance")),
        "split_importance": {
            split: compact_importance(payload)
            for split, payload in (analysis.get("split_importance") or {}).items()
        },
        "shap_importance": {
            split: compact_importance(payload)
            for split, payload in (analysis.get("shap_importance") or {}).items()
        },
        "warnings": analysis.get("warnings") or [],
    }


def _compact_model(model):
    if not model:
        return None
    metadata = dict(model.get("metadata") or {})
    args = metadata.get("args")
    if isinstance(args, dict):
        metadata["args"] = {
            key: value
            for key, value in args.items()
            if key
            in {
                "task_type",
                "tokenizer",
                "encoder",
                "model",
                "classifier",
                "sequence_model",
                "training_mode",
                "residue_adapter_dim",
                "pair_strategy",
                "interaction_level",
                "interaction_model",
                "predictor_type",
                "graph_predictor",
                "fusion",
                "max_length",
                "epochs",
                "batch_size",
                "ml_auto_tune",
                "ml_tuning_objective",
            }
        }
    return {
        "id": model.get("id"),
        "name": model.get("name"),
        "task_type": model.get("task_type"),
        "supports_inference": model.get("supports_inference"),
        "created_at": model.get("created_at"),
        "metadata": metadata,
    }


def _score_stats(values):
    numeric = []
    for value in values:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    if not numeric:
        return None
    return {
        "count": len(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "mean": round(sum(numeric) / len(numeric), 6),
    }


def _compact_inference_runs(runs):
    compact = []
    for run in list(runs or [])[:5]:
        response = run.get("response") or {}
        scores = []
        for prediction in response.get("predictions") or []:
            if isinstance(prediction, dict):
                if "score" in prediction:
                    scores.append(prediction.get("score"))
                scores.extend(prediction.get("scores") or [])
        compact.append(
            {
                "id": run.get("id"),
                "input_kind": run.get("input_kind"),
                "input_count": run.get("input_count"),
                "prediction_count": run.get("prediction_count"),
                "labels": response.get("labels") or [],
                "score_stats": _score_stats(scores),
                "created_at": run.get("created_at"),
            }
        )
    return compact


def _model_from_experiment(store, experiment):
    model_id = experiment.get("model_id")
    if model_id:
        model = store.get_model(model_id)
        if model:
            return model
    models = store.list_models_for_experiment(experiment["id"])
    return models[0] if models else None


def _redacted_experiment(experiment):
    return {
        "id": experiment.get("id"),
        "name": experiment.get("name"),
        "task_type": experiment.get("task_type"),
        "status": experiment.get("status"),
        "config": experiment.get("config") or {},
        "metrics": {
            "summary": (experiment.get("metrics") or {}).get("summary") or {},
            "recent_events": _compact_metric_events(
                (experiment.get("metrics") or {}).get("events") or []
            ),
        },
        "error": experiment.get("error") or "",
        "created_at": experiment.get("created_at"),
        "started_at": experiment.get("started_at"),
        "finished_at": experiment.get("finished_at"),
    }


def build_experiment_context(store, experiment, dataset, include=None):
    include = include or {}
    include_dataset_preview = bool(include.get("dataset_preview"))
    include_inference_summary = bool(include.get("inference_summary"))
    include_extended_logs = bool(include.get("extended_logs"))
    log_limit = 12_000 if include_extended_logs else 4_000
    model = _model_from_experiment(store, experiment)
    log_path = Path(experiment.get("log_path") or "")
    log_tail = ""
    if log_path.is_file():
        log_tail = _redact_sequence_like_text(
            _trim_text(log_path.read_text(encoding="utf-8", errors="replace"), log_limit)
        )
    try:
        analysis = load_analysis(experiment["artifact_dir"])
    except (OSError, json.JSONDecodeError, ValueError):
        analysis = empty_analysis()
    context = {
        "experiment": _redacted_experiment(experiment),
        "dataset": {
            "id": dataset.get("id"),
            "name": dataset.get("name"),
            "task_type": dataset.get("task_type"),
            "source": dataset.get("source"),
            "format": dataset.get("format"),
            "report": _compact_dataset_report(dataset, include_dataset_preview),
        },
        "analysis": _compact_analysis_payload(analysis),
        "model": _compact_model(model),
        "log_tail": log_tail,
    }
    if include_inference_summary and model:
        context["inference_runs"] = _compact_inference_runs(
            store.list_inference_runs(model["id"], limit=5)
        )

    manifest = {
        "sections": [
            {
                "id": "experiment",
                "sent": True,
                "detail": "configuration, status, metric summary, recent metric events",
            },
            {
                "id": "dataset",
                "sent": True,
                "detail": "dataset report with preview"
                if include_dataset_preview
                else "dataset report without sample preview",
            },
            {
                "id": "analysis",
                "sent": True,
                "detail": "analysis artifact summaries without curve or projection point arrays",
            },
            {"id": "model", "sent": bool(model), "detail": "model metadata summary"},
            {
                "id": "logs",
                "sent": bool(log_tail),
                "detail": f"last {log_limit} characters" if log_tail else "no readable log file",
            },
            {
                "id": "inference_runs",
                "sent": bool(include_inference_summary and model),
                "detail": "recent inference aggregate summaries only",
            },
        ],
        "include": {
            "dataset_preview": include_dataset_preview,
            "inference_summary": include_inference_summary,
            "extended_logs": include_extended_logs,
        },
        "redactions": [
            "API keys are never sent",
            "full logs are not sent",
            "raw prediction rows are not sent",
            "dataset sample preview is sent only when explicitly enabled",
        ],
    }
    return context, manifest


def build_analysis_messages(context, manifest, user_prompt, language):
    context_json = json.dumps(context, ensure_ascii=False, sort_keys=True)
    max_context_chars = max(8_000, _env_int("PEP_LLM_MAX_CONTEXT_CHARS", 60_000))
    warnings = []
    if len(context_json) > max_context_chars:
        warnings.append("context was truncated to fit the configured maximum size")
        context_json = context_json[:max_context_chars]
    manifest["context_chars"] = len(context_json)
    language_hint = "Chinese" if str(language or "").lower().startswith("zh") else "English"
    system_prompt = (
        "You are the PepPlatform experiment analysis assistant. "
        "Use only the provided JSON context. Be precise, concise, and actionable. "
        "Do not invent metrics, datasets, model behavior, or unseen raw data. "
        "If evidence is missing, say what is missing. "
        "Focus on experimental interpretation, likely failure modes, and next experiment suggestions."
    )
    user_message = (
        f"Answer in {language_hint} unless the user's prompt strongly requests another language.\n"
        f"User question:\n{str(user_prompt or '').strip()}\n\n"
        f"Context manifest:\n{json.dumps(manifest, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Experiment context JSON:\n{context_json}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ], warnings


def markdown_to_plain_summary(markdown, limit=160):
    text = re.sub(r"\s+", " ", re.sub(r"[#*_>`\[\]\(\)]", " ", str(markdown or ""))).strip()
    return text[:limit]
