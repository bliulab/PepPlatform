import json
import urllib.error

import pytest

fastapi = pytest.importorskip("fastapi")

from pepplatform.web.app import create_app
from pepplatform.web.ai import (
    AiChatClient,
    AiProviderError,
    AiStreamInterruptedError,
    decrypt_api_key,
    normalize_ai_provider_payload,
    provider_presets,
)


class _Response:
    def __init__(self):
        self.cookies = {}

    def set_cookie(self, key, value, **_kwargs):
        self.cookies[key] = value

    def delete_cookie(self, key, **_kwargs):
        self.cookies.pop(key, None)


class _Request:
    def __init__(self, cookies=None):
        self.cookies = cookies or {}
        self.headers = {}
        self.client = None


class _FakeAiClient:
    def __init__(self):
        self.calls = []

    def chat(self, provider_config, api_key, messages, max_tokens=1200, temperature=0.2):
        self.calls.append(
            {
                "provider_config": dict(provider_config),
                "api_key": api_key,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return {
            "answer": "AI summary",
            "provider": provider_config["provider"],
            "model": provider_config["model"],
            "usage": {"total_tokens": 42},
        }


def _endpoint(app, path, method):
    for route in app.routes:
        if getattr(route, "path", "") == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"{method} {path} endpoint was not registered")


def _auth_app(tmp_path, monkeypatch, encryption_secret="test-llm-encryption-secret"):
    monkeypatch.setenv("PEP_AUTH_ENABLED", "1")
    monkeypatch.setenv("PEP_SESSION_SECRET", "test-session-secret")
    if encryption_secret is None:
        monkeypatch.delenv("PEP_LLM_KEY_ENCRYPTION_SECRET", raising=False)
    else:
        monkeypatch.setenv("PEP_LLM_KEY_ENCRYPTION_SECRET", encryption_secret)
    project_root = tmp_path / "project"
    project_root.mkdir()
    app = create_app(project_root=project_root, runtime_root=tmp_path / "runtime")
    app.state.ai_client = _FakeAiClient()
    return app


def _register_verified(app, email):
    register = _endpoint(app, "/api/auth/register", "POST")
    verify = _endpoint(app, "/api/auth/verify-email", "POST")
    response = register({"email": email, "password": "password123"})
    verify_response = _Response()
    verified = verify({"email": email, "code": response["debug_code"]}, verify_response)
    return verified["user"], verify_response.cookies


def _provider_payload(api_key="sk-test-123456"):
    return {
        "provider": "deepseek",
        "display_name": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "api_key": api_key,
        "enabled": True,
    }


def _create_provider(app, cookies, api_key="sk-test-123456"):
    create_provider = _endpoint(app, "/api/ai/providers", "POST")
    return create_provider(_provider_payload(api_key), _Request(cookies))


def test_ai_provider_crud_encrypts_and_never_returns_api_key(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch)
    try:
        _user, cookies = _register_verified(app, "ai@example.com")
        list_providers = _endpoint(app, "/api/ai/providers", "GET")
        update_provider = _endpoint(app, "/api/ai/providers/{provider_id}", "PATCH")
        test_provider = _endpoint(app, "/api/ai/providers/{provider_id}/test", "POST")
        delete_provider = _endpoint(app, "/api/ai/providers/{provider_id}", "DELETE")

        provider = _create_provider(app, cookies)
        assert provider["api_key_last4"] == "3456"
        assert provider["has_api_key"] is True
        assert "api_key" not in provider
        assert "api_key_ciphertext" not in provider

        stored = app.state.store.get_ai_provider_config(provider["id"], include_secret=True)
        assert stored["api_key_ciphertext"] != "sk-test-123456"
        assert decrypt_api_key(stored["api_key_ciphertext"]) == "sk-test-123456"

        providers = list_providers(_Request(cookies))
        assert providers[0]["id"] == provider["id"]
        assert "api_key_ciphertext" not in providers[0]

        updated = update_provider(
            provider["id"],
            {"display_name": "DeepSeek lab", "api_key": "sk-rotated-9999", "enabled": False},
            _Request(cookies),
        )
        assert updated["display_name"] == "DeepSeek lab"
        assert updated["api_key_last4"] == "9999"
        assert updated["enabled"] is False

        with pytest.raises(fastapi.HTTPException) as exc_info:
            test_provider(provider["id"], _Request(cookies))
        assert exc_info.value.status_code == 400

        deleted = delete_provider(provider["id"], _Request(cookies))
        assert deleted["deleted"] is True
        assert list_providers(_Request(cookies)) == []
    finally:
        app.state.runner.stop()


def test_openai_compatible_provider_persists_capabilities(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch)
    try:
        _user, cookies = _register_verified(app, "openai-compatible@example.com")
        create_provider = _endpoint(app, "/api/ai/providers", "POST")

        provider = create_provider(
            {
                "provider": "openai_compatible",
                "display_name": "OpenAI compatible",
                "base_url": "https://llm.example.test/v1",
                "model": "custom-chat-model",
                "api_key": "sk-compatible-1234",
                "enabled": True,
                "capabilities": {
                    "supports_json_object": False,
                    "supports_reasoning_toggle": False,
                    "default_max_tokens": 1024,
                },
            },
            _Request(cookies),
        )

        assert provider["provider"] == "openai_compatible"
        assert provider["capabilities"]["supports_json_object"] is False
        assert provider["capabilities"]["default_max_tokens"] == 1024
        stored = app.state.store.get_ai_provider_config(provider["id"], include_secret=True)
        assert stored["capabilities"]["supports_reasoning_toggle"] is False
    finally:
        app.state.runner.stop()


def test_ai_provider_presets_endpoint_returns_public_defaults(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch)
    try:
        presets_endpoint = _endpoint(app, "/api/ai/provider-presets", "GET")
        presets = presets_endpoint()

        provider_ids = [preset["provider"] for preset in presets]
        assert provider_ids[:5] == [
            "deepseek",
            "qwen_dashscope",
            "kimi",
            "zhipu_glm",
            "xiaomi_mimo",
        ]
        assert {
            "minimax",
            "siliconflow",
            "openai",
            "google_gemini",
            "anthropic_claude",
            "openrouter",
            "together",
            "deepinfra",
        }.issubset(provider_ids)
        assert all("api_key" not in preset for preset in presets)
        deepseek = next(preset for preset in presets if preset["provider"] == "deepseek")
        assert deepseek["model"] == "deepseek-flash"
        qwen = next(preset for preset in presets if preset["provider"] == "qwen_dashscope")
        assert qwen["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        assert qwen["model"] == "qwen-plus"
        assert qwen["capabilities"]["supports_json_object"] is True
        gemini = next(preset for preset in presets if preset["provider"] == "google_gemini")
        assert gemini["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai"
        assert gemini["model"] == "gemini-3.6-flash"
        claude = next(preset for preset in presets if preset["provider"] == "anthropic_claude")
        assert claude["base_url"] == "https://api.anthropic.com/v1"
        assert claude["model"] == "claude-sonnet-4-6"
    finally:
        app.state.runner.stop()


@pytest.mark.parametrize(
    ("provider", "base_url", "model"),
    [
        ("qwen_dashscope", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
        ("kimi", "https://api.moonshot.ai/v1", "kimi-k2.6"),
        ("zhipu_glm", "https://open.bigmodel.cn/api/paas/v4", "GLM-4.7-Flash"),
        ("xiaomi_mimo", "https://api.xiaomimimo.com/v1", "mimo-v2.5-pro"),
        ("minimax", "https://api.minimax.io/v1", "MiniMax-M3"),
        ("siliconflow", "https://api.siliconflow.cn/v1", "Pro/zai-org/GLM-4.7"),
        ("openai", "https://api.openai.com/v1", "gpt-4o-mini"),
        (
            "google_gemini",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "gemini-3.6-flash",
        ),
        ("anthropic_claude", "https://api.anthropic.com/v1", "claude-sonnet-4-6"),
        ("openrouter", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
        ("together", "https://api.together.ai/v1", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        (
            "deepinfra",
            "https://api.deepinfra.com/v1/openai",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
        ),
    ],
)
def test_additional_ai_providers_normalize_to_provider_defaults(provider, base_url, model):
    normalized = normalize_ai_provider_payload(
        {
            "provider": provider,
            "api_key": "sk-test",
            "enabled": True,
        },
        require_api_key=True,
    )

    assert normalized["provider"] == provider
    assert normalized["display_name"] == next(
        preset["display_name"] for preset in provider_presets() if preset["provider"] == provider
    )
    assert normalized["base_url"] == base_url
    assert normalized["model"] == model
    assert normalized["capabilities"]["supports_json_object"] is True
    assert normalized["capabilities"]["reasoning_control"] == "none"


def test_ai_provider_save_requires_encryption_secret(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, encryption_secret=None)
    try:
        _user, cookies = _register_verified(app, "no-secret@example.com")
        create_provider = _endpoint(app, "/api/ai/providers", "POST")

        with pytest.raises(fastapi.HTTPException) as exc_info:
            create_provider(_provider_payload(), _Request(cookies))
        assert exc_info.value.status_code == 503
    finally:
        app.state.runner.stop()


def test_ai_provider_save_loads_encryption_secret_from_project_env(tmp_path, monkeypatch):
    monkeypatch.delenv("PEP_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("PEP_SESSION_SECRET", raising=False)
    monkeypatch.delenv("PEP_LLM_KEY_ENCRYPTION_SECRET", raising=False)
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / ".env").write_text(
        "\n".join(
            [
                "PEP_AUTH_ENABLED=1",
                "PEP_SESSION_SECRET=local-session-secret",
                "PEP_LLM_KEY_ENCRYPTION_SECRET=local-llm-encryption-secret",
            ]
        ),
        encoding="utf-8",
    )
    app = create_app(project_root=project_root, runtime_root=tmp_path / "runtime")
    try:
        _user, cookies = _register_verified(app, "env-secret@example.com")
        provider = _create_provider(app, cookies)
        assert provider["has_api_key"] is True
        assert provider["api_key_last4"] == "3456"
    finally:
        app.state.runner.stop()


def test_ai_provider_permissions_are_owner_scoped(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch)
    try:
        _owner, owner_cookies = _register_verified(app, "owner@example.com")
        _other, other_cookies = _register_verified(app, "other@example.com")
        provider = _create_provider(app, owner_cookies)
        get_provider_test = _endpoint(app, "/api/ai/providers/{provider_id}/test", "POST")

        assert provider["id"] not in [
            item["id"]
            for item in _endpoint(app, "/api/ai/providers", "GET")(_Request(other_cookies))
        ]
        with pytest.raises(fastapi.HTTPException) as exc_info:
            get_provider_test(provider["id"], _Request(other_cookies))
        assert exc_info.value.status_code == 404
    finally:
        app.state.runner.stop()


class _FakeUrlopenResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class _FakeStreamResponse:
    def __init__(self, chunks):
        self.chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self.chunks)


def test_deepseek_v4_requests_disable_thinking(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeUrlopenResponse(
            {"choices": [{"message": {"content": "OK"}}], "usage": {"total_tokens": 1}}
        )

    monkeypatch.setattr("pepplatform.web.ai.urllib.request.urlopen", fake_urlopen)
    response = AiChatClient(timeout_seconds=3).chat(
        {
            "provider": "deepseek",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
        },
        "sk-test",
        [{"role": "user", "content": "ping"}],
        max_tokens=32,
        temperature=0,
    )

    assert captured["timeout"] == 3
    assert captured["payload"]["thinking"] == {"type": "disabled"}
    assert response["answer"] == "OK"


def test_openai_compatible_request_respects_json_capability(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["url"] = request.full_url
        return _FakeUrlopenResponse({"choices": [{"message": {"content": '{"ok": true}'}}]})

    monkeypatch.setattr("pepplatform.web.ai.urllib.request.urlopen", fake_urlopen)
    AiChatClient(timeout_seconds=3).chat(
        {
            "provider": "openai_compatible",
            "base_url": "https://llm.example.test/v1",
            "model": "custom-chat-model",
            "capabilities": {"supports_json_object": False, "supports_reasoning_toggle": False},
        },
        "sk-test",
        [{"role": "user", "content": "ping"}],
        max_tokens=32,
        temperature=0,
        response_format={"type": "json_object"},
    )

    assert captured["url"] == "https://llm.example.test/v1/chat/completions"
    assert "response_format" not in captured["payload"]
    assert "thinking" not in captured["payload"]


@pytest.mark.parametrize(
    ("provider", "base_url", "model"),
    [
        ("kimi", "https://api.moonshot.ai/v1", "kimi-k2.6"),
        ("qwen_dashscope", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
        ("zhipu_glm", "https://open.bigmodel.cn/api/paas/v4", "GLM-4.7-Flash"),
        ("xiaomi_mimo", "https://api.xiaomimimo.com/v1", "mimo-v2.5-pro"),
        ("minimax", "https://api.minimax.io/v1", "MiniMax-M3"),
        ("siliconflow", "https://api.siliconflow.cn/v1", "Pro/zai-org/GLM-4.7"),
        (
            "google_gemini",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "gemini-3.6-flash",
        ),
        ("openrouter", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
        ("together", "https://api.together.ai/v1", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        (
            "deepinfra",
            "https://api.deepinfra.com/v1/openai",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
        ),
    ],
)
def test_openai_compatible_provider_family_uses_chat_completions(
    monkeypatch, provider, base_url, model
):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["url"] = request.full_url
        return _FakeUrlopenResponse({"choices": [{"message": {"content": '{"ok": true}'}}]})

    monkeypatch.setattr("pepplatform.web.ai.urllib.request.urlopen", fake_urlopen)
    AiChatClient(timeout_seconds=3).chat(
        {"provider": provider, "base_url": base_url, "model": model},
        "sk-test",
        [{"role": "user", "content": "ping"}],
        max_tokens=32,
        temperature=0,
        response_format={"type": "json_object"},
    )

    assert captured["url"] == f"{base_url}/chat/completions"
    assert captured["payload"]["model"] == model
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert "thinking" not in captured["payload"]


def test_anthropic_provider_uses_messages_api_and_hoists_system(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["url"] = request.full_url
        captured["api_key"] = request.get_header("X-api-key")
        captured["version"] = request.get_header("Anthropic-version")
        return _FakeUrlopenResponse(
            {
                "id": "msg-test",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": '{"ok": true}'}],
                "usage": {"input_tokens": 10, "output_tokens": 4},
            }
        )

    monkeypatch.setattr("pepplatform.web.ai.urllib.request.urlopen", fake_urlopen)
    response = AiChatClient(timeout_seconds=3).chat(
        {
            "provider": "anthropic_claude",
            "base_url": "https://api.anthropic.com/v1",
            "model": "claude-sonnet-4-6",
        },
        "sk-ant-test",
        [
            {"role": "system", "content": "You are a JSON bot."},
            {"role": "user", "content": "First instruction."},
            {"role": "user", "content": "Second instruction."},
        ],
        max_tokens=32,
        temperature=0,
        response_format={"type": "json_object"},
    )

    assert captured["timeout"] == 3
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["api_key"] == "sk-ant-test"
    assert captured["version"] == "2023-06-01"
    assert captured["payload"]["model"] == "claude-sonnet-4-6"
    assert captured["payload"]["system"] == "You are a JSON bot."
    assert captured["payload"]["messages"] == [
        {"role": "user", "content": "First instruction.\n\nSecond instruction."}
    ]
    assert captured["payload"]["output_config"]["format"]["type"] == "json_schema"
    assert response["answer"] == '{"ok": true}'
    assert response["usage"] == {"input_tokens": 10, "output_tokens": 4}


def test_anthropic_provider_retries_without_output_config_on_bad_request(monkeypatch):
    calls = []

    class _HttpError(urllib.error.HTTPError):
        def __init__(self):
            super().__init__("https://api.anthropic.com/v1/messages", 400, "bad request", {}, None)

        def read(self):
            return b'{"error":"unsupported output_config"}'

    def fake_urlopen(request, timeout):
        calls.append(json.loads(request.data.decode("utf-8")))
        if len(calls) == 1:
            raise _HttpError()
        return _FakeUrlopenResponse(
            {
                "id": "msg-retry",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": '{"ok": true}'}],
                "usage": {"input_tokens": 8, "output_tokens": 3},
            }
        )

    monkeypatch.setattr("pepplatform.web.ai.urllib.request.urlopen", fake_urlopen)
    response = AiChatClient(timeout_seconds=3).chat(
        {
            "provider": "anthropic_claude",
            "base_url": "https://api.anthropic.com/v1",
            "model": "claude-sonnet-4-6",
        },
        "sk-ant-test",
        [
            {"role": "system", "content": "Return strict JSON."},
            {"role": "user", "content": "ping"},
        ],
        max_tokens=32,
        temperature=0,
        response_format={"type": "json_object"},
    )

    assert len(calls) == 2
    assert "output_config" in calls[0]
    assert "output_config" not in calls[1]
    assert calls[1]["system"].endswith(
        "Return only a valid JSON object. Do not wrap it in Markdown."
    )
    assert response["answer"] == '{"ok": true}'


def test_openai_compatible_stream_parses_arbitrary_chunks(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeStreamResponse(
            [
                b'data: {"id":"chat-1","model":"model-a","choices":[{"delta":{"content":"Hel',
                b'lo"},"finish_reason":null}]}\n\n',
                b": keepalive\n\n",
                b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}],',
                b'"usage":{"total_tokens":7}}\n\ndata: [DONE]\n\n',
            ]
        )

    monkeypatch.setattr("pepplatform.web.ai.urllib.request.urlopen", fake_urlopen)
    events = list(
        AiChatClient(timeout_seconds=3).stream_chat(
            {"provider": "openai", "base_url": "https://api.openai.com/v1", "model": "model-a"},
            "sk-test",
            [{"role": "user", "content": "ping"}],
            max_tokens=32,
            temperature=0,
            response_format={"type": "json_object"},
        )
    )

    assert captured["payload"]["stream"] is True
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert "stream_options" not in captured["payload"]
    assert "".join(event.get("content", "") for event in events) == "Hello world"
    assert events[-1]["type"] == "done"
    assert events[-1]["model"] == "model-a"
    assert events[-1]["usage"] == {"total_tokens": 7}


def test_openai_compatible_stream_retries_transient_failure_before_first_delta(monkeypatch):
    calls = []

    def fake_urlopen(_request, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise BrokenPipeError(32, "Broken pipe")
        return _FakeStreamResponse(
            [b'data: {"choices":[{"delta":{"content":"Recovered"},"finish_reason":"stop"}]}\n\n']
        )

    monkeypatch.setattr("pepplatform.web.ai.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("pepplatform.web.ai.time.sleep", lambda _seconds: None)
    events = list(
        AiChatClient(timeout_seconds=3).stream_chat(
            {
                "provider": "deepseek",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-v4-flash",
            },
            "sk-test",
            [{"role": "user", "content": "ping"}],
            max_tokens=32,
            temperature=0,
        )
    )

    assert calls == [3, 3]
    assert "".join(event.get("content", "") for event in events) == "Recovered"
    assert events[-1]["type"] == "done"


def test_openai_compatible_stream_reports_interruption_after_retry_is_exhausted(monkeypatch):
    calls = []

    def fake_urlopen(_request, timeout):
        calls.append(timeout)
        raise urllib.error.URLError(ConnectionResetError(104, "connection reset"))

    monkeypatch.setattr("pepplatform.web.ai.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("pepplatform.web.ai.time.sleep", lambda _seconds: None)
    with pytest.raises(AiStreamInterruptedError) as exc_info:
        list(
            AiChatClient(timeout_seconds=3).stream_chat(
                {
                    "provider": "deepseek",
                    "base_url": "https://api.deepseek.com",
                    "model": "deepseek-v4-flash",
                },
                "sk-test",
                [{"role": "user", "content": "ping"}],
                max_tokens=32,
                temperature=0,
            )
        )

    assert calls == [3, 3]
    assert exc_info.value.partial is False
    assert exc_info.value.retries == 1


def test_openai_compatible_stream_preserves_partial_content_without_retry(monkeypatch):
    calls = []

    class _InterruptedResponse(_FakeStreamResponse):
        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"Partial"},"finish_reason":null}]}\n\n'
            raise BrokenPipeError(32, "Broken pipe")

    def fake_urlopen(_request, timeout):
        calls.append(timeout)
        return _InterruptedResponse([])

    monkeypatch.setattr("pepplatform.web.ai.urllib.request.urlopen", fake_urlopen)
    stream = AiChatClient(timeout_seconds=3).stream_chat(
        {
            "provider": "deepseek",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
        },
        "sk-test",
        [{"role": "user", "content": "ping"}],
        max_tokens=32,
        temperature=0,
    )

    assert next(stream) == {"type": "delta", "content": "Partial"}
    with pytest.raises(AiStreamInterruptedError) as exc_info:
        next(stream)
    assert calls == [3]
    assert exc_info.value.partial is True
    assert exc_info.value.retries == 0


def test_anthropic_stream_normalizes_text_and_usage(monkeypatch):
    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["stream"] is True
        return _FakeStreamResponse(
            [
                b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg-1","model":"claude-test","usage":{"input_tokens":5}}}\n\n',
                b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}\n\n',
                b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\n\n',
                b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
            ]
        )

    monkeypatch.setattr("pepplatform.web.ai.urllib.request.urlopen", fake_urlopen)
    events = list(
        AiChatClient(timeout_seconds=3).stream_chat(
            {
                "provider": "anthropic_claude",
                "base_url": "https://api.anthropic.com/v1",
                "model": "claude-test",
            },
            "sk-ant-test",
            [{"role": "user", "content": "ping"}],
            max_tokens=32,
            temperature=0,
        )
    )

    assert events[0] == {"type": "delta", "content": "Hi"}
    assert events[-1]["finish_reason"] == "end_turn"
    assert events[-1]["usage"] == {"input_tokens": 5, "output_tokens": 2}


def test_deepseek_reasoning_only_response_has_diagnostic_error(monkeypatch):
    def fake_urlopen(_request, timeout):
        return _FakeUrlopenResponse({"choices": [{"message": {"reasoning_content": "thinking"}}]})

    monkeypatch.setattr("pepplatform.web.ai.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(AiProviderError, match="reasoning content"):
        AiChatClient(timeout_seconds=3).chat(
            {
                "provider": "deepseek",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-v4-flash",
            },
            "sk-test",
            [{"role": "user", "content": "ping"}],
            max_tokens=32,
            temperature=0,
        )


def test_experiment_ai_analysis_uses_redacted_default_context(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch)
    try:
        user, cookies = _register_verified(app, "analysis@example.com")
        paths = app.state.paths
        dataset_root = paths.datasets / "demo"
        train_dir = dataset_root / "train"
        val_dir = dataset_root / "val"
        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)
        (train_dir / "seqs.fasta").write_text(">s1\nACDEFG\n", encoding="utf-8")
        (train_dir / "labels.csv").write_text("active\n1\n", encoding="utf-8")
        (val_dir / "seqs.fasta").write_text(">v1\nGGGGGG\n", encoding="utf-8")
        (val_dir / "labels.csv").write_text("active\n0\n", encoding="utf-8")
        dataset = app.state.store.create_dataset(
            "demo",
            "pep_function",
            "upload",
            "sequence_dir",
            dataset_root,
            owner_user_id=user["id"],
        )
        experiment_dir = paths.experiments / "ai-run"
        artifact_dir = experiment_dir / "artifact"
        artifact_dir.mkdir(parents=True)
        (experiment_dir / "train.log").write_text(
            "epoch one\nsecret sequence ACDEFGHIKLMNP should be redacted from logs\n",
            encoding="utf-8",
        )
        experiment = app.state.store.create_experiment(
            "ai run",
            dataset["id"],
            "pep_function",
            {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
            experiment_dir / "train.log",
            artifact_dir,
            experiment_id="ai-run",
            owner_user_id=user["id"],
        )
        app.state.store.update_experiment(
            experiment["id"],
            status="succeeded",
            metrics_json={
                "events": [{"kind": "test", "test_f1": 0.7}],
                "summary": {"test_f1": 0.7},
            },
        )
        provider = _create_provider(app, cookies)
        analyze = _endpoint(app, "/api/experiments/{experiment_id}/ai-analysis", "POST")

        response = analyze(
            experiment["id"],
            {
                "provider_config_id": provider["id"],
                "prompt": "Summarize",
                "language": "en",
                "include": {
                    "dataset_preview": False,
                    "inference_summary": False,
                    "extended_logs": False,
                },
            },
            _Request(cookies),
        )

        assert response["answer"] == "AI summary"
        assert response["model"] == "deepseek-v4-flash"
        assert response["context_manifest"]["include"]["dataset_preview"] is False
        assert app.state.ai_client.calls[-1]["api_key"] == "sk-test-123456"
        sent_message = app.state.ai_client.calls[-1]["messages"][-1]["content"]
        assert '"preview"' not in sent_message
        assert "ACDEFGHIKLMNP" not in sent_message
        assert "<redacted_sequence len=13>" in sent_message
        events = {item["event_type"] for item in app.state.store.list_audit_logs(limit=20)["items"]}
        assert "ai.experiment_analysis" in events
    finally:
        app.state.runner.stop()
