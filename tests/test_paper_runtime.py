"""Exercise CUDA/CPU profile validation without requiring CUDA hardware."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_runtime", ROOT / "docker/check_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def observed(channel, device=None, usable=None):
    profile = runtime.PROFILES[channel]
    return {
        "python_version": runtime.PYTHON_VERSION,
        "cuda_runtime_version": profile["cuda_runtime_version"],
        "cudnn_version": profile["cudnn_version"],
        "packages": {
            **runtime.CORE_PACKAGES,
            "torch": profile["torch"],
            "torchvision": profile["torchvision"],
        },
        "requested_device": device,
        "device_usable": usable,
    }


@pytest.mark.parametrize("channel", ["cpu", "cu121"])
def test_image_build_validation_does_not_claim_hardware_was_tested(channel):
    report = observed(channel)
    assert runtime.mismatches(report, channel) == {}
    assert report["requested_device"] is None
    assert report["device_usable"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [("python_version", "3.10.20"), ("cuda_runtime_version", "12.4"), ("cudnn_version", 8900)],
)
def test_runtime_drift_is_rejected(field, value):
    report = observed("cu121")
    report[field] = value
    assert field in runtime.mismatches(report, "cu121")


@pytest.mark.parametrize(
    ("name", "value"),
    [("torch", "2.5.1+cpu"), ("torchvision", "0.20.1+cu124"), ("sentencepiece", "0.2.2")],
)
def test_package_drift_cannot_pass_as_the_paper_runtime(name, value):
    report = observed("cu121")
    report["packages"][name] = value
    assert name in runtime.mismatches(report, "cu121")


def test_missing_gpu_is_a_failure_and_cpu_is_an_explicit_alternative():
    assert "device" in runtime.mismatches(observed("cu121", "cuda", False), "cu121")
    assert runtime.mismatches(observed("cu121", "cuda", True), "cu121") == {}
    assert runtime.mismatches(observed("cpu", "cpu", True), "cpu") == {}
    assert "device" in runtime.mismatches(observed("cpu", "cuda", True), "cpu")


def test_compiled_cuda_profile_can_be_checked_without_initializing_a_gpu(monkeypatch):
    expected = observed("cu121")
    monkeypatch.setattr(runtime.platform, "python_version", lambda: runtime.PYTHON_VERSION)
    monkeypatch.setattr(runtime.importlib.metadata, "version", expected["packages"].__getitem__)
    fake = SimpleNamespace(
        version=SimpleNamespace(cuda="12.1"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 90100)),
        cuda=SimpleNamespace(is_available=lambda: pytest.fail("Build must not probe GPU hardware")),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    report = runtime.inspect_runtime("cu121")
    assert report["status"] == "PASS"
    assert report["device_usable"] is None


def test_unavailable_cuda_never_runs_a_cpu_probe(monkeypatch):
    expected = observed("cu121")
    monkeypatch.setattr(runtime.platform, "python_version", lambda: runtime.PYTHON_VERSION)
    monkeypatch.setattr(runtime.importlib.metadata, "version", expected["packages"].__getitem__)
    fake = SimpleNamespace(
        version=SimpleNamespace(cuda="12.1"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 90100)),
        cuda=SimpleNamespace(is_available=lambda: False),
        ones=lambda *a, **k: pytest.fail("No fallback tensor allocation allowed"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    report = runtime.inspect_runtime("cu121", "cuda")
    assert report["status"] == "FAIL"
    assert report["device_usable"] is False
    assert "Container Toolkit" in report["device_error"]


def test_actual_smoke_check_rejects_missing_cuda_before_loading_weights(tmp_path, monkeypatch):
    import torch
    from test_docker_models import models

    monkeypatch.setenv("PEP_TORCH_CHANNEL", "cu121")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(models.ModelError, match="Runtime/device validation failed"):
        models.smoke_test(tmp_path / "no-models", models.load_lock(), device="cuda")


def test_device_option_requires_a_model_smoke_check():
    from test_docker_models import models

    with pytest.raises(SystemExit) as result:
        models.main(["--check-only", "--device", "cuda"])
    assert result.value.code == 2


def test_gpu_rng_report_includes_selected_cuda_state_without_touching_other_devices(monkeypatch):
    import torch

    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    from verify_prott5_equivalence import tensor_sha
    from verify_prott5_loader_rng import rng_state

    value = torch.tensor([1, 2, 3], dtype=torch.uint8)
    calls = []

    def selected(device):
        calls.append(str(device))
        return value

    monkeypatch.setattr(torch.cuda, "get_rng_state", selected)
    report = rng_state("cuda:0")
    assert report["torch_cuda"] == tensor_sha(value)
    assert calls == ["cuda:0"]
    assert "torch_cuda" not in rng_state("cpu")


def test_cpu_override_removes_gpu_requests_without_changing_model_storage():
    if not shutil.which("docker"):
        pytest.skip("Docker Compose is needed to validate YAML merge semantics")
    version = subprocess.run(["docker", "compose", "version"], capture_output=True, check=False)
    if version.returncode:
        pytest.skip("Docker Compose is unavailable")
    environment = dict(
        os.environ,
        POSTGRES_PASSWORD="test-database-value",
        PEP_SESSION_SECRET="test-session-value",
        PEP_LLM_KEY_ENCRYPTION_SECRET="test-encryption-value",
        HF_TOKEN="",
        PEP_PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple",
        PEP_TORCH_INDEX_BASE_URL="https://mirrors.nju.edu.cn/pytorch/whl",
        PEP_PIP_TIMEOUT="120",
    )
    command = [
        "docker",
        "compose",
        "--env-file",
        "/dev/null",
        "-p",
        "paper-runtime-test",
        "-f",
        str(ROOT / "docker-compose.yml"),
    ]

    def render(extra):
        process = subprocess.run(
            [*command, *extra, "config", "--format", "json"],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(process.stdout)["services"]["backend"]

    gpu = render([])
    cpu = render(["-f", str(ROOT / "docker-compose.cpu.yml")])
    assert gpu["build"]["args"]["TORCH_CHANNEL"] == "cu121"
    assert gpu["gpus"]
    assert cpu["build"]["args"]["TORCH_CHANNEL"] == "cpu"
    assert not cpu.get("gpus")
    for backend in [gpu, cpu]:
        assert backend["build"]["args"]["PIP_INDEX_URL"] == environment["PEP_PIP_INDEX_URL"]
        assert (
            backend["build"]["args"]["TORCH_INDEX_BASE_URL"]
            == environment["PEP_TORCH_INDEX_BASE_URL"]
        )
        assert backend["build"]["args"]["PIP_DEFAULT_TIMEOUT"] == "120"
    assert gpu["volumes"] == cpu["volumes"]
    assert gpu["environment"]["HF_HOME"] == cpu["environment"]["HF_HOME"]


def test_prott5_cuda_audit_rejects_missing_gpu_before_reading_models(tmp_path, monkeypatch):
    import torch

    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    import verify_prott5_restore as audit

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        with pytest.raises(RuntimeError, match="cannot fall back to CPU"):
            audit.main(
                [
                    "--official",
                    str(tmp_path / "official"),
                    "--sharded",
                    str(tmp_path / "sharded"),
                    "--restored",
                    str(tmp_path / "restored"),
                    "--reference",
                    str(tmp_path / "reference.json"),
                    "--output",
                    str(tmp_path / "audit"),
                    "--device",
                    "cuda",
                ]
            )
    finally:
        torch.use_deterministic_algorithms(original_deterministic)
    report = json.loads((tmp_path / "audit/report.json").read_text())
    assert report["status"] == "ERROR"
    assert "CUDA unavailable" in report["error"]
