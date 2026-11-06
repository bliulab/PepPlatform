"""Check the paper's core training packages and the selected Torch distribution.

Image builds check the compiled runtime without requiring a physical GPU. Pass
--device cuda at runtime to require and actually exercise an exposed CUDA device.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform

PYTHON_VERSION = "3.10.19"
CORE_PACKAGES = {
    "joblib": "1.5.3",
    "gensim": "4.4.0",
    "numpy": "2.2.6",
    "scipy": "1.15.3",
    "scikit-learn": "1.7.2",
    "xgboost": "3.1.3",
    "lightgbm": "4.6.0",
    "tqdm": "4.67.1",
    "transformers": "4.48.1",
    "tokenizers": "0.21.4",
    "esm": "3.2.0",
    "sentencepiece": "0.2.1",
    "huggingface-hub": "0.36.0",
    "safetensors": "0.7.0",
}
PROFILES = {
    "cu121": {
        "torch": "2.5.1+cu121",
        "torchvision": "0.20.1+cu121",
        "cuda_runtime_version": "12.1",
        "cudnn_version": 90100,
    },
    "cpu": {
        "torch": "2.5.1+cpu",
        "torchvision": "0.20.1+cpu",
        "cuda_runtime_version": None,
        "cudnn_version": None,
    },
}


def mismatches(report, channel):
    """Compare observed facts; a missing device check is distinct from success."""
    expected = PROFILES[channel]
    differences = {}
    for key, wanted in {
        "python_version": PYTHON_VERSION,
        "cuda_runtime_version": expected["cuda_runtime_version"],
        "cudnn_version": expected["cudnn_version"],
    }.items():
        if report.get(key) != wanted:
            differences[key] = {"expected": wanted, "actual": report.get(key)}
    for name, wanted in {
        **CORE_PACKAGES,
        "torch": expected["torch"],
        "torchvision": expected["torchvision"],
    }.items():
        actual = report["packages"].get(name)
        if actual != wanted:
            differences[name] = {"expected": wanted, "actual": actual}
    if report.get("requested_device") == "cuda":
        if channel != "cu121" or not report.get("device_usable"):
            differences["device"] = {
                "expected": "usable CUDA 12.1 device",
                "actual": report.get("device_error") or "CUDA unavailable",
            }
    elif report.get("requested_device") == "cpu" and not report.get("device_usable"):
        differences["device"] = {"expected": "usable CPU", "actual": report.get("device_error")}
    return differences


def inspect_runtime(channel, device=None):
    import torch

    names = [*CORE_PACKAGES, "torch", "torchvision"]
    packages = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "channel": channel,
        "python_version": platform.python_version(),
        "packages": packages,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "requested_device": device,
        "device_usable": None,
    }
    if device is not None:
        try:
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA unavailable; check the NVIDIA driver and Container Toolkit"
                )
            probe = torch.ones(1, device=device) + 1
            if float(probe.cpu()[0]) != 2:
                raise RuntimeError("Device arithmetic probe failed")
            report["device_usable"] = True
            if device == "cuda":
                report["gpu_name"] = torch.cuda.get_device_name(0)
            del probe
        except RuntimeError as exc:
            report.update(device_usable=False, device_error=str(exc))
    report["mismatches"] = mismatches(report, channel)
    report["status"] = "PASS" if not report["mismatches"] else "FAIL"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", choices=PROFILES, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args(argv)
    report = inspect_runtime(args.channel, args.device)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
