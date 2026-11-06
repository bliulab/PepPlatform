"""Verify restored paper files, loading RNG, features and three synthetic CNN steps.

Prepare the official checkpoint with docker/download_models.py first. Run this
audit in the frozen paper environment with the original registered assets present.
All source model directories are read-only; reports require a new directory.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import random
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from verify_prott5_equivalence import (
    ROOT,
    compare_outputs,
    file_sha,
    forward_source,
    model_manager,
    read_json,
    registered_cases,
    tensor_sha,
    write_json,
)
from verify_prott5_loader_rng import rng_state


def markdown(report):
    return "\n".join(
        [
            "# 官方 ProtT5 自动还原验收",
            "",
            "结果：**" + report["status"] + "**",
            "",
            "- 文件完整匹配原冻结清单：" + str(report.get("files_match_lock")),
            "- 原训练产物身份校验通过：" + str(report.get("identity_validation_passed")),
            "- 加载后的编码器状态一致：" + str(report.get("loaded_encoder_state_equal")),
            "- 所有阶段随机数状态一致：" + str(report.get("rng_equal")),
            "- CNN 初始权重一致：" + str(report.get("classifier_initialization_equal")),
            "- CNN 三步合成数据训练一致：" + str(report.get("synthetic_training_equal")),
            "- 63 条序列特征输出逐位一致：" + str(report.get("outputs_equal")),
            "",
            "本次只验证模型准备和短训练一致性，未重训全部候选。原冻结资源均保持只读。",
            "",
            "## 复跑",
            "",
            "```bash",
            report.get("rerun_command", ""),
            "```",
            "",
            *(["错误：" + report["error"], ""] if "error" in report else []),
        ]
    )


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Exact restoration, offline loading, RNG, features and three synthetic CNN steps",
        "official_directory": str(args.official.resolve()),
        "original_directory": str(args.sharded.resolve()),
        "restored_directory": str(args.restored.resolve()),
    }
    command = [sys.executable, str(Path(__file__).resolve())]
    for name in ["official", "sharded", "restored", "reference"]:
        command.extend(["--" + name, str(getattr(args, name).resolve())])
    command.extend(
        [
            "--output",
            str(output.with_name(output.name + "-rerun")),
            "--threads",
            str(args.threads),
            "--device",
            args.device,
        ]
    )
    report["rerun_command"] = shlex.join(command)

    def progress(message):
        print(message, flush=True)
        write_json(output / "progress.json", {"stage": message})

    try:
        os.environ.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            PEPPLATFORM_OFFLINE_MODE="1",
            TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS=str(args.threads),
            MKL_NUM_THREADS=str(args.threads),
            OPENBLAS_NUM_THREADS=str(args.threads),
        )
        if args.device == "cuda":
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        sys.path.insert(0, str(ROOT))
        source_names = [
            "tools/verify_prott5_restore.py",
            "tools/verify_prott5_equivalence.py",
            "tools/verify_prott5_loader_rng.py",
            "docker/download_models.py",
            "docker/prott5.restore.json",
            "docker/models.lock.json",
            "pepplatform/encoders.py",
            "pepplatform/tokenizers.py",
            "pepplatform/prott5.py",
            "pepplatform/models.py",
            "pepplatform/workflow.py",
            "pepplatform/cache_provenance.py",
        ]
        report["source_files"] = {name: file_sha(ROOT / name) for name in source_names}
        for name in source_names:
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        import numpy as np
        import torch

        from pepplatform.models import PretrainedResiduePredictor, ResidueModelInput
        from pepplatform.prott5 import validate_prott5_model_identity
        from pepplatform.workflow import parse_prediction_head_hidden_dims

        torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; GPU verification cannot fall back to CPU")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        report["environment"] = {
            p: importlib.metadata.version(p)
            for p in ["torch", "transformers", "tokenizers", "sentencepiece", "safetensors"]
        }
        manager = model_manager()
        lock = manager.load_lock()
        progress("Auditing frozen reference and exact original/restored file hashes")
        manager.reference_sources(args.reference, lock)
        model = next(m for m in lock["models"] if m["key"] == "prott5")
        recipe = manager.load_prott5_recipe(model)
        manager.verify_source_files(args.official, recipe["source"]["files"])
        manager.verify_tree(args.sharded, model)
        manager.verify_tree(args.restored, model)
        report.update(
            files_match_lock=True,
            reference=lock["reference"],
            official_revision=recipe["source"]["revision"],
            official_sha256=recipe["source"]["files"]["pytorch_model.bin"]["sha256"],
            recipe_sha256=manager.PROTT5_RECIPE_SHA256,
            lock_sha256=file_sha(manager.LOCK_PATH),
            weight_sha256=model["weight_sha256"],
            model_fingerprint=model["model_fingerprint"],
            verified_files=model["files"],
        )
        cases = registered_cases(args.reference, lock, 8)
        write_json(output / "cases.json", cases)
        report["case_manifest_sha256"] = file_sha(output / "cases.json")
        registration = read_json(args.reference)
        index = read_json(registration["results_index"]["path"])
        row = next(
            r
            for r in index["candidates"]
            if r["task"] == "pep_function" and r["candidate"] == "c001_prott5_residue_cnn_mlp"
        )
        metadata_path = Path(row["artifact"]) / "metadata.json"
        if file_sha(metadata_path) != row["metadata_sha256"]:
            raise ValueError("Registered classifier metadata changed")
        metadata = read_json(metadata_path)
        original = metadata["args"]
        seed = int(original["seed"])
        head_kwargs = {
            "architecture": original["sequence_model"] or original["model"],
            "input_dim": 1024,
            "adapter_dim": original["residue_adapter_dim"],
            "max_length": original["max_length"],
            "num_classes": len(metadata["label_names"]),
            "dropout": original["dropout"],
            "head_hidden_dims": parse_prediction_head_hidden_dims(original["head_hidden_dims"]),
            "head_activation": original["head_activation"],
            "head_dropout": original["head_dropout"],
        }
        report["classifier_probe"] = {
            "candidate": row["candidate"],
            "metadata_sha256": row["metadata_sha256"],
            "constructor_arguments": head_kwargs,
            "synthetic_training": {
                "steps": 3,
                "optimizer": "AdamW",
                "lr": 0.001,
                "weight_decay": 0,
            },
        }
        report["execution"] = {
            "device": args.device,
            "threads": args.threads,
            "seed": seed,
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
        if args.device == "cuda":
            report["execution"]["gpu_name"] = torch.cuda.get_device_name(0)
        generator = torch.Generator().manual_seed(20261106)
        length = int(original["max_length"])
        features = torch.randn(2, length, 1024, generator=generator)
        mask = torch.zeros(2, length, dtype=torch.bool)
        mask[0, :8], mask[1, :13] = True, True
        features *= mask.unsqueeze(-1)
        batch = ResidueModelInput(features.to(args.device), mask.to(args.device))
        labels = torch.zeros(2, len(metadata["label_names"]))
        labels[0, 0], labels[1, 1] = 1, 1
        labels = labels.to(args.device)

        def head_state(head):
            return {name: tensor_sha(tensor) for name, tensor in head.state_dict().items()}

        results = {}
        for role, source in [("original", args.sharded), ("restored", args.restored)]:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            states = {"before_loading": rng_state(args.device)}

            def loaded(_encoder, states=states):
                states["after_loading"] = rng_state(args.device)

            result = forward_source(
                source, role, cases, output, progress, on_encoder_loaded=loaded, device=args.device
            )
            states["after_features"] = rng_state(args.device)
            head = PretrainedResiduePredictor(**head_kwargs).to(args.device)
            initial = head_state(head)
            states["after_classifier_initialization"] = rng_state(args.device)
            head.train()
            optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0)
            criterion = torch.nn.BCEWithLogitsLoss()
            losses = []
            for _step in range(3):
                optimizer.zero_grad()
                loss = criterion(head(batch), labels)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite synthetic training loss")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            states["after_synthetic_training"] = rng_state(args.device)
            result.update(
                rng=states,
                classifier_initial_state=initial,
                synthetic_losses=losses,
                classifier_trained_state=head_state(head),
            )
            results[role] = result
            write_json(output / (role + ".json"), result)
            del head, optimizer
            if args.device == "cuda":
                torch.cuda.empty_cache()
        left, right = results["original"], results["restored"]
        validate_prott5_model_identity(left["model_identity"], right["model_identity"])
        report["identity_validation_passed"] = True
        report["loaded_encoder_state_equal"] = left["loaded_state"] == right["loaded_state"]
        report["loader_class_equal"] = (
            left["loader_class"]
            == right["loader_class"]
            == "transformers.models.t5.modeling_t5.T5EncoderModel"
        )
        report["rng_equal"] = left["rng"] == right["rng"]
        report["classifier_initialization_equal"] = (
            left["classifier_initial_state"] == right["classifier_initial_state"]
        )
        report["synthetic_training_equal"] = (
            left["synthetic_losses"] == right["synthetic_losses"]
            and left["classifier_trained_state"] == right["classifier_trained_state"]
        )
        report["synthetic_losses"] = {
            role: result["synthetic_losses"] for role, result in results.items()
        }
        report["rng_states"] = {role: result["rng"] for role, result in results.items()}
        report["output_cases"] = []
        for case in cases["probes"]:
            name = case["name"]
            original_outputs = torch.load(output / "original" / (name + ".pt"), weights_only=True)
            restored_outputs = torch.load(output / "restored" / (name + ".pt"), weights_only=True)
            report["output_cases"].append(
                {
                    "name": name,
                    "sample_count": len(case["sequences"]),
                    **compare_outputs(original_outputs, restored_outputs),
                }
            )
        report["outputs_equal"] = (
            all(c["bitwise_equal"] for c in report["output_cases"])
            and left["rejected_inputs"] == right["rejected_inputs"]
        )
        report["status"] = (
            "PASS"
            if all(
                report[key]
                for key in [
                    "files_match_lock",
                    "identity_validation_passed",
                    "loaded_encoder_state_equal",
                    "loader_class_equal",
                    "rng_equal",
                    "classifier_initialization_equal",
                    "synthetic_training_equal",
                    "outputs_equal",
                ]
            )
            else "DIFFERENT"
        )
        if any(file_sha(ROOT / name) != digest for name, digest in report["source_files"].items()):
            raise ValueError("Verification source changed during execution")
    except Exception as exc:
        report.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "report.json", report)
        (output / "REPORT.md").write_text(markdown(report), encoding="utf-8")
    progress("Verification complete: " + report["status"])
    return 0 if report["status"] == "PASS" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["official", "sharded", "restored", "reference", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("Thread count must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
