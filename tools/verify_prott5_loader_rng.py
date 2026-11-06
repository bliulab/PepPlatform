"""Audit ProtT5 loading RNG and a registered classifier's initialization on CPU.

This is a separate diagnostic from checkpoint and feature equivalence. It does
not train a classifier or change the platform's loading/seeding behavior.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import pickle
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from verify_prott5_equivalence import (
    ROOT,
    file_sha,
    model_manager,
    read_json,
    tensor_sha,
    write_json,
)


def rng_state(device="cpu"):
    import numpy as np
    import torch

    result = {
        "python": hashlib.sha256(pickle.dumps(random.getstate(), protocol=4)).hexdigest(),
        "numpy": hashlib.sha256(pickle.dumps(np.random.get_state(), protocol=4)).hexdigest(),
        "torch_cpu": tensor_sha(torch.get_rng_state()),
    }
    if torch.device(device).type == "cuda":
        result["torch_cuda"] = tensor_sha(torch.cuda.get_rng_state(torch.device(device)))
    return result


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PEPPLATFORM_OFFLINE_MODE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS=str(args.threads),
    )
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch

    from pepplatform.encoders import ProT5Encoder
    from pepplatform.models import PretrainedResiduePredictor
    from pepplatform.prott5 import PROTT5_OFFICIAL_WEIGHT_SHA256
    from pepplatform.tokenizers import ProT5Tokenizer
    from pepplatform.workflow import parse_prediction_head_hidden_dims

    report = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Loader RNG and registered Function CNN initialization; no training",
        "sources": {},
    }
    try:
        source_names = [
            "tools/verify_prott5_loader_rng.py",
            "tools/verify_prott5_equivalence.py",
            "pepplatform/encoders.py",
            "pepplatform/tokenizers.py",
            "pepplatform/prott5.py",
            "pepplatform/models.py",
            "pepplatform/workflow.py",
            "scripts/train_sequence_pipeline.py",
            "scripts/train_pair_pipeline.py",
        ]
        report["source_files"] = {name: file_sha(ROOT / name) for name in source_names}
        for name in source_names:
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        manager = model_manager()
        lock = manager.load_lock()
        print("Auditing frozen cohort and checkpoint file identities", flush=True)
        manager.reference_sources(args.reference, lock)
        model = next(m for m in lock["models"] if m["key"] == "prott5")
        manager.verify_tree(args.sharded, model)
        if file_sha(args.official / "pytorch_model.bin") != PROTT5_OFFICIAL_WEIGHT_SHA256:
            raise ValueError("Official checkpoint SHA256 mismatch")
        report["reference"] = lock["reference"]
        report["official_sha256"] = PROTT5_OFFICIAL_WEIGHT_SHA256
        report["sharded_weight_sha256"] = model["weight_sha256"]
        registration = read_json(args.reference)
        index = read_json(registration["results_index"]["path"])
        row = next(
            r
            for r in index["candidates"]
            if r["task"] == "pep_function" and r["candidate"] == "c001_prott5_residue_cnn_mlp"
        )
        metadata_path = Path(row["artifact"]) / "metadata.json"
        if file_sha(metadata_path) != row["metadata_sha256"]:
            raise ValueError("Registered classifier metadata SHA256 mismatch")
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
            "constructor": "pepplatform.models.PretrainedResiduePredictor",
            "arguments": head_kwargs,
        }
        report["execution"] = {
            "seed": seed,
            "device": "cpu",
            "threads": args.threads,
            "torch": torch.__version__,
        }
        torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True)

        def reset_seed():
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        def head_state():
            head = PretrainedResiduePredictor(**head_kwargs)
            return {name: tensor_sha(tensor) for name, tensor in head.state_dict().items()}

        for role, source in [("sharded", args.sharded), ("official", args.official)]:
            print("Checking loader and registered CNN initialization:", role, flush=True)
            reset_seed()
            before = rng_state()
            tokenizer = ProT5Tokenizer(model_name=str(source), max_length=16)
            encoder = ProT5Encoder(model_name=str(source), device=torch.device("cpu"))
            loaded = rng_state()
            torch.save(torch.get_rng_state(), output / (role + "-rng-after-loading.pt"))
            encoder.encode(tokenizer.batch_tokenize(["ACDEFGHIK", "UZOB"]))
            forwarded = rng_state()
            unseeded_head = head_state()
            reset_seed()
            reseeded_head = head_state()
            report["sources"][role] = {
                "before": before,
                "after_loading": loaded,
                "after_forward": forwarded,
                "classifier_state_sha256": unseeded_head,
                "classifier_state_after_explicit_reseed_sha256": reseeded_head,
            }
            del encoder, tokenizer
            gc.collect()
            write_json(output / "report.json", report)
        left, right = report["sources"]["sharded"], report["sources"]["official"]
        report["loader_rng_equal"] = left["after_loading"] == right["after_loading"]
        report["forward_rng_equal"] = left["after_forward"] == right["after_forward"]
        report["classifier_initialization_equal"] = (
            left["classifier_state_sha256"] == right["classifier_state_sha256"]
        )
        report["reseeded_classifier_initialization_equal"] = (
            left["classifier_state_after_explicit_reseed_sha256"]
            == right["classifier_state_after_explicit_reseed_sha256"]
        )
        report["status"] = "PASS" if report["loader_rng_equal"] else "DIFFERENT"
        if any(file_sha(ROOT / name) != digest for name, digest in report["source_files"].items()):
            raise ValueError("Diagnostic source changed during execution")
    except Exception as exc:
        report.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "report.json", report)
    print("Loader RNG:", report["status"], flush=True)
    print(
        "Registered CNN initialization equal:",
        report["classifier_initialization_equal"],
        flush=True,
    )
    return 0 if report["status"] == "PASS" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", required=True, type=Path)
    parser.add_argument("--sharded", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument(
        "--output", required=True, type=Path, help="A new isolated output directory"
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("Thread count must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
