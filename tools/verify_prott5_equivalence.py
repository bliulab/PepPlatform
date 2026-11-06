"""Read-only proof of equivalence between official ProtT5 and paper shards.

Run with the frozen paper Python environment. Official downloads must already
exist in an isolated directory; this verifier performs no network requests.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import os
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {
    "encoder.embed_tokens.weight": "shared.weight",
    "decoder.embed_tokens.weight": "shared.weight",
    "lm_head.weight": "shared.weight",
}
CONFIG_METADATA = {"_name_or_path", "transformers_version", "torch_dtype", "dtype", "architectures"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(tensor):
    import torch

    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(data)).hexdigest()


def compare_tensor(left, right):
    import torch

    record = {
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "left_sha256": tensor_sha(left),
        "right_sha256": tensor_sha(right),
    }
    record["bitwise_equal"] = (
        record["left_shape"] == record["right_shape"]
        and left.dtype == right.dtype
        and record["left_sha256"] == record["right_sha256"]
    )
    if record["left_shape"] == record["right_shape"]:
        different = int(torch.count_nonzero(left != right).item())
        record["different_values"] = different
        if different and (left.is_floating_point() or right.is_floating_point()):
            difference = (left.double() - right.double()).abs()
            maximum = float(difference.max().item()) if difference.numel() else 0.0
            record["max_absolute_error"] = maximum if maximum < float("inf") else None
        else:
            record["max_absolute_error"] = 0.0 if not different else None
    return record


def canonical_keys(keys):
    return set(keys) - {alias for alias in ALIASES if alias in keys}


def compare_state(official_file, sharded_root, progress=print):
    import torch
    from safetensors import safe_open

    state = torch.load(official_file, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Official file is not a tensor state dictionary")
    sharded_root = Path(sharded_root)
    index = read_json(sharded_root / "model.safetensors.index.json")
    mapping = index["weight_map"]
    official_keys, shard_keys = canonical_keys(state), canonical_keys(mapping)
    records = []
    alias_checks = []
    for alias, canonical in ALIASES.items():
        if alias in state:
            check = compare_tensor(state[alias], state[canonical])
            alias_checks.append(
                {"source": "official", "alias": alias, "canonical": canonical, **check}
            )
        if alias in mapping:
            with safe_open(sharded_root / mapping[alias], framework="pt", device="cpu") as source:
                alias_value = source.get_tensor(alias)
            with safe_open(
                sharded_root / mapping[canonical], framework="pt", device="cpu"
            ) as source:
                canonical_value = source.get_tensor(canonical)
            check = compare_tensor(alias_value, canonical_value)
            alias_checks.append(
                {"source": "sharded", "alias": alias, "canonical": canonical, **check}
            )
    for filename in sorted(set(mapping.values())):
        progress("Comparing tensor bytes: " + filename)
        with safe_open(sharded_root / filename, framework="pt", device="cpu") as source:
            if set(source.keys()) != {key for key, value in mapping.items() if value == filename}:
                raise ValueError("Shard header and index disagree: " + filename)
            for name in sorted(source.keys()):
                if name not in official_keys & shard_keys:
                    continue
                original, shard = state[name], source.get_tensor(name)
                comparison = compare_tensor(original, shard)
                records.append({"name": name, "parameter_count": original.numel(), **comparison})
    groups = {}
    for group in ("shared", "encoder", "decoder"):
        selected = [r for r in records if r["name"].split(".")[0] == group]
        groups[group] = {
            "tensor_count": len(selected),
            "parameter_count": sum(r["parameter_count"] for r in selected),
            "bitwise_equal": all(r["bitwise_equal"] for r in selected),
        }
    fingerprints = {}
    for side in ("left", "right"):
        h = hashlib.sha256()
        for row in sorted(records, key=lambda r: r["name"]):
            fields = [
                row["name"],
                row[side + "_shape"],
                row[side + "_dtype"],
                row[side + "_sha256"],
            ]
            h.update(json.dumps(fields, separators=(",", ":")).encode())
            h.update(b"\n")
        fingerprints[side] = h.hexdigest()
    missing = sorted(official_keys - shard_keys)
    extra = sorted(shard_keys - official_keys)
    aliases_equal = all(r["bitwise_equal"] for r in alias_checks)
    result = {
        "official_raw_tensor_count": len(state),
        "sharded_raw_tensor_count": len(mapping),
        "canonical_tensor_count": len(records),
        "missing_from_shards": missing,
        "extra_in_shards": extra,
        "aliases": alias_checks,
        "groups": groups,
        "canonical_fingerprints": fingerprints,
        "tensors": records,
    }
    result["full_state_bitwise_equal"] = (
        not missing and not extra and aliases_equal and all(r["bitwise_equal"] for r in records)
    )
    encoder_names = {n for n in official_keys | shard_keys if n.startswith(("shared.", "encoder."))}
    checked_encoder = {r["name"] for r in records if r["name"] in encoder_names}
    encoder_aliases_equal = all(
        r["bitwise_equal"] for r in alias_checks if r["alias"].startswith("encoder.")
    )
    result["encoder_state_bitwise_equal"] = (
        encoder_names == checked_encoder
        and encoder_aliases_equal
        and all(r["bitwise_equal"] for r in records if r["name"] in encoder_names)
    )
    del state
    gc.collect()
    return result


def dictionary_differences(left, right):
    return {
        key: {"left": left.get(key), "right": right.get(key)}
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }


def compare_config(official, sharded):
    """Compare configuration for the platform's explicit T5EncoderModel loader.

    The architectures field selects Auto classes, whereas ProT5Encoder explicitly
    instantiates T5EncoderModel. Record that metadata difference and separately
    require the actual loaded classes, tensor dtypes and state to match.
    """
    from transformers import T5Config

    left_raw, right_raw = read_json(official / "config.json"), read_json(sharded / "config.json")
    left = T5Config.from_pretrained(official, local_files_only=True).to_dict()
    right = T5Config.from_pretrained(sharded, local_files_only=True).to_dict()
    for field in CONFIG_METADATA:
        left.pop(field, None)
        right.pop(field, None)
    differences = dictionary_differences(left, right)
    return {
        "effective_equal": not differences,
        "effective_differences": differences,
        "raw_differences": dictionary_differences(left_raw, right_raw),
        "excluded_serialization_metadata": sorted(CONFIG_METADATA),
        "effective_official": left,
        "effective_sharded": right,
        "precision_check": "Actual raw and loaded tensor dtypes are compared separately.",
        "architecture_check": "The platform explicitly uses T5EncoderModel; actual loaded classes are checked separately.",
    }


def validate_dataset(path, declared):
    digest = file_sha(path)
    # Relocated copies must have the exact content of a registered input.
    matched = sorted(name for name, value in declared.items() if value == digest)
    if not matched:
        raise ValueError("Validation input is not registered: " + str(path))
    if str(path) in declared and declared[str(path)] != digest:
        raise ValueError("Registered validation input changed: " + str(path))
    return {"path": str(path), "sha256": digest, "registered_identical_paths": matched}


def first_fasta(path, count):
    records = []
    identifier, parts = None, []
    with Path(path).open() as stream:
        for line in stream:
            if line.startswith(">"):
                if identifier is not None:
                    records.append((identifier, "".join(parts)))
                    if len(records) == count:
                        return records
                identifier, parts = line[1:].strip(), []
            else:
                parts.append(line.strip())
    if identifier is not None:
        records.append((identifier, "".join(parts)))
    if len(records) < count:
        raise ValueError("Insufficient registered FASTA samples")
    return records[:count]


def model_manager():
    spec = importlib.util.spec_from_file_location(
        "paper_model_manager", ROOT / "docker/download_models.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registered_cases(reference, lock, count):
    registration = read_json(reference)
    index_path = Path(registration["results_index"]["path"])
    index = read_json(index_path)
    protocol = read_json(index_path.parent / "completion_protocol.json")
    declared = protocol["datasets"]
    probes, sources = [], []
    for task in ("pep_function", "pep_protein_interaction", "pep_mhci", "pep_hlaii"):
        row = next(
            r for r in index["candidates"] if r["task"] == task and "prott5" in r["candidate"]
        )
        metadata_path = Path(row["artifact"]) / "metadata.json"
        if file_sha(metadata_path) != row["metadata_sha256"]:
            raise ValueError("Candidate metadata changed during validation")
        metadata = read_json(metadata_path)
        args = metadata["args"]
        source = (
            Path(args["data_dir"]) / "val/seqs.fasta"
            if task == "pep_function"
            else Path(args["val_csv"])
        )
        identity = validate_dataset(source, declared)
        identity.update(
            task=task, candidate=row["candidate"], metadata_sha256=row["metadata_sha256"]
        )
        sources.append(identity)
        streams = {}
        if task == "pep_function":
            entries = first_fasta(source, count)
            streams["sequence"] = [sequence for _identifier, sequence in entries]
            identifiers = [identifier for identifier, _sequence in entries]
        else:
            with source.open() as stream:
                entries = list(itertools.islice(csv.DictReader(stream), count))
            if len(entries) != count:
                raise ValueError("Insufficient registered CSV samples")
            identifiers = [entry.get("sample_id", str(i)) for i, entry in enumerate(entries)]
            if task == "pep_mhci":
                mappings = [
                    Path(p)
                    for p in declared
                    if p.endswith("/task_peptide_MHCI_interaction/pseudoseqs_decoded.csv")
                ]
                if len(mappings) != 1:
                    raise ValueError("Ambiguous registered HLA-I pseudosequence map")
                mapping_path = mappings[0]
                sources.append(validate_dataset(mapping_path, declared))
                with mapping_path.open() as stream:
                    mapping = {
                        r["mhc"]: r["pseudoseq"]
                        for r in csv.DictReader(stream)
                        if r["status"] == "ok"
                    }
                streams["peptide"] = [r["pep"] for r in entries]
                streams["partner"] = [mapping[r["mhc"]] for r in entries]
            else:
                streams["peptide"] = [r["peptide"] for r in entries]
                streams["partner"] = [
                    r["protein" if task == "pep_protein_interaction" else "pair_pseudo"]
                    for r in entries
                ]
        for name, sequences in streams.items():
            probes.append(
                {
                    "name": task + "." + name,
                    "max_length": args["max_length"],
                    "sample_ids": identifiers,
                    "sequences": sequences,
                }
            )
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    probes.extend(
        [
            {
                "name": "boundary.normalization_padding_truncation",
                "max_length": 16,
                "sequences": ["A", "acdef ghik", "UZOB", "ACDEFGHIKLMNPQRSTVWYX", "X"],
            },
            {
                "name": "boundary.max_length_511",
                "max_length": 511,
                "sequences": [(alphabet * 26)[:511], (alphabet * 26)[:512]],
            },
        ]
    )
    return {
        "reference": lock["reference"],
        "input_files": sources,
        "probes": probes,
        "rejected_inputs": ["", "  ", "A|C", "123"],
    }


def compare_outputs(left, right):
    if set(left) != set(right):
        raise ValueError("Output fields disagree")
    comparisons = {key: compare_tensor(left[key], right[key]) for key in left}
    return {
        "bitwise_equal": all(v["bitwise_equal"] for v in comparisons.values()),
        "fields": comparisons,
    }


def forward_source(source, role, cases, output, progress, *, on_encoder_loaded=None, device="cpu"):
    import torch

    from pepplatform.encoders import ProT5Encoder
    from pepplatform.tokenizers import ProT5Tokenizer

    progress("Loading " + role + " encoder on " + str(device))
    encoder = ProT5Encoder(model_name=str(source), device=torch.device(device))
    if on_encoder_loaded is not None:
        on_encoder_loaded(encoder)
    directory = output / role
    directory.mkdir()
    state = encoder.model.state_dict()
    loaded = {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": tensor_sha(tensor),
        }
        for name, tensor in sorted(state.items())
    }
    del state
    results = []
    rejected = {}
    tokenizer = ProT5Tokenizer(model_name=str(source), max_length=16)
    for text in cases["rejected_inputs"]:
        try:
            tokenizer.batch_tokenize([text])
        except ValueError:
            rejected[text] = "ValueError"
        else:
            rejected[text] = "accepted"
    for case in cases["probes"]:
        progress(role + ": " + case["name"])
        limit = case["max_length"]
        tokenizer = ProT5Tokenizer(model_name=str(source), max_length=limit)
        inputs = tokenizer.batch_tokenize(case["sequences"])
        lengths = [tokenizer.residue_length(s, max_residues=limit) for s in case["sequences"]]
        captured = {}

        def remember_hidden(_module, _args, result, captured=captured):
            captured["hidden_fp32"] = result.last_hidden_state.detach().cpu()

        hook = encoder.model.register_forward_hook(remember_hidden)
        try:
            pooled = encoder.encode(inputs)
        finally:
            hook.remove()
        residue = encoder.encode_residue_batch(inputs, lengths, limit)
        tensors = {"tokens." + k: v.detach().cpu() for k, v in inputs.items()}
        tensors.update(
            pooled=pooled.detach().cpu(),
            residue=residue.features.detach().cpu(),
            residue_mask=residue.mask.detach().cpu(),
            **captured,
        )
        if any(not torch.isfinite(t).all() for t in tensors.values() if t.is_floating_point()):
            raise ValueError("Non-finite output: " + case["name"])
        torch.save(tensors, directory / (case["name"] + ".pt"))
        results.append(
            {
                "name": case["name"],
                "sample_count": len(lengths),
                "residue_lengths": lengths,
                "max_length": limit,
            }
        )
    identity = dict(encoder.model_identity)
    loader_class = encoder.model.__class__.__module__ + "." + encoder.model.__class__.__name__
    del encoder, tokenizer
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return {
        "loaded_state": loaded,
        "rejected_inputs": rejected,
        "cases": results,
        "model_identity": identity,
        "loader_class": loader_class,
    }


def markdown_report(report):
    state = report.get("state", {})
    lines = [
        "# ProtT5 单文件与六分片等价性验证",
        "",
        "结果：**" + report["status"] + "**",
        "",
        "- 固定官方 revision：`" + report.get("revision", "") + "`",
        "- 官方文件 SHA-256：`" + report.get("official_sha256", "未完成") + "`",
        "- 原实验分片聚合 SHA-256：`" + report.get("sharded_weight_sha256", "未完成") + "`",
        "- 完整模型逐张量位级一致：" + str(state.get("full_state_bitwise_equal")),
        "- 编码器与共享嵌入逐张量位级一致：" + str(state.get("encoder_state_bitwise_equal")),
        "- 有效配置一致：" + str(report.get("config", {}).get("effective_equal")),
        "- 加载后的编码器状态一致：" + str(report.get("loaded_encoder_state_equal")),
        "- 分词、掩码、残基及池化输出位级一致：" + str(report.get("outputs_equal")),
        "",
        "| 参数组 | 张量数 | 参数数 | 位级一致 |",
        "| --- | ---: | ---: | --- |",
    ]
    for group, item in state.get("groups", {}).items():
        lines.append(
            f"| {group} | {item['tensor_count']} | {item['parameter_count']} | {item['bitwise_equal']} |"
        )
    lines.extend(
        [
            "",
            "## 运行环境",
            "",
            "```json",
            json.dumps(report.get("environment", {}), indent=2),
            "```",
            "",
            "## 范围",
            "",
            "这是固定 checkpoint 的参数和平台特征提取等价性验证，未重新训练全部候选。",
            "文件格式不同会导致文件哈希不同；结果以逐张量数据、有效配置和相同输入的实际输出为依据。",
            "",
            "## 复跑",
            "",
            "```bash",
            report.get("rerun_command", ""),
            "```",
            "",
        ]
    )
    if "error" in report:
        lines.extend(["错误：" + report["error"], ""])
    return "\n".join(lines)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "official_directory": str(args.official.resolve()),
        "sharded_directory": str(args.sharded.resolve()),
    }
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--official",
        str(args.official.resolve()),
        "--sharded",
        str(args.sharded.resolve()),
        "--reference",
        str(args.reference.resolve()),
        "--output",
        str(output.with_name(output.name + "-rerun")),
        "--threads",
        str(args.threads),
        "--samples-per-task",
        str(args.samples_per_task),
    ]
    report["rerun_command"] = shlex.join(command)

    def progress(message):
        print(message, flush=True)
        write_json(
            output / "progress.json",
            {"stage": message, "updated_at": datetime.now(timezone.utc).isoformat()},
        )

    try:
        os.environ.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            PEPPLATFORM_OFFLINE_MODE="1",
            TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS=str(args.threads),
            OPENBLAS_NUM_THREADS=str(args.threads),
        )
        sys.path.insert(0, str(ROOT))
        source_names = [
            "tools/verify_prott5_equivalence.py",
            "pepplatform/encoders.py",
            "pepplatform/tokenizers.py",
            "pepplatform/prott5.py",
            "pepplatform/cache_provenance.py",
            "pepplatform/workflow.py",
            "docker/download_models.py",
            "docker/models.lock.json",
        ]
        report["source_files"] = {name: file_sha(ROOT / name) for name in source_names}
        for name in source_names:
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        from pepplatform.prott5 import PROTT5_HF_REVISION, PROTT5_OFFICIAL_WEIGHT_SHA256

        manager = model_manager()
        lock = manager.load_lock()
        progress("Auditing frozen candidate reference and original file identities")
        manager.reference_sources(args.reference, lock)
        model = next(m for m in lock["models"] if m["key"] == "prott5")
        manager.verify_tree(args.sharded, model)
        report.update(
            revision=PROTT5_HF_REVISION,
            reference=lock["reference"],
            sharded_weight_sha256=model["weight_sha256"],
            sharded_model_fingerprint=model["model_fingerprint"],
        )
        progress("Verifying official single-file SHA256 before loading")
        official_file = args.official / "pytorch_model.bin"
        digest = file_sha(official_file)
        report["official_sha256"] = digest
        if digest != PROTT5_OFFICIAL_WEIGHT_SHA256:
            raise ValueError("Official file SHA256 does not match the frozen upstream checkpoint")
        import torch

        torch.set_num_threads(args.threads)
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        report["environment"] = {
            p: importlib.metadata.version(p)
            for p in [
                "torch",
                "transformers",
                "tokenizers",
                "sentencepiece",
                "safetensors",
                "huggingface-hub",
            ]
        }
        report["execution"] = {
            "device": "cpu",
            "threads": args.threads,
            "seed": 42,
            "deterministic_algorithms": True,
            "network": "disabled by HF offline flags",
        }
        report["state"] = compare_state(official_file, args.sharded, progress)
        write_json(output / "report.json", report)
        report["config"] = compare_config(args.official, args.sharded)
        report["tokenizer_files"] = {}
        for name in ["spiece.model", "special_tokens_map.json", "tokenizer_config.json"]:
            left_sha, right_sha = file_sha(args.official / name), file_sha(args.sharded / name)
            report["tokenizer_files"][name] = {
                "official_sha256": left_sha,
                "sharded_sha256": right_sha,
                "byte_equal": left_sha == right_sha,
            }
        cases = registered_cases(args.reference, lock, args.samples_per_task)
        write_json(output / "cases.json", cases)
        report["case_manifest_sha256"] = file_sha(output / "cases.json")
        results = {}
        for role, source in [("sharded", args.sharded), ("official", args.official)]:
            results[role] = forward_source(source, role, cases, output, progress)
            write_json(output / (role + ".json"), results[role])
        report["loaded_encoder_state_equal"] = (
            results["sharded"]["loaded_state"] == results["official"]["loaded_state"]
        )
        report["loader_class_equal"] = (
            results["sharded"]["loader_class"]
            == results["official"]["loader_class"]
            == "transformers.models.t5.modeling_t5.T5EncoderModel"
        )
        report["rejected_inputs_equal"] = (
            results["sharded"]["rejected_inputs"] == results["official"]["rejected_inputs"]
        )
        report["output_cases"] = []
        for case in cases["probes"]:
            name = case["name"]
            left = torch.load(
                output / "sharded" / (name + ".pt"), map_location="cpu", weights_only=True
            )
            right = torch.load(
                output / "official" / (name + ".pt"), map_location="cpu", weights_only=True
            )
            report["output_cases"].append(
                {
                    "name": name,
                    "sample_count": len(case["sequences"]),
                    **compare_outputs(left, right),
                }
            )
        report["outputs_equal"] = (
            all(c["bitwise_equal"] for c in report["output_cases"])
            and report["rejected_inputs_equal"]
        )
        passed = (
            report["state"]["full_state_bitwise_equal"]
            and report["config"]["effective_equal"]
            and report["loaded_encoder_state_equal"]
            and report["loader_class_equal"]
            and report["outputs_equal"]
            and all(v["byte_equal"] for v in report["tokenizer_files"].values())
        )
        report["status"] = "PASS" if passed else "DIFFERENT"
        report["loaded_models"] = {role: value["model_identity"] for role, value in results.items()}
        if any(file_sha(ROOT / name) != digest for name, digest in report["source_files"].items()):
            raise ValueError("Verification source changed during execution")
    except Exception as exc:
        report.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "report.json", report)
        (output / "REPORT.md").write_text(markdown_report(report), encoding="utf-8")
    progress("Verification complete: " + report["status"])
    return 0 if report["status"] == "PASS" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", required=True, type=Path)
    parser.add_argument("--sharded", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument(
        "--output", required=True, type=Path, help="A new isolated report directory"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--samples-per-task", type=int, default=8)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.samples_per_task < 1:
        parser.error("Thread and sample counts must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
