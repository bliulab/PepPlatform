"""Deterministic offline builder and verifier for the existing upload protocol."""

from __future__ import annotations
from pepplatform.task_compat import normalize_task_type
from pepplatform.hlai_input import canonical_hlai_columns, HLAIInputError


import csv
import platform
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from . import adapters
from .benchmarks import overlap_report, sample_key
from .common import (
    AA,
    DataError,
    VERSION,
    canonical,
    check_inventory,
    digest,
    file_inventory,
    load_recipe,
    new_directory,
    read_json,
    require_keys,
    sequence,
    sha,
    split_for,
    write_json,
)


def _builder_sources():
    """Freeze all standard-library dependencies for standalone replay."""
    directory = Path(__file__).parent
    sources = {p.name: p.read_bytes() for p in sorted(directory.glob("*.py"))}
    for helper in ("task_compat.py", "hlai_input.py"):
        if helper not in sources:
            sources[helper] = (directory.parent / helper).read_bytes()
    return {
        name: content.replace(
            b"from pepplatform.task_compat import", b"from .task_compat import"
        ).replace(b"from pepplatform.hlai_input import", b"from .hlai_input import")
        for name, content in sources.items()
    }


def _supplement(row, recipe, mapping):
    require_keys(
        row,
        {
            "source_id",
            "peptide",
            "partner_id",
            "protein",
            "labels",
            "reference",
            "evidence_type",
            "unmodified",
            "task",
            "label_definition",
        },
        {
            "source_id",
            "peptide",
            "labels",
            "reference",
            "evidence_type",
            "unmodified",
            "task",
            "label_definition",
        },
        "supplement",
    )
    if (
        row["task"] != recipe["task"]
        or row["evidence_type"] != "experimental"
        or row["unmodified"] is not True
        or not isinstance(row["reference"], str)
        or not row["reference"]
        or not row["source_id"]
    ):
        raise DataError("supplement_requires_explicit_experimental_provenance")
    expected = (
        recipe.get("labels")
        if normalize_task_type(recipe["task"]) == "pep_function"
        else "direct_binding_positive=1,negative=0"
    )
    if row["label_definition"] != expected:
        raise DataError("supplement_label_definition_mismatch")
    width = len(recipe["labels"]) if normalize_task_type(recipe["task"]) == "pep_function" else 1
    if (
        not isinstance(row["labels"], list)
        or len(row["labels"]) != width
        or any(type(x) is not int or x not in {0, 1} for x in row["labels"])
    ):
        raise DataError("invalid_supplement_labels")
    partner, identity = "", ""
    if normalize_task_type(recipe["task"]) == "pep_protein_interaction":
        partner = identity = sequence(row.get("protein"))
    elif normalize_task_type(recipe["task"]) in {"pep_hlai", "pep_hlaii"}:
        identity = adapters.allele(row.get("partner_id"), recipe["task"])
        if identity not in mapping:
            raise DataError("missing_exact_pseudosequence:" + identity)
        partner = mapping[identity]
    return {
        "peptide": sequence(row["peptide"]),
        "partner": partner,
        "partner_id": identity,
        "labels": row["labels"],
        "evidence": [row],
    }


def observations(snapshot, manifest, recipe):
    resources = manifest["resources"]
    mapping = (
        adapters.load_pseudosequences(snapshot / resources["pseudosequences"], recipe["task"])
        if "pseudosequences" in resources
        else {}
    )
    groups, rejected, poisoned = defaultdict(list), [], set()
    raw = read_json(snapshot / "records.json")
    if not isinstance(raw, list) or len(raw) != manifest["record_count"]:
        raise DataError("snapshot record count mismatch")
    sources = [(recipe["source"], x["source_id"], x["record"]) for x in raw]
    if "supplements" in resources:
        extra = read_json(snapshot / resources["supplements"])
        if not isinstance(extra, list):
            raise DataError("supplements must be a list")
        sources.extend(("user_experimental_supplement", x.get("source_id", ""), x) for x in extra)
    seen = set()
    for source, source_id, raw_row in sources:
        if not str(source_id) or (source, str(source_id)) in seen:
            raise DataError("missing or duplicate provenance ID")
        seen.add((source, str(source_id)))
        try:
            if source == "user_experimental_supplement":
                row = _supplement(raw_row, recipe, mapping)
            elif source == "dbaasp":
                row = adapters.dbaasp(raw_row, recipe)
            elif source == "iedb":
                row = adapters.iedb(raw_row, recipe, mapping)
            else:
                row = adapters.intact(raw_row, recipe)
            if (
                not recipe["peptide_length"][0]
                <= len(row["peptide"])
                <= recipe["peptide_length"][1]
            ):
                raise DataError("peptide_length_outside_recipe")
            row["source"] = source
            row["source_id"] = str(source_id)
            key = sample_key(recipe["task"], row["peptide"], row["partner_id"])
            groups[key].append(row)
        except DataError as exc:
            if str(exc).startswith("source_schema_missing"):
                raise  # A schema change cannot silently remove a subset of the cohort.
            if str(exc) == "conflicting_assay_labels":
                poisoned.add(sample_key(recipe["task"], raw_row["sequence"]))
            rejected.append({"source": source, "source_id": str(source_id), "reason": str(exc)})
    accepted = []
    for key, items in sorted(groups.items()):
        if (
            key in poisoned
            or len({tuple(x["labels"]) for x in items}) != 1
            or len({x["partner"] for x in items}) != 1
        ):
            rejected.extend(
                {
                    "source": x["source"],
                    "source_id": x["source_id"],
                    "reason": "conflicting_sample_evidence",
                }
                for x in items
            )
            continue
        first = items[0]
        accepted.append(
            {
                "sample_id": key,
                "peptide": first["peptide"],
                "partner": first["partner"],
                "partner_id": first["partner_id"],
                "labels": first["labels"],
                "split": split_for(first["peptide"], recipe),
                "observations": sorted(items, key=lambda x: (x["source"], x["source_id"])),
            }
        )
    return accepted, sorted(rejected, key=lambda x: (x["source"], x["source_id"])), len(sources)


def summarize(rows, recipe):
    width = len(recipe["labels"]) if normalize_task_type(recipe["task"]) == "pep_function" else 1
    result, problems = {}, []
    for split in ["train", "val", "test"]:
        selected = [r for r in rows if r["split"] == split]
        counts = [
            {
                "positive": sum(r["labels"][i] for r in selected),
                "negative": sum(1 - r["labels"][i] for r in selected),
            }
            for i in range(width)
        ]
        lengths = [len(r["peptide"]) for r in selected]
        result[split] = {
            "samples": len(selected),
            "peptides": len({r["peptide"] for r in selected}),
            "label_counts": counts,
            "peptide_length_min": min(lengths, default=None),
            "peptide_length_max": max(lengths, default=None),
        }
        if not selected or any(not c["positive"] or not c["negative"] for c in counts):
            problems.append(f"{split}: every label requires positive and negative samples")
    return result, problems


def _csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def pair_record(row, task):
    task = normalize_task_type(task)
    partner_col = (
        "protein" if normalize_task_type(task) == "pep_protein_interaction" else "pair_pseudo"
    )
    result = {
        "sample_id": row["sample_id"],
        "peptide": row["peptide"],
        partner_col: row["partner"],
        "partner_id": row["partner_id"],
        "label": str(row["labels"][0]),
    }
    if normalize_task_type(task) == "pep_hlai":
        result["hla"] = row["partner_id"]
    elif normalize_task_type(task) == "pep_hlaii":
        result["hla_pair"] = row["partner_id"]
        result["hla_alpha"], result["hla_beta"] = row["partner_id"].split("|")
    return result


def write_dataset(root, rows, recipe):
    function = normalize_task_type(recipe["task"]) == "pep_function"
    columns = [x["name"] for x in recipe["labels"]] if function else ["label"]
    for split in ["train", "val", "test"]:
        folder = root / split
        folder.mkdir(parents=True)
        selected = [r for r in rows if r["split"] == split]
        _csv(folder / "labels.csv", columns, [r["labels"] for r in selected])
        (folder / "seqs.fasta").write_text(
            "".join(
                f">{r['sample_id']}\n{r['peptide'] if function else r['peptide'] + 'GGG' + r['partner']}\n"
                for r in selected
            ),
            encoding="utf-8",
        )
        if not function:
            pairs = [pair_record(r, recipe["task"]) for r in selected]
            _csv(folder / "pairs.csv", list(pairs[0]), [list(p.values()) for p in pairs])


def _zip(root, destination):
    # ZIP_STORED avoids compressor-version-dependent bytes.
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_STORED) as archive:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                info = zipfile.ZipInfo(
                    p.relative_to(root).as_posix(), date_time=(1980, 1, 1, 0, 0, 0)
                )
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, p.read_bytes())


def build(snapshot, recipe_path, out):
    snapshot = Path(snapshot).resolve()
    recipe = load_recipe(recipe_path)
    manifest = read_json(snapshot / "snapshot.json")
    if (
        manifest.get("version") != VERSION
        or manifest.get("kind") != "snapshot"
        or manifest.get("complete") is not True
    ):
        raise DataError("snapshot is incomplete or unsupported")
    check_inventory(snapshot, manifest["files"], {"snapshot.json"})
    if recipe != manifest["recipe"]:
        raise DataError("recipe differs from frozen snapshot recipe; fetch a new snapshot")
    # Only inventory-listed, relative resource paths may be read.
    for relative in manifest["resources"].values():
        if relative not in manifest["files"] or not (snapshot / relative).resolve().is_relative_to(
            snapshot
        ):
            raise DataError("unsafe snapshot resource path")
    root = new_directory(out)
    report = {"version": VERSION, "ready": False, "task": recipe["task"]}
    try:
        rows, rejected, input_count = observations(snapshot, manifest, recipe)
        splits, problems = summarize(rows, recipe)
        report.update(
            input_records=input_count,
            accepted_samples=len(rows),
            rejected_records=len(rejected),
            rejection_reasons=dict(sorted(Counter(r["reason"] for r in rejected).items())),
            splits=splits,
            problems=problems,
            cross_split_peptide_overlap=0,
            overlap=overlap_report(
                rows, read_json(snapshot / manifest["resources"]["benchmarks"]), recipe["task"]
            ),
        )
        write_json(root / "rejections.json", rejected)
        write_json(root / "accepted.json", rows)
        if problems:
            raise DataError("insufficient evidence for a trainable dataset; see report.json")
        dataset = root / "dataset"
        write_dataset(dataset, rows, recipe)
        builder_sources = _builder_sources()
        builder_identity = {
            name: digest(content) for name, content in sorted(builder_sources.items())
        }
        write_json(
            dataset / "provenance.json",
            {
                "version": VERSION,
                "recipe": recipe,
                "snapshot_sha256": sha(snapshot / "snapshot.json"),
                "snapshot_files": manifest["files"],
                "builder_identity": builder_identity,
                "python": platform.python_version(),
                "samples": rows,
            },
        )
        report["ready"] = True
        write_json(dataset / "data_report.json", report)
        write_json(dataset / "recipe.json", recipe)
        code = root / "builder"
        code.mkdir()
        for name, content in sorted(builder_sources.items()):
            (code / name).write_bytes(content)
        _zip(dataset, root / "dataset.zip")
        write_json(root / "report.json", report)
        write_json(
            root / "manifest.json",
            {
                "version": VERSION,
                "kind": "dataset",
                "ready": True,
                "task": recipe["task"],
                "python": platform.python_version(),
                "snapshot_sha256": sha(snapshot / "snapshot.json"),
                "files": file_inventory(root, {"manifest.json"}),
            },
        )
        verify(root)
    except Exception as exc:
        report.update(ready=False, error=str(exc))
        write_json(root / "report.json", report)
        # Never leave a failed publication looking usable.
        (root / "dataset.zip").unlink(missing_ok=True)
        (root / "manifest.json").unlink(missing_ok=True)
        raise
    return report


def verify(root):
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    if (
        manifest.get("version") != VERSION
        or manifest.get("kind") != "dataset"
        or manifest.get("ready") is not True
    ):
        raise DataError("dataset was not published successfully")
    check_inventory(root, manifest["files"], {"manifest.json"})
    dataset = root / "dataset"
    recipe = load_recipe(dataset / "recipe.json")
    rows = read_json(root / "accepted.json")
    provenance = read_json(dataset / "provenance.json")
    if (
        provenance["samples"] != rows
        or provenance["recipe"] != recipe
        or provenance["snapshot_sha256"] != manifest["snapshot_sha256"]
    ):
        raise DataError("provenance mismatch")
    if (
        provenance["builder_identity"] != file_inventory(root / "builder")
        or provenance["python"] != manifest["python"]
        or recipe["task"] != manifest["task"]
    ):
        raise DataError("builder identity mismatch")
    seen, peptide_splits = set(), {}
    width = len(recipe["labels"]) if normalize_task_type(recipe["task"]) == "pep_function" else 1
    for row in rows:
        if row["sample_id"] in seen or row["sample_id"] != sample_key(
            recipe["task"], row["peptide"], row["partner_id"]
        ):
            raise DataError("duplicate or invalid sample identity")
        seen.add(row["sample_id"])
        sequence(row["peptide"])
        if not recipe["peptide_length"][0] <= len(row["peptide"]) <= recipe["peptide_length"][1]:
            raise DataError("peptide length mismatch")
        if normalize_task_type(recipe["task"]) == "pep_protein_interaction":
            if sequence(row["partner"]) != row["partner_id"]:
                raise DataError("protein identity mismatch")
        elif normalize_task_type(recipe["task"]) in {"pep_hlai", "pep_hlaii"}:
            if len(row["partner"]) != (
                30 if normalize_task_type(recipe["task"]) == "pep_hlai" else 34
            ) or not set(row["partner"]) <= AA | {"X"}:
                raise DataError("invalid pseudosequence")
        if len(row["labels"]) != width or any(
            type(x) is not int or x not in {0, 1} for x in row["labels"]
        ):
            raise DataError("invalid labels")
        if row["split"] != split_for(row["peptide"], recipe):
            raise DataError("split assignment mismatch")
        if row["peptide"] in peptide_splits and peptide_splits[row["peptide"]] != row["split"]:
            raise DataError("cross-split peptide leakage")
        peptide_splits[row["peptide"]] = row["split"]
    summary, problems = summarize(rows, recipe)
    report = read_json(root / "report.json")
    if (
        problems
        or report["splits"] != summary
        or report["accepted_samples"] != len(rows)
        or report["ready"] is not True
        or read_json(dataset / "data_report.json") != report
    ):
        raise DataError("dataset report mismatch")
    rejected = read_json(root / "rejections.json")
    if (
        report["rejected_records"] != len(rejected)
        or report["input_records"] != len(rejected) + sum(len(r["observations"]) for r in rows)
        or report["rejection_reasons"] != dict(Counter(x["reason"] for x in rejected))
        or report["cross_split_peptide_overlap"] != 0
    ):
        raise DataError("rejection report mismatch")
    function = normalize_task_type(recipe["task"]) == "pep_function"
    for split in ["train", "val", "test"]:
        selected = [r for r in rows if r["split"] == split]
        fasta = "".join(
            f">{r['sample_id']}\n{r['peptide'] if function else r['peptide'] + 'GGG' + r['partner']}\n"
            for r in selected
        )
        if (dataset / split / "seqs.fasta").read_text() != fasta:
            raise DataError("FASTA/provenance mismatch")
        with (dataset / split / "labels.csv").open(newline="") as handle:
            actual = list(csv.reader(handle))
        expected = [[x["name"] for x in recipe["labels"]] if function else ["label"]] + [
            [str(x) for x in r["labels"]] for r in selected
        ]
        if actual != expected:
            raise DataError("label order/provenance mismatch")
        if not function:
            with (dataset / split / "pairs.csv").open(newline="") as handle:
                pairs = list(csv.DictReader(handle))
            expected = [pair_record(r, recipe["task"]) for r in selected]
            if normalize_task_type(recipe["task"]) == "pep_hlai":
                try:
                    pairs = [canonical_hlai_columns(row) for row in pairs]
                except HLAIInputError as exc:
                    raise DataError(str(exc)) from exc
            if pairs != expected:
                raise DataError("pair order/provenance mismatch")
    with zipfile.ZipFile(root / "dataset.zip") as archive:
        files = file_inventory(dataset)
        if len(archive.namelist()) != len(files) or set(archive.namelist()) != set(files):
            raise DataError("ZIP inventory mismatch")
        if any(digest(archive.read(p)) != h for p, h in files.items()):
            raise DataError("ZIP contents mismatch")
    return report
