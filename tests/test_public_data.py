"""Offline scientific, failure, reproducibility and upload contract tests."""

import copy
import importlib
import io
import itertools
import os
import socket
import subprocess
import urllib.error
import urllib.parse
from pathlib import Path

import pytest

from pepplatform.public_data import DataError, build, fetch, verify
from pepplatform.public_data.adapters import allele, dbaasp, iedb, intact, mic_label
from pepplatform.public_data.common import (
    AA,
    TASKS,
    file_inventory,
    load_recipe,
    new_directory,
    read_json,
    split_for,
    write_json,
)
from pepplatform.public_data.fetch import collect


def definition():
    return {
        "name": "ABP_Ecoli_MIC",
        "target_species": "Escherichia coli",
        "measure": "MIC",
        "unit": "µM",
        "positive_max": 8,
        "negative_min": 64,
    }


def recipe_file(tmp, task):
    r = {"version": 1, "task": task, "source": TASKS[task], "query": {}, "peptide_length": [3, 50]}
    if task == "pep_function":
        r["labels"] = [definition()]
    elif task == "pep_protein_interaction":
        r["query"] = {"query": 'ptype:"peptide"'}
    else:
        r["binding_methods"] = ["purified MHC/direct/fluorescence"]
        mapping = tmp / "pseudo.csv"
        mapping.write_text(
            (
                "hla,pseudoseq\nHLA-A*02:01," + "A" * 30
                if task == "pep_hlai"
                else "hla_pair,pseudoseq\nDRA*01:01|DRB1*01:01," + "A" * 34
            )
            + "\n"
        )
        r["resources"] = {"pseudosequences": str(mapping)}
    path = tmp / "recipe.json"
    write_json(path, r)
    return path


def source_row(task, peptide, label, i):
    if task == "pep_function":
        return {
            "id": i,
            "sequence": peptide,
            "complexity": {"name": "Monomer"},
            "unusualAminoAcids": [],
            "intrachainBonds": [],
            "interchainBonds": [],
            "coordinationBonds": [],
            "articles": [{"id": 1, "pubmed": {"pubmedId": "123"}}],
            "targetActivities": [
                {
                    "id": i,
                    "targetSpecies": {"name": "Escherichia coli"},
                    "activityMeasureValue": "MIC",
                    "unit": {"name": "µM"},
                    "concentration": "4" if label else ">64",
                    "reference": "1",
                }
            ],
        }
    if task in {"pep_hlai", "pep_hlaii"}:
        return {
            "assay_id": i,
            "epitope__name": peptide,
            "epitope__object_type": "Linear peptide",
            "epitope__modified_residues": None,
            "epitope__modifications": None,
            "assay__method": "purified MHC/direct/fluorescence",
            "assay__response_measured": "qualitative binding",
            "assay__qualitative_measurement": "Positive" if label else "Negative",
            "reference__pmid": "123",
            "mhc_restriction__name": "HLA-A*02:01" if task == "pep_hlai" else "HLA-DRB1*01:01",
            "mhc_restriction__class": "I" if task == "pep_hlai" else "II",
        }
    xml = f"""<entry><interactionList><interaction><xref><primaryRef id="EBI-{i}"/></xref>
      <interactionType><xref><primaryRef id="MI:0407"/></xref></interactionType>
      <negative>{str(not label).lower()}</negative><participantList>
      <participant><interactor id="1"><sequence>{peptide}</sequence><interactorType><xref><primaryRef id="MI:0327"/></xref></interactorType></interactor></participant>
      <participant><interactor id="2"><sequence>{"W" * 70}</sequence><interactorType><xref><primaryRef id="MI:0326"/></xref></interactorType></interactor></participant>
      </participantList></interaction></interactionList></entry>"""
    return {
        "ac": f"EBI-{i}",
        "binaryInteractionId": i,
        "negative": not bool(label),
        "typeMIIdentifier": "MI:0407",
        "affectedByMutation": False,
        "expansionMethod": None,
        "publicationPubmedIdentifier": "123",
        "xml25Format": xml,
    }


def records_for(recipe):
    counts = dict.fromkeys(["train", "val", "test"], 0)
    rows = []
    for chars in itertools.product(sorted(AA), repeat=3):
        peptide = "ACDE" + "".join(chars)
        split = split_for(peptide, recipe)
        if counts[split] >= 2:
            continue
        label = counts[split]
        counts[split] += 1
        rows.append(source_row(recipe["task"], peptide, label, len(rows) + 1))
        if sum(counts.values()) == 6:
            return rows
    raise AssertionError("could not create fixture groups")


class FixtureClient:
    def __init__(self, root, recipe, records):
        self.root, self.recipe, self.records = Path(root), recipe, records
        self.requests = []
        (self.root / "responses").mkdir()

    def get(self, url):
        params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        if self.recipe["source"] == "dbaasp" and "/peptides/" in url:
            value = next(x for x in self.records if str(x["id"]) == url.rsplit("/", 1)[1])
        else:
            size = int(params.get("limit", params.get("pageSize", [100]))[0])
            offset = (
                int(params["page"][0]) * size
                if "page" in params
                else int(params.get("offset", [0])[0])
            )
            selected = self.records[offset : offset + size]
            if self.recipe["source"] == "dbaasp":
                value = {
                    "data": [{"id": x["id"]} for x in selected],
                    "totalCount": len(self.records),
                }
            elif self.recipe["source"] == "iedb":
                value = [{"elution_id": x["assay_id"], "mhc_export": [x]} for x in selected]
            else:
                value = {"content": selected, "totalElements": len(self.records)}
        name = f"responses/{len(self.requests):06}.json"
        write_json(self.root / name, value)
        self.requests.append({"url": url, "response": name})
        return value


def snapshot_for(tmp, task, records=None):
    path = recipe_file(tmp, task)
    recipe = load_recipe(path)
    records = records if records is not None else records_for(recipe)
    root = tmp / "snapshot"
    fetch(path, root, client_factory=lambda p: FixtureClient(p, recipe, records))
    return path, root, records


@pytest.mark.parametrize("task", TASKS)
def test_four_tasks_offline_deterministic_and_uploadable(tmp_path, monkeypatch, task):
    path, snapshot, _ = snapshot_for(tmp_path, task)
    manifest = read_json(snapshot / "snapshot.json")
    index = read_json(snapshot / manifest["resources"]["benchmarks"])
    assert index["status"] == "not_available"
    assert index["task"] == task and index["files"] == []

    def denied(*args, **kwargs):
        raise AssertionError("build attempted network access")

    monkeypatch.setattr(socket, "create_connection", denied)
    # Delete original local mappings: build must use the frozen resource.
    (tmp_path / "pseudo.csv").unlink(missing_ok=True)
    a, b = tmp_path / "a", tmp_path / "b"
    report = build(snapshot, path, a)
    assert report["overlap"] == {
        "status": "not_available",
        "independence_claimed": False,
        "homology_checked": False,
        "files": [],
    }
    build(snapshot, path, b)
    assert file_inventory(a) == file_inventory(b)
    assert verify(a) == report
    assert all(x["samples"] == 2 for x in report["splits"].values())
    from pepplatform.web.dataset_reports import build_dataset_report
    from pepplatform.web.datasets import DatasetService
    from pepplatform.web.paths import RuntimePaths

    class Store:
        def upsert_dataset(
            self, id, name, task_type, source, fmt, root, metadata, owner_user_id=None
        ):
            return {
                "id": id,
                "name": name,
                "task_type": task_type,
                "source": source,
                "format": fmt,
                "root_path": str(root),
                "metadata": metadata,
            }

    runtime = RuntimePaths.from_project_root(tmp_path, tmp_path / "web").ensure()
    dataset = DatasetService(Store(), runtime).upload_archive(
        task, "public", "dataset.zip", (a / "dataset.zip").read_bytes()
    )
    uploaded = build_dataset_report(dataset)
    assert not [x for x in uploaded["issues"] if x["severity"] == "error"]
    assert dataset["metadata"]["splits"] == {"train": 2, "val": 2, "test": 2}
    for p in (a / "dataset").rglob("*"):
        if p.is_file():
            assert (
                Path(dataset["root_path"]) / p.relative_to(a / "dataset")
            ).read_bytes() == p.read_bytes()
    # Existing readers: label width derives from the new file, not fixed 15 classes.
    if task == "pep_function":
        from pepplatform.dataset import PeptideDataset

        class Featurizer:
            def encode(self, seq, **kwargs):
                return [1] * len(seq)

        ds = PeptideDataset(
            a / "dataset/train/seqs.fasta", a / "dataset/train/labels.csv", featurizer=Featurizer()
        )
        assert ds.get_num_classes() == 1 and len(ds) == 2


@pytest.mark.parametrize(
    "text,expected",
    [
        ("8", 1),
        ("64", 0),
        ("<=8", 1),
        (">64", 0),
        ("<64", None),
        (">8", None),
        ("1-8", 1),
        ("64-128", 0),
        ("8-64", None),
        ("10-1", None),
        ("nan", None),
        ("0", None),
        ("8 mg/ml", None),
        ("≤8", 1),
        ("64.0", 0),
        ("1e1", None),
        ("-1", None),
    ],
)
def test_mic_intervals(text, expected):
    assert mic_label(text, definition()) == expected


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"sequence": "AcDEF"}, "nonstandard"),
        ({"nTerminus": {"name": "Ac"}}, "modified"),
        ({"targetActivities": []}, "missing"),
        ({"intrachainBonds": [{}]}, "modified"),
    ],
)
def test_function_exclusions(change, reason):
    row = source_row("pep_function", "ACDEF", 1, 1)
    row.update(change)
    with pytest.raises(DataError, match=reason):
        dbaasp(row, {"labels": [definition()]})


def test_conflict_poisoning_prevents_cherry_picking(tmp_path):
    path = recipe_file(tmp_path, "pep_function")
    recipe = load_recipe(path)
    rows = records_for(recipe)
    conflict = copy.deepcopy(rows[0])
    conflict["id"] = 99
    assay = copy.deepcopy(conflict["targetActivities"][0])
    assay["concentration"] = "4" if assay["concentration"] == ">64" else ">64"
    conflict["targetActivities"].append(assay)
    snapshot = tmp_path / "snapshot"
    fetch(
        path,
        snapshot,
        client_factory=lambda p: FixtureClient(p, recipe, rows + [conflict]),
    )
    with pytest.raises(DataError, match="insufficient"):
        build(snapshot, path, tmp_path / "out")
    report = read_json(tmp_path / "out/report.json")
    assert report["rejection_reasons"]["conflicting_assay_labels"] == 1
    assert report["rejection_reasons"]["conflicting_sample_evidence"] == 1
    assert not (tmp_path / "out/dataset.zip").exists()


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("assay__response_measured", "ligand presentation", "not_selected"),
        ("mhc_restriction__name", "HLA-A*99:99", "missing_exact"),
        ("mhc_restriction__class", "II", "wrong_mhc"),
        ("assay__qualitative_measurement", "unknown", "unknown"),
        ("epitope__modifications", "acetylation", "modified"),
    ],
)
def test_iedb_exclusions(field, value, reason):
    row = source_row("pep_hlai", "ACDEFGHIK", 1, 1)
    row[field] = value
    with pytest.raises(DataError, match=reason):
        iedb(
            row,
            {"task": "pep_hlai", "binding_methods": ["purified MHC/direct/fluorescence"]},
            {"HLA-A*02:01": "A" * 30},
        )


def test_hla_exact_identity():
    assert allele("HLA-DPA1*01:03/DPB1*04:01", "pep_hlaii") == "DPA1*01:03|DPB1*04:01"
    assert allele("HLA-DRB1*01:01", "pep_hlaii") == "DRA*01:01|DRB1*01:01"
    for text in ["HLA-A2", "HLA-A*02", "H2-Kb"]:
        with pytest.raises(DataError):
            allele(text, "pep_hlai")


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("negative", None, "outcome"),
        ("typeMIIdentifier", "MI:0915", "direct"),
        ("expansionMethod", "spoke", "expanded"),
        ("xml25Format", None, "sequence_evidence"),
        ("affectedByMutation", True, "mutation"),
        ("ac", "EBI-other", "identity"),
    ],
)
def test_intact_exclusions(field, value, reason):
    row = source_row("pep_protein_interaction", "ACDE", 1, 1)
    row[field] = value
    with pytest.raises(DataError, match=reason):
        intact(row, {})


def test_intact_features_and_xml_outcome():
    row = source_row("pep_protein_interaction", "ACDE", 1, 1)
    row["xml25Format"] = row["xml25Format"].replace(
        "<participant>", "<participant><featureList/>", 1
    )
    with pytest.raises(DataError, match="features"):
        intact(row, {})
    row = source_row("pep_protein_interaction", "ACDE", 1, 1)
    row["negative"] = True
    with pytest.raises(DataError, match="label_mismatch"):
        intact(row, {})


def test_snapshot_and_output_tamper(tmp_path):
    path, snapshot, _ = snapshot_for(tmp_path, "pep_function")
    out = tmp_path / "out"
    build(snapshot, path, out)
    (out / "dataset/train/labels.csv").write_text("label\n1\n")
    with pytest.raises(DataError, match="hash mismatch"):
        verify(out)
    (snapshot / "records.json").write_text("[]")
    with pytest.raises(DataError, match="hash mismatch"):
        build(snapshot, path, tmp_path / "other")
    assert not (tmp_path / "other").exists()


def test_cap_failure_never_publishes_complete_snapshot(tmp_path):
    path = recipe_file(tmp_path, "pep_function")
    r = read_json(path)
    r["max_records"] = 1
    write_json(path, r)
    r = load_recipe(path)
    with pytest.raises(DataError, match="max_records"):
        fetch(
            path,
            tmp_path / "snap",
            client_factory=lambda p: FixtureClient(p, r, records_for(r)),
        )
    assert read_json(tmp_path / "snap/snapshot.json")["complete"] is False
    with pytest.raises(DataError, match="incomplete"):
        build(tmp_path / "snap", path, tmp_path / "out")


def test_schema_change_and_failed_fetch(tmp_path):
    path = recipe_file(tmp_path, "pep_function")

    class Bad(FixtureClient):
        def get(self, url):
            return {"unexpected": []}

    with pytest.raises(DataError, match="schema changed"):
        fetch(
            path,
            tmp_path / "snap",
            client_factory=lambda p: Bad(p, load_recipe(path), []),
        )
    assert read_json(tmp_path / "snap/snapshot.json")["complete"] is False


def test_path_protection_and_overwrite(tmp_path):
    for p in [tmp_path / "experiment_paper/new", tmp_path / "datasets/new"]:
        with pytest.raises(DataError, match="protected"):
            new_directory(p)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").write_text("gitdir: elsewhere")
    with pytest.raises(DataError, match="Git checkout"):
        new_directory(repo / "out")
    new_directory(tmp_path / "out")
    with pytest.raises(DataError, match="already exists"):
        new_directory(tmp_path / "out")


def test_missing_label_is_not_negative_and_width_not_fixed(tmp_path):
    path = recipe_file(tmp_path, "pep_function")
    r = read_json(path)
    second = definition()
    second.update(name="other", target_species="other")
    r["labels"].append(second)
    write_json(path, r)
    r = load_recipe(path)
    fetch(
        path,
        tmp_path / "snap",
        client_factory=lambda p: FixtureClient(p, r, records_for(r)),
    )
    with pytest.raises(DataError, match="insufficient"):
        build(tmp_path / "snap", path, tmp_path / "out")
    assert read_json(tmp_path / "out/report.json")["accepted_samples"] == 0


def test_repeated_pages_fail(tmp_path):
    path = recipe_file(tmp_path, "pep_function")
    r = load_recipe(path)
    r["page_size"] = 1

    class Repeat:
        def get(self, url):
            return {"data": [{"id": 1}], "totalCount": 2} if "?" in url else {"id": 1}

    with pytest.raises(DataError, match="repeated"):
        collect(r, Repeat())


def test_real_source_responses_are_conservatively_classified():
    folder = Path(__file__).parent / "fixtures/public_data"
    d = read_json(folder / "dbaasp.fixture")["record"]
    with pytest.raises(DataError, match="indeterminate"):
        dbaasp(d, {"labels": [definition()]})  # 1.35–21.67 µM spans the active cutoff.
    label = definition()
    label["positive_max"] = 32
    assert dbaasp(d, {"labels": [label]})["labels"] == [1]
    with pytest.raises(DataError, match="features"):
        intact(read_json(folder / "intact.fixture")["record"], {})
    r = read_json(folder / "iedb.fixture")["record"]
    with pytest.raises(DataError, match="not_selected"):
        iedb(r, {"task": "pep_hlai", "binding_methods": [r["assay__method"]]}, {})


def test_pair_reader_retains_original_allele_context(tmp_path):
    from scripts.train_pair_pipeline import read_ppi_csv

    for task in ["pep_hlai", "pep_hlaii"]:
        parent = tmp_path / task
        parent.mkdir()
        recipe, snap, _ = snapshot_for(parent, task)
        build(snap, recipe, parent / "out")
        p = parent / "out/dataset/train/pairs.csv"
        rows, contexts = read_ppi_csv(p, return_contexts=True)
        assert len(rows) == 2
        assert contexts == (
            ["HLA-A*02:01"] * 2 if task == "pep_hlai" else ["DRA*01:01|DRB1*01:01"] * 2
        )
        assert all(len(x[2]) == (30 if task == "pep_hlai" else 34) for x in rows)


def test_http_retry_and_response_capture(tmp_path, monkeypatch):
    module = importlib.import_module("pepplatform.public_data.fetch")
    calls = []

    class Response(io.BytesIO):
        status = 200
        headers = {"ETag": "frozen-v1"}

    def request(req, timeout):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "limit", {"Retry-After": "0"}, None)
        return Response(b'{"ok": true}')

    monkeypatch.setattr(module.urllib.request, "urlopen", request)
    monkeypatch.setattr(module.time, "sleep", lambda n: None)
    client = module.HTTPClient(tmp_path, interval=0)
    assert client.get("https://dbaasp.org/peptides") == {"ok": True}
    assert len(calls) == 2
    assert client.requests[0]["headers"]["etag"] == "frozen-v1"
    assert read_json(tmp_path / "responses/000000.json") == {"ok": True}


def test_timeout_is_bounded(tmp_path, monkeypatch):
    module = importlib.import_module("pepplatform.public_data.fetch")
    calls = []

    def request(*a, **kw):
        calls.append(1)
        raise TimeoutError("offline")

    monkeypatch.setattr(module.urllib.request, "urlopen", request)
    monkeypatch.setattr(module.time, "sleep", lambda n: None)
    with pytest.raises(DataError, match="request failed"):
        module.HTTPClient(tmp_path, attempts=3, interval=0).get("https://dbaasp.org/peptides")
    assert len(calls) == 3


def test_same_peptide_multiple_partners_stays_in_one_split(tmp_path):
    path = recipe_file(tmp_path, "pep_protein_interaction")
    r = load_recipe(path)
    rows = records_for(r)
    other = copy.deepcopy(rows[0])
    other.update(ac="EBI-99", binaryInteractionId=99)
    other["xml25Format"] = (
        other["xml25Format"].replace("EBI-1", "EBI-99").replace("W" * 70, "Y" * 70)
    )
    fetch(
        path,
        tmp_path / "snap",
        client_factory=lambda p: FixtureClient(p, r, rows + [other]),
    )
    build(tmp_path / "snap", path, tmp_path / "out")
    accepted = read_json(tmp_path / "out/accepted.json")
    paired = [x for x in accepted if x["peptide"] == accepted[0]["peptide"]]
    original_peptide = intact(rows[0], r)["peptide"]
    paired = [x for x in accepted if x["peptide"] == original_peptide]
    assert len(paired) == 2 and len({x["split"] for x in paired}) == 1


def reference_index(task):
    from pepplatform.public_data.benchmarks import sample_key
    from pepplatform.public_data.common import digest

    peptide = "ACDEAAA"
    partner = {
        "pep_function": "",
        "pep_protein_interaction": "W" * 70,
        "pep_hlai": "HLA-A*02:01",
        "pep_hlaii": "DRA*01:01|DRB1*01:01",
    }[task]
    return {
        "status": "checked",
        "task": task,
        "files": [
            {
                "path": "reference/train.csv",
                "sha256": digest(b"synthetic reference"),
                "peptides": [digest(peptide.encode())],
                "samples": [sample_key(task, peptide, partner)],
            }
        ],
    }


def test_fetch_without_reference_does_not_discover_a_checkout(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("fetch attempted to discover a source checkout")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "check_output", denied)
    assert not (tmp_path / ".git").exists()
    path, snapshot, _ = snapshot_for(tmp_path, "pep_function")
    assert build(snapshot, path, tmp_path / "out")["overlap"]["status"] == "not_available"


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("legacy_index", [False, True])
def test_explicit_reference_index_is_frozen_and_checked(tmp_path, task, legacy_index):
    path = recipe_file(tmp_path, task)
    index = reference_index(task)
    if legacy_index:
        index["lock_sha256"] = "a" * 64
    reference = tmp_path / "reference.json"
    write_json(reference, index)
    recipe = read_json(path)
    recipe.setdefault("resources", {})["benchmarks"] = reference.name
    write_json(path, recipe)
    recipe = load_recipe(path)
    snapshot = tmp_path / "snapshot"
    fetch(
        path,
        snapshot,
        client_factory=lambda p: FixtureClient(p, recipe, records_for(recipe)),
    )
    manifest = read_json(snapshot / "snapshot.json")
    frozen = snapshot / manifest["resources"]["benchmarks"]
    assert frozen.read_bytes() == reference.read_bytes()
    reference.unlink()
    report = build(snapshot, path, tmp_path / "out")
    assert report["overlap"]["status"] == "checked"
    assert not report["overlap"]["independence_claimed"]
    assert not report["overlap"]["homology_checked"]
    overlap = report["overlap"]["files"][0]
    assert overlap["exact_peptide_overlap"] == overlap["exact_sample_overlap"] == 1
    assert verify(tmp_path / "out") == report
    frozen.write_text("{}")
    with pytest.raises(DataError, match="hash mismatch"):
        build(snapshot, path, tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()


@pytest.mark.parametrize("task", TASKS)
def test_explicit_reference_index_requires_the_same_task(tmp_path, task):
    path = recipe_file(tmp_path, task)
    reference = tmp_path / "reference.json"
    other_task = next(t for t in TASKS if t != task)
    write_json(reference, reference_index(other_task))
    recipe = read_json(path)
    recipe.setdefault("resources", {})["benchmarks"] = str(reference)
    write_json(path, recipe)
    recipe = load_recipe(path)
    fetch(
        path,
        tmp_path / "snapshot",
        client_factory=lambda p: FixtureClient(p, recipe, records_for(recipe)),
    )
    with pytest.raises(DataError, match="invalid benchmark index"):
        build(tmp_path / "snapshot", path, tmp_path / "out")
    assert not (tmp_path / "out/dataset.zip").exists()


def test_supplements_require_matching_experimental_definition(tmp_path):
    path = recipe_file(tmp_path, "pep_function")
    r = load_recipe(path)
    supplemental = tmp_path / "manual.json"
    write_json(
        supplemental,
        [
            {
                "source_id": "manual-1",
                "peptide": "ACDEF",
                "labels": [0],
                "reference": "PMID:123",
                "evidence_type": "predicted",
                "unmodified": True,
                "task": "pep_function",
                "label_definition": r["labels"],
            }
        ],
    )
    r["resources"]["supplements"] = str(supplemental)
    write_json(path, r)
    fetch(
        path,
        tmp_path / "snap",
        client_factory=lambda p: FixtureClient(p, r, records_for(r)),
    )
    build(tmp_path / "snap", path, tmp_path / "out")
    report = read_json(tmp_path / "out/report.json")
    assert report["rejection_reasons"]["supplement_requires_explicit_experimental_provenance"] == 1


def test_duplicate_evidence_merges_and_conflicting_pair_is_excluded(tmp_path):
    path = recipe_file(tmp_path, "pep_hlai")
    r = load_recipe(path)
    rows = records_for(r)
    duplicate = copy.deepcopy(rows[0])
    duplicate["assay_id"] = 99
    fetch(
        path,
        tmp_path / "snap",
        client_factory=lambda p: FixtureClient(p, r, rows + [duplicate]),
    )
    report = build(tmp_path / "snap", path, tmp_path / "out")
    assert report["input_records"] == 7 and report["accepted_samples"] == 6
    assert max(len(x["observations"]) for x in read_json(tmp_path / "out/accepted.json")) == 2
    duplicate["assay__qualitative_measurement"] = (
        "Positive" if duplicate["assay__qualitative_measurement"] == "Negative" else "Negative"
    )
    fetch(
        path,
        tmp_path / "snap2",
        client_factory=lambda p: FixtureClient(p, r, rows + [duplicate]),
    )
    with pytest.raises(DataError, match="insufficient"):
        build(tmp_path / "snap2", path, tmp_path / "out2")
    assert (
        read_json(tmp_path / "out2/report.json")["rejection_reasons"]["conflicting_sample_evidence"]
        == 2
    )


@pytest.mark.parametrize("task", list(TASKS))
def test_frozen_builder_replays_outside_repository(tmp_path, task):
    import sys

    path, snapshot, _ = snapshot_for(tmp_path, task)
    build(snapshot, path, tmp_path / "original")
    code = "from builder import build; import sys; build(*sys.argv[1:])"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(tmp_path / "original"))
    subprocess.run(
        [sys.executable, "-c", code, str(snapshot), str(path), str(tmp_path / "replayed")],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert file_inventory(tmp_path / "original") == file_inventory(tmp_path / "replayed")
    assert verify(tmp_path / "original")["ready"]
    with pytest.raises(DataError, match="nested"):
        build(snapshot, path, snapshot / "do-not-write")


def test_schema_drift_cannot_publish_partial_cohort(tmp_path):
    path = recipe_file(tmp_path, "pep_function")
    r = load_recipe(path)
    rows = records_for(r)
    bad = copy.deepcopy(rows[0])
    bad["id"] = 99
    del bad["complexity"]
    fetch(
        path,
        tmp_path / "snap",
        client_factory=lambda p: FixtureClient(p, r, rows + [bad]),
    )
    with pytest.raises(DataError, match="source_schema_missing"):
        build(tmp_path / "snap", path, tmp_path / "out")
    assert not (tmp_path / "out/dataset.zip").exists()
