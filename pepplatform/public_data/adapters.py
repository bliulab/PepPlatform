"""Conservative, explicit scientific mappings; no label imputation."""

from __future__ import annotations
from pepplatform.hlai_input import hla_allele, HLAIInputError
from pepplatform.task_compat import LEGACY_HLAI_COLUMN
from pepplatform.task_compat import normalize_task_type


import csv
import re
from decimal import Decimal
from xml.etree import ElementTree as ET

from .common import AA, DataError, sequence


def name(value):
    return value.get("name", "") if isinstance(value, dict) else (value or "")


def mic_label(text, definition):
    """Only label an interval when *every* possible value implies that label."""
    n = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    text = str(text).strip().replace("≤", "<=").replace("≥", ">=")
    m = re.fullmatch(rf"(<=|>=|<|>|=)?\s*({n})(?:\s*-\s*({n}))?", text)
    if not m:
        return None
    op, first, last = m.groups()
    low = high = Decimal(first)
    if low <= 0:
        return None
    if last:
        if op:
            return None
        high = Decimal(last)
        if high < low:
            return None
    if op in {"<", "<="}:
        low = Decimal(0)
    if op in {">", ">="}:
        high = Decimal("Infinity")
    if high <= Decimal(str(definition["positive_max"])):
        return 1
    if low >= Decimal(str(definition["negative_min"])):
        return 0
    return None


def dbaasp(row, recipe):
    for field in [
        "sequence",
        "complexity",
        "unusualAminoAcids",
        "intrachainBonds",
        "interchainBonds",
        "coordinationBonds",
        "targetActivities",
        "articles",
    ]:
        if field not in row:
            raise DataError(f"source_schema_missing:{field}")
    if name(row["complexity"]).lower() != "monomer" or any(
        row.get(k)
        for k in [
            "unusualAminoAcids",
            "intrachainBonds",
            "interchainBonds",
            "coordinationBonds",
            "nTerminus",
            "cTerminus",
            "NTerminus",
            "CTerminus",
            "synergies",
        ]
    ):
        raise DataError("modified_or_complex_peptide")
    peptide = sequence(row["sequence"])
    labels, evidence = [], []
    for definition in recipe["labels"]:
        votes = set()
        assays = []
        for activity in row["targetActivities"]:
            if (
                name(activity.get("targetSpecies")) != definition["target_species"]
                or activity.get("activityMeasureValue") != definition["measure"]
                or name(activity.get("unit")) != definition["unit"]
            ):
                continue
            label = mic_label(activity.get("concentration", ""), definition)
            refs = str(activity.get("reference", ""))
            refs_valid = re.fullmatch(r"\d+(?:\s*[,;]\s*\d+)*", refs) and all(
                1 <= int(ref) <= len(row["articles"]) for ref in re.findall(r"\d+", refs)
            )
            if label is not None and refs_valid:
                votes.add(label)
                assays.append(activity)
        if len(votes) > 1:
            raise DataError("conflicting_assay_labels")
        if not votes:
            raise DataError("missing_or_indeterminate_label:" + definition["name"])
        labels.append(votes.pop())
        evidence.append(
            {"label": definition["name"], "assays": assays, "articles": row["articles"]}
        )
    return {
        "peptide": peptide,
        "partner": "",
        "partner_id": "",
        "labels": labels,
        "evidence": evidence,
    }


def allele(value, task):
    task = normalize_task_type(task)
    value = str(value).strip().replace("HLA-", "")
    if normalize_task_type(task) == "pep_hlai":
        if not re.fullmatch(r"[ABC]\*\d{2,3}:\d{2,3}(?::\d{2,3})*", value):
            raise DataError("ambiguous_or_unsupported_hla_i")
        return "HLA-" + value
    value = value.replace("/", "|")
    # Established DR convention; this is not nearest-allele substitution.
    if re.fullmatch(r"DRB[1345]\*\d{2,3}:\d{2,3}(?::\d{2,3})*", value):
        value = "DRA*01:01|" + value
    pattern = r"(D[PRQ]A[12]?\*\d{2,3}:\d{2,3}(?::\d{2,3})*)\|(D[PRQ]B[12345]?\*\d{2,3}:\d{2,3}(?::\d{2,3})*)"
    m = re.fullmatch(pattern, value)
    if not m or m[1][1] != m[2][1]:
        raise DataError("ambiguous_or_unsupported_hla_ii")
    return value


def load_pseudosequences(path, task):
    task = normalize_task_type(task)
    result = {}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        key = "hla" if normalize_task_type(task) == "pep_hlai" else "hla_pair"
        fields = set(reader.fieldnames or [])
        if "pseudoseq" not in fields or (
            key not in fields and not (key == "hla" and LEGACY_HLAI_COLUMN in fields)
        ):
            raise DataError(f"pseudosequences CSV requires {key},pseudoseq")
        for row in reader:
            if row.get("status", "ok").lower() not in {"", "ok"}:
                continue
            try:
                raw_identity = hla_allele(row) if key == "hla" else row[key]
            except HLAIInputError as exc:
                raise DataError(str(exc)) from exc
            try:
                identity = allele(raw_identity, task)
            except DataError:
                continue  # Resource can include non-human alleles outside v1 scope.
            seq = row["pseudoseq"].strip()
            if len(seq) != (30 if normalize_task_type(task) == "pep_hlai" else 34) or not set(
                seq
            ) <= AA | {"X"}:
                raise DataError("invalid_pseudosequence")
            if identity in result and result[identity] != seq:
                raise DataError("conflicting_pseudosequence_mapping")
            result[identity] = seq
    return result


def iedb(row, recipe, mapping):
    required = [
        "epitope__name",
        "epitope__object_type",
        "epitope__modified_residues",
        "epitope__modifications",
        "assay__method",
        "assay__response_measured",
        "assay__qualitative_measurement",
        "mhc_restriction__name",
        "mhc_restriction__class",
    ]
    for field in required:
        if field not in row:
            raise DataError(f"source_schema_missing:{field}")
    wanted = "I" if normalize_task_type(recipe["task"]) == "pep_hlai" else "II"
    if row["mhc_restriction__class"] != wanted:
        raise DataError("wrong_mhc_class")
    if (
        row["epitope__object_type"] != "Linear peptide"
        or row["epitope__modified_residues"]
        or row["epitope__modifications"]
    ):
        raise DataError("modified_or_non_linear_epitope")
    measure = str(row["assay__response_measured"]).lower()
    if (
        row["assay__method"] not in recipe["binding_methods"]
        or any(s in measure for s in ["presentation", "elution"])
        or not any(s in measure for s in ["binding", "ic50", "ec50", "dissociation constant"])
    ):
        raise DataError("not_selected_binding_assay")
    qualitative = row["assay__qualitative_measurement"]
    if qualitative not in {
        "Positive",
        "Positive-Low",
        "Positive-Intermediate",
        "Positive-High",
        "Negative",
    }:
        raise DataError("unknown_qualitative_label")
    if not row.get("reference__pmid") and not row.get("reference__iedb_iri"):
        raise DataError("missing_reference_evidence")
    identity = allele(row["mhc_restriction__name"], recipe["task"])
    if identity not in mapping:
        raise DataError("missing_exact_pseudosequence:" + identity)
    return {
        "peptide": sequence(row["epitope__name"]),
        "partner": mapping[identity],
        "partner_id": identity,
        "labels": [int(qualitative != "Negative")],
        "evidence": [row],
    }


def intact(row, recipe):
    """Read the actual participants from the source's PSI-MI XML evidence."""
    if row.get("typeMIIdentifier") != "MI:0407":  # direct interaction
        raise DataError("not_direct_interaction_evidence")
    if type(row.get("negative")) is not bool:
        raise DataError("missing_explicit_interaction_outcome")
    if (
        row.get("expansionMethod") not in {None, "", "-"}
        or row.get("affectedByMutation") is not False
    ):
        raise DataError("expanded_complex_or_mutation")
    xml = row.get("xml25Format") or row.get("xml30Format")
    if not isinstance(xml, str) or not xml.startswith("<"):
        raise DataError("missing_participant_sequence_evidence")
    if "<!DOCTYPE" in xml.upper() or "<!ENTITY" in xml.upper():
        raise DataError("unsupported_xml_declaration")
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise DataError("invalid_interaction_xml") from exc
    for node in root.iter():
        node.tag = node.tag.rsplit("}", 1)[-1]
    interactions = list(root.iter("interaction"))
    if len(interactions) != 1:
        raise DataError("ambiguous_interaction_xml")
    inter = interactions[0]
    identifiers = {x.get("id") for x in inter.findall("./xref/primaryRef")}
    if row.get("ac") not in identifiers:
        raise DataError("interaction_xml_identity_mismatch")
    xml_negative = inter.findtext("negative", "false").strip()
    if (
        xml_negative not in {"true", "false", "1", "0"}
        or (xml_negative in {"true", "1"}) != row["negative"]
    ):
        raise DataError("interaction_xml_label_mismatch")
    if "MI:0407" not in {x.get("id") for x in inter.findall("./interactionType/xref/primaryRef")}:
        raise DataError("interaction_xml_type_mismatch")
    participants = inter.findall("./participantList/participant")
    if len(participants) != 2:
        raise DataError("non_binary_interaction")
    interactors = {x.get("id"): x for x in root.iter("interactor")}
    parsed = []
    for participant in participants:
        if participant.find("featureList") is not None:
            raise DataError("participant_features_require_manual_review")
        molecule = participant.find("interactor")
        if molecule is None:
            molecule = interactors.get(participant.findtext("interactorRef"))
        if molecule is None:
            raise DataError("missing_interactor")
        types = {x.get("id") for x in molecule.findall("./interactorType/xref/primaryRef")}
        kind = "peptide" if "MI:0327" in types else "protein" if "MI:0326" in types else "unknown"
        parsed.append((kind, sequence(molecule.findtext("sequence")), molecule.get("id")))
    if sorted(x[0] for x in parsed) != ["peptide", "protein"]:
        raise DataError("not_explicit_peptide_protein_pair")
    peptide = next(x for x in parsed if x[0] == "peptide")
    protein = next(x for x in parsed if x[0] == "protein")
    if not row.get("publicationPubmedIdentifier") and not row.get("publicationIdentifiers"):
        raise DataError("missing_publication_evidence")
    return {
        "peptide": peptide[1],
        "partner": protein[1],
        "partner_id": protein[1],
        "labels": [int(not row["negative"])],
        "evidence": [row],
    }
