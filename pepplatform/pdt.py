"""Paper-faithful physicochemical distance transformation (PDT).

The implementation follows Liu et al. (2012),
doi:10.1371/journal.pone.0046633, including the 531 AAIndex1 properties
published with the paper and the authors' property-major feature ordering.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np


PDT_ENCODERS = frozenset({"pdt"})
PDT_ENCODER_CONTRACT_VERSION = 1
PDT_DEFAULT_LAG = 8
PDT_DEFAULT_MAX_RAW_LENGTH = 100_000
PDT_MAX_RAW_LENGTH = 100_000
PDT_PROPERTY_BLOCK_SIZE = 32

PDT_AMINO_ACID_ORDER = tuple("ARNDCQEGHILKMFPSTWYV")
PDT_PROPERTY_COUNT = 531
PDT_SOURCE_DOI = "10.1371/journal.pone.0046633"
PDT_SOURCE_TITLE = (
    "Using Amino Acid Physicochemical Distance Transformation for Fast "
    "Protein Remote Homology Detection"
)
PDT_SUPPLEMENT_TEXT_S1_SHA256 = "6b473d7eb682c68ed9ebbf1214834336d27b0dee93cc6f416dd9102eb9a3bfd9"
PDT_SUPPLEMENT_TEXT_S2_RAR_SHA256 = (
    "79f285c9d56e557b62ed52af2d07b7418de33be598cd844929dc8cfbef8a99e9"
)
PDT_REFERENCE_SOURCE_CODE_SHA256 = (
    "0463f38f50fcde24204a7bc8c1b53b8a380f1039c255e562221af762b16bb9f5"
)
PDT_REFERENCE_NUMERIC_TABLE_SHA256 = (
    "ba50e351ceee7d9551ad5e2feca48bc7dd406a60b4a46c31bb1decc00c98c78b"
)
PDT_DATA_SNAPSHOT_SHA256 = "84490233ec4f98885d7e29edeb489001542fefd1f6dca3826d4f574f3ae6b01f"

_PDT_DATA_PATH = Path(__file__).with_name("pdt_aaindex1.tsv")
_AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(PDT_AMINO_ACID_ORDER)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_property_snapshot() -> tuple[tuple[str, ...], np.ndarray]:
    if not _PDT_DATA_PATH.is_file():
        raise RuntimeError(f"PDT AAIndex1 snapshot is missing: {_PDT_DATA_PATH}")
    actual_sha256 = _sha256_file(_PDT_DATA_PATH)
    if actual_sha256 != PDT_DATA_SNAPSHOT_SHA256:
        raise RuntimeError(
            "PDT AAIndex1 snapshot checksum mismatch: "
            f"expected {PDT_DATA_SNAPSHOT_SHA256}, got {actual_sha256}"
        )

    expected_header = ("accession", *PDT_AMINO_ACID_ORDER)
    accessions: list[str] = []
    values: list[list[float]] = []
    with _PDT_DATA_PATH.open("r", encoding="ascii", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise RuntimeError("PDT AAIndex1 snapshot is empty") from exc
        if header != expected_header:
            raise RuntimeError("PDT AAIndex1 snapshot has an unsupported amino-acid order")
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(expected_header):
                raise RuntimeError(
                    f"PDT AAIndex1 snapshot row {line_number} must have "
                    f"{len(expected_header)} fields"
                )
            accession = row[0].strip()
            if not accession:
                raise RuntimeError(f"PDT AAIndex1 snapshot row {line_number} has no accession")
            try:
                property_values = [float(value) for value in row[1:]]
            except ValueError as exc:
                raise RuntimeError(
                    f"PDT AAIndex1 snapshot row {line_number} is not numeric"
                ) from exc
            accessions.append(accession)
            values.append(property_values)

    if len(accessions) != PDT_PROPERTY_COUNT:
        raise RuntimeError(
            f"PDT AAIndex1 snapshot must contain {PDT_PROPERTY_COUNT} properties, "
            f"got {len(accessions)}"
        )
    if len(set(accessions)) != len(accessions):
        raise RuntimeError("PDT AAIndex1 snapshot contains duplicate accessions")
    table = np.asarray(values, dtype=np.float64)
    expected_shape = (PDT_PROPERTY_COUNT, len(PDT_AMINO_ACID_ORDER))
    if table.shape != expected_shape:
        raise RuntimeError(
            f"PDT AAIndex1 table must have shape {expected_shape}, got {table.shape}"
        )
    if not np.isfinite(table).all():
        raise RuntimeError("PDT AAIndex1 table must contain only finite values")
    if np.any(table.std(axis=1, ddof=0) == 0.0):
        raise RuntimeError("PDT AAIndex1 properties must have non-zero variance")
    return tuple(accessions), table


def standardize_pdt_property_table(raw_table) -> np.ndarray:
    """Z-score every AAIndex property across the 20 standard residues."""

    try:
        table = np.asarray(raw_table, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("PDT property table must be numeric") from exc
    expected_shape = (PDT_PROPERTY_COUNT, len(PDT_AMINO_ACID_ORDER))
    if table.shape != expected_shape:
        raise ValueError(f"PDT property table must have shape {expected_shape}, got {table.shape}")
    if not np.isfinite(table).all():
        raise ValueError("PDT property table must contain only finite values")
    standard_deviation = table.std(axis=1, ddof=0)
    if np.any(standard_deviation == 0.0):
        raise ValueError("PDT property table rows must have non-zero variance")
    return (table - table.mean(axis=1, keepdims=True)) / standard_deviation[:, None]


def _property_table_sha256(table: np.ndarray, *, normalized: bool) -> str:
    payload = {
        "amino_acid_order": list(PDT_AMINO_ACID_ORDER),
        "property_accessions": list(PDT_PROPERTY_ACCESSIONS),
        "normalized": bool(normalized),
        "values": np.asarray(table, dtype=np.float64).tolist(),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


PDT_PROPERTY_ACCESSIONS, PDT_RAW_PROPERTY_TABLE = _load_property_snapshot()
PDT_NORMALIZED_PROPERTY_TABLE = standardize_pdt_property_table(PDT_RAW_PROPERTY_TABLE)
PDT_RAW_PROPERTY_TABLE_SHA256 = _property_table_sha256(
    PDT_RAW_PROPERTY_TABLE,
    normalized=False,
)
PDT_NORMALIZED_PROPERTY_TABLE_SHA256 = _property_table_sha256(
    PDT_NORMALIZED_PROPERTY_TABLE,
    normalized=True,
)
PDT_RAW_PROPERTY_TABLE.setflags(write=False)
PDT_NORMALIZED_PROPERTY_TABLE.setflags(write=False)


def validate_pdt_lag(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("PDT lag must be a positive integer")
    value = int(value)
    if value < 1:
        raise ValueError("PDT lag must be a positive integer")
    return value


def parse_pdt_lag(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("PDT lag must be an integer") from exc
    try:
        return validate_pdt_lag(parsed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def validate_pdt_max_raw_length(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("pdt_max_raw_length must be a positive integer")
    value = int(value)
    if value < 1 or value > PDT_MAX_RAW_LENGTH:
        raise ValueError(f"pdt_max_raw_length must be in [1,{PDT_MAX_RAW_LENGTH}], got {value}")
    return value


def parse_pdt_max_raw_length(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("pdt_max_raw_length must be an integer") from exc
    try:
        return validate_pdt_max_raw_length(parsed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def normalize_pdt_sequence(sequence, *, max_raw_length: int) -> str:
    if not isinstance(sequence, str):
        raise ValueError(f"PDT requires a sequence string, got {type(sequence).__name__}")
    limit = validate_pdt_max_raw_length(max_raw_length)
    normalized = sequence.strip().upper()
    if not normalized:
        raise ValueError("PDT requires a non-empty sequence")
    if any(character.isspace() for character in normalized):
        raise ValueError("PDT sequence must not contain internal whitespace")
    if len(normalized) > limit:
        raise ValueError(f"sequence length {len(normalized)} exceeds pdt_max_raw_length={limit}")
    invalid = sorted(set(normalized).difference(_AA_TO_INDEX))
    if invalid:
        raise ValueError(
            f"PDT sequence contains non-standard amino acid residue(s): {', '.join(invalid)}"
        )
    return normalized


def pdt_feature_names(lag: int) -> tuple[str, ...]:
    lag = validate_pdt_lag(lag)
    return tuple(
        f"pdt_{accession}_lag{distance}"
        for accession in PDT_PROPERTY_ACCESSIONS
        for distance in range(1, lag + 1)
    )


def pdt_feature_dimension(lag: int) -> int:
    return PDT_PROPERTY_COUNT * validate_pdt_lag(lag)


def compute_pdt_features(
    sequence,
    *,
    lag: int = PDT_DEFAULT_LAG,
    max_raw_length: int = PDT_DEFAULT_MAX_RAW_LENGTH,
    property_table=PDT_NORMALIZED_PROPERTY_TABLE,
) -> np.ndarray:
    lag = validate_pdt_lag(lag)
    sequence = normalize_pdt_sequence(
        sequence,
        max_raw_length=max_raw_length,
    )
    if len(sequence) <= lag:
        raise ValueError(
            f"PDT requires sequence length > lag; got length={len(sequence)}, lag={lag}"
        )

    try:
        table = np.asarray(property_table, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("normalized PDT property table must be numeric") from exc
    expected_shape = (PDT_PROPERTY_COUNT, len(PDT_AMINO_ACID_ORDER))
    if table.shape != expected_shape:
        raise ValueError(
            f"normalized PDT property table must have shape {expected_shape}, got {table.shape}"
        )
    if not np.isfinite(table).all():
        raise ValueError("normalized PDT property table must contain only finite values")

    residue_indices = np.fromiter(
        (_AA_TO_INDEX[amino_acid] for amino_acid in sequence),
        dtype=np.intp,
        count=len(sequence),
    )
    output = np.empty((PDT_PROPERTY_COUNT, lag), dtype=np.float64)
    for start in range(0, PDT_PROPERTY_COUNT, PDT_PROPERTY_BLOCK_SIZE):
        stop = min(start + PDT_PROPERTY_BLOCK_SIZE, PDT_PROPERTY_COUNT)
        values = table[start:stop, residue_indices]
        for distance in range(1, lag + 1):
            difference = values[:, :-distance] - values[:, distance:]
            np.square(difference, out=difference)
            output[start:stop, distance - 1] = np.sum(difference, axis=1, dtype=np.float64) / (
                len(sequence) - distance
            )
    return output.reshape(-1)


def build_pdt_encoder_contract(
    *,
    lag: int,
    max_raw_length: int,
) -> dict:
    lag = validate_pdt_lag(lag)
    max_raw_length = validate_pdt_max_raw_length(max_raw_length)
    return {
        "contract_version": PDT_ENCODER_CONTRACT_VERSION,
        "encoder": "pdt",
        "lag": lag,
        "method_source_doi": PDT_SOURCE_DOI,
        "method_source_title": PDT_SOURCE_TITLE,
        "supplement_text_s1_sha256": PDT_SUPPLEMENT_TEXT_S1_SHA256,
        "supplement_text_s2_rar_sha256": PDT_SUPPLEMENT_TEXT_S2_RAR_SHA256,
        "reference_source_code_sha256": PDT_REFERENCE_SOURCE_CODE_SHA256,
        "reference_numeric_table_sha256": PDT_REFERENCE_NUMERIC_TABLE_SHA256,
        "data_snapshot_sha256": PDT_DATA_SNAPSHOT_SHA256,
        "raw_property_table_sha256": PDT_RAW_PROPERTY_TABLE_SHA256,
        "normalized_property_table_sha256": (PDT_NORMALIZED_PROPERTY_TABLE_SHA256),
        "aaindex_property_count": PDT_PROPERTY_COUNT,
        "amino_acid_order": list(PDT_AMINO_ACID_ORDER),
        "property_normalization": {
            "formula": "(P[j,a]-mean_over_20_AA[j])/population_sd_over_20_AA[j]",
            "ddof": 0,
            "dtype": "float64",
        },
        "distance_formula": ("sum_i((I[j,A_i]-I[j,A_i+lag])^2)/(sequence_length-lag)"),
        "feature_ordering": "aaindex_property_then_lag_ascending",
        "full_length_policy": "complete_raw_sequence_no_tokenizer_truncation",
        "pdt_max_raw_length": max_raw_length,
        "calculation_dtype": "float64",
        "output_dtype": "float32",
        "output_dimension": pdt_feature_dimension(lag),
    }


def validate_pdt_encoder_contract(
    contract,
    *,
    lag: int,
    max_raw_length: int,
) -> dict:
    if not isinstance(contract, dict):
        raise ValueError("pdt_encoder_contract must be an object")
    expected = build_pdt_encoder_contract(
        lag=lag,
        max_raw_length=max_raw_length,
    )
    if contract != expected:
        raise ValueError(
            "pdt_encoder_contract does not match the supported Liu et al. PDT v1 contract"
        )
    return expected
