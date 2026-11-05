import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

import pepplatform.profile_generation as profile_generation
from pepplatform.encoders import (
    Blosum62Encoder,
    PSFMBlosum62HybridEncoder,
    PSSMBlosum62HybridEncoder,
    PSSMEncoder,
    PSFMEncoder,
)
from pepplatform.featurizers import ComposedFeaturizer
from pepplatform.models import (
    FeatureGRU,
    FeatureLSTM,
    FeatureRNN,
    FeatureTextCNN,
    FeatureTransformer,
    ResidueGraphClassifier,
    ResidueModelInput,
)
from pepplatform.profile_generation import (
    HIT_SUMMARY_OUTFMT,
    PROFILE_FALLBACK_BLOSUM62_HYBRID,
    PROFILE_FALLBACK_NONE,
    ProfileGenerationConfig,
    build_profile_caches,
    build_psiblast_command,
    index_subject_fasta,
    parse_hit_summary,
)
from pepplatform.sequence_profiles import (
    AA_ORDER,
    BLOSUM62_AMBIGUOUS_ROWS,
    BLOSUM62_BACKGROUND_SHA256,
    BLOSUM62_HALF_BIT_SCORES,
    BLOSUM62_MATRIX_SHA256,
    PROFILE_SCIENTIFIC_REFERENCES,
    PROFILE_HIT_SUMMARY_FIELDS,
    PROFILE_QUERY_COVERAGE_FORMULA,
    PSFM_ASCII_CROSSCHECK_TOLERANCE_PERCENTAGE_POINTS,
    PSFM_NAMING_REFERENCE,
    PSFM_HYBRID_SEMANTICS,
    PSFM_SEMANTICS,
    PSSM_HYBRID_SEMANTICS,
    TARGET_FREQUENCY_HYBRID_SEMANTICS,
    TARGET_FREQUENCY_STRICT_SEMANTICS,
    FEATURE_PSFM,
    FEATURE_PSFM_BLOSUM62_HYBRID,
    FEATURE_ACC_PSSM_BLOSUM62_HYBRID,
    FEATURE_AC_PSSM_BLOSUM62_HYBRID,
    FEATURE_CC_PSSM_BLOSUM62_HYBRID,
    FEATURE_PSSM,
    FEATURE_PSSM_BLOSUM62_HYBRID,
    FEATURE_TARGET_FREQUENCY,
    PSSM_HYBRID_FEATURE_TYPES,
    PSFM_HYBRID_FEATURE_TYPES,
    NCBISTDAA_CODES,
    PROFILE_CONTRACT_VERSION,
    PROFILE_SCHEMA,
    ProfileStore,
    SOURCE_BLOSUM62_FALLBACK,
    SOURCE_BLOSUM62_PSFM_FALLBACK,
    SOURCE_BLOSUM62_TARGET_FREQUENCY,
    SOURCE_PSI_BLAST_PSSM,
    SOURCE_PSI_BLAST_WEIGHTED_OBSERVED,
    SequenceProfileError,
    atomic_save_profile_matrix,
    atomic_write_json,
    blosum62_contract_metadata,
    blosum62_row,
    blosum62_score_profile,
    blosum62_target_frequency_profile,
    blosum62_token_mean,
    new_profile_manifest,
    parse_ascii_pssm,
    parse_psiblast_psfm,
    profile_audit_summary,
    profile_sample_metadata,
    normalize_psfm_sequence,
    sha256_file,
)


def _provenance(feature_type, *, fallback_policy):
    config = ProfileGenerationConfig()
    return {
        "profile_contract_version": PROFILE_CONTRACT_VERSION,
        "algorithm": "NCBI PSI-BLAST",
        "algorithm_lineage": "Altschul_et_al_1997",
        "implementation": "NCBI BLAST+",
        "blast_version": "psiblast: test",
        "makeblastdb_version": "makeblastdb: test",
        "scientific_parameters": config.scientific_parameters(),
        "runtime_parameters": config.runtime_parameters(),
        "database_scope": "train_only",
        "subject_fasta_sha256": "a" * 64,
        "subject_fasta_artifact": "subjects/test.fasta",
        "database": {"train_fasta_sha256": "b" * 64},
        "self_exclusion": ("all subject IDs with an identical normalized query sequence"),
        "sequence_normalization": (
            {
                "uppercase": True,
                "strip_whitespace": True,
                "alphabet": AA_ORDER,
                "ambiguous_residues": "error",
                "non_letters": "error",
            }
            if feature_type == FEATURE_PSFM
            else {
                "uppercase": True,
                "strip_whitespace": True,
                "J_U_O_to_X": True,
                "B_Z_X_preserved": True,
                "non_letters": "error",
            }
        ),
        "blosum62_matrix_sha256": BLOSUM62_MATRIX_SHA256,
        "blosum62_background_sha256": BLOSUM62_BACKGROUND_SHA256,
        "scientific_references": [dict(reference) for reference in PROFILE_SCIENTIFIC_REFERENCES],
        "quality_summary": {
            "outfmt_fields": list(PROFILE_HIT_SUMMARY_FIELDS),
            "query_coverage": PROFILE_QUERY_COVERAGE_FORMULA,
            "iteration_scope": "final_completed_iteration",
            "hit_count_unit": "unique_subject",
        },
        "profile_semantics": (
            {
                **PSFM_SEMANTICS,
                "aa_order": AA_ORDER,
                "scientific_reference": dict(PSFM_NAMING_REFERENCE),
            }
            if feature_type == FEATURE_PSFM
            else {
                **PSFM_HYBRID_SEMANTICS,
                "aa_order": AA_ORDER,
                "scientific_reference": dict(PSFM_NAMING_REFERENCE),
            }
            if feature_type in PSFM_HYBRID_FEATURE_TYPES
            else PSSM_HYBRID_SEMANTICS[feature_type]
            if feature_type in PSSM_HYBRID_FEATURE_TYPES
            else TARGET_FREQUENCY_HYBRID_SEMANTICS
            if (
                feature_type == FEATURE_TARGET_FREQUENCY
                and fallback_policy == "blosum62_target_frequency"
            )
            else TARGET_FREQUENCY_STRICT_SEMANTICS
            if feature_type == FEATURE_TARGET_FREQUENCY
            else {
                "method_name": feature_type,
                "matrix_layout": "query_position_by_amino_acid",
            }
        ),
        "audit_artifact": "profile_audit.json",
        "audit_sha256": "d" * 64,
        "audit_summary": {},
        "generation_fingerprint": "c" * 64,
        "feature_type": feature_type,
        "fallback_policy": fallback_policy,
    }


def _write_profile_store(
    tmp_path,
    *,
    feature_type,
    sequence,
    matrix,
    source,
    fallback=False,
    fallback_reason=None,
    nonself_hit_count=None,
    zero_positions=(),
):
    root = tmp_path / feature_type
    matrix_path = root / "matrices" / "sample.npz"
    matrix_sha = atomic_save_profile_matrix(matrix_path, matrix)
    if nonself_hit_count is None:
        nonself_hit_count = 0 if fallback else 1
    zero_hits = nonself_hit_count == 0
    if fallback_reason is None:
        fallback_reason = "profile_output_missing" if fallback else ""
    sample = profile_sample_metadata(
        path=Path("matrices") / "sample.npz",
        matrix=np.asarray(matrix),
        sequence=sequence,
        matrix_sha256=matrix_sha,
        profile_source=source,
        fallback=fallback,
        fallback_reason=fallback_reason,
        zero_observation_positions=tuple(zero_positions),
        quality={
            "nonself_hit_count": nonself_hit_count,
            "max_query_coverage": 0.0 if zero_hits else 1.0,
            "max_identity": 0.0 if zero_hits else 75.0,
            "min_evalue": None if zero_hits else 1e-20,
            "zero_nonself_hits": zero_hits,
            "final_iteration": 1,
            "final_iteration_hit_rows": 0 if zero_hits else 1,
            "blast_warnings": [],
        },
    )
    subject_path = root / "subjects" / "test.fasta"
    subject_path.parent.mkdir(parents=True, exist_ok=True)
    subject_path.write_text(f">TRAIN_000000\n{sequence}\n", encoding="ascii")
    provenance = _provenance(
        feature_type,
        fallback_policy=(
            PROFILE_FALLBACK_BLOSUM62_HYBRID
            if feature_type in (PSSM_HYBRID_FEATURE_TYPES | PSFM_HYBRID_FEATURE_TYPES)
            else PROFILE_FALLBACK_NONE
        ),
    )
    provenance["subject_fasta_sha256"] = sha256_file(subject_path)
    audit_summary = profile_audit_summary({"sample": sample})
    audit = {
        "feature_type": feature_type,
        "generation_fingerprint": provenance["generation_fingerprint"],
        "summary": audit_summary,
        "samples": [
            {
                "sample_id": "sample",
                "sequence_length": len(sequence),
                "zero_observation_positions": list(zero_positions),
                "profile_source": source,
                "fallback": fallback,
                "fallback_reason": fallback_reason,
                "cache_reused": False,
                **sample["quality"],
            }
        ],
    }
    audit_path = root / "profile_audit.json"
    atomic_write_json(audit_path, audit)
    provenance["audit_sha256"] = sha256_file(audit_path)
    provenance["audit_summary"] = audit_summary
    manifest = new_profile_manifest(
        feature_type=feature_type,
        samples={"sample": sample},
        provenance=provenance,
    )
    index_path = root / "index.json"
    atomic_write_json(index_path, manifest)
    return index_path


def test_blosum62_uses_canonical_scores_and_explicit_extended_rows():
    assert AA_ORDER == "ARNDCQEGHILKMFPSTWYV"
    assert BLOSUM62_HALF_BIT_SCORES.dtype == np.int8
    assert blosum62_row("A").tolist() == [
        4,
        -1,
        -2,
        -2,
        0,
        -1,
        -1,
        0,
        -2,
        -1,
        -1,
        -1,
        -1,
        -2,
        -1,
        1,
        0,
        -3,
        -2,
        0,
    ]
    for residue in "BZX":
        assert np.array_equal(
            blosum62_row(residue),
            BLOSUM62_AMBIGUOUS_ROWS[residue],
        )
    for residue in "JUO":
        assert np.array_equal(blosum62_row(residue), blosum62_row("X"))
    assert np.array_equal(blosum62_row("f"), blosum62_row("F"))
    with pytest.raises(SequenceProfileError, match="one ASCII letter"):
        blosum62_row("*")


def test_blosum62_kmer_mean_is_a_named_platform_extension():
    expected = (
        BLOSUM62_HALF_BIT_SCORES[0].astype(np.float32)
        + BLOSUM62_HALF_BIT_SCORES[1].astype(np.float32)
    ) / 2.0
    assert np.array_equal(blosum62_token_mean("AR"), expected)
    encoder = Blosum62Encoder(max_length=2)
    encoded = encoder.encode(["AR"])
    assert encoded.shape == (2, 20)
    assert torch.equal(encoded[0], torch.from_numpy(expected))
    assert torch.count_nonzero(encoded[1]) == 0
    assert encoder.score_unit == "half_bit"
    assert encoder.token_aggregation == "mean_within_kmer"
    direct_contract = blosum62_contract_metadata(k=1)
    assert direct_contract["method_name"] == "BLOSUM62"
    assert direct_contract["canonical_blosum62_per_residue"] is True
    assert direct_contract["matrix_sha256"] == BLOSUM62_MATRIX_SHA256
    assert direct_contract["score_unit"] == "half_bit"
    extension_contract = blosum62_contract_metadata(k=2)
    assert extension_contract["method_name"] == "BLOSUM62-derived k-mer mean"
    assert extension_contract["canonical_blosum62_per_residue"] is False
    assert extension_contract["kmer_aggregation"] == "arithmetic_mean_of_residue_score_rows"


def _write_ascii_pssm(
    path,
    sequence,
    header=AA_ORDER,
    frequencies_by_row=None,
):
    rows = []
    for position, residue in enumerate(sequence, start=1):
        scores = list(range(20))
        frequencies = (
            ["0"] * 20
            if frequencies_by_row is None
            else [str(value) for value in frequencies_by_row[position - 1]]
        )
        rows.append(
            f"{position:4d} {residue} "
            + " ".join(str(value) for value in scores)
            + " "
            + " ".join(frequencies)
        )
    path.write_text(
        "Last position-specific scoring matrix computed\n"
        + "          "
        + " ".join(header)
        + "\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


def _write_psfm_checkpoint(path, sequence, matrix):
    serialized = np.zeros((len(sequence), 28), dtype=np.float64)
    serialized[:, NCBISTDAA_CODES] = matrix
    values = ",\n".join(_asn_real(value) for value in serialized.reshape(-1))
    encoded_query = (
        bytes(int(NCBISTDAA_CODES[AA_ORDER.index(residue)]) for residue in sequence).hex().upper()
    )
    path.write_text(
        "\n".join(
            [
                "PssmWithParameters ::= {",
                "  query seq {",
                "    inst {",
                "      repr raw,",
                "      mol aa,",
                f"      length {len(sequence)},",
                f"      seq-data ncbistdaa '{encoded_query}'H",
                "    }",
                "  },",
                "  pssm {",
                "    isProtein TRUE,",
                "    numRows 28,",
                f"    numColumns {len(sequence)},",
                "    byRow FALSE,",
                "    intermediateData {",
                "      weightedResFreqsPerPos {",
                values,
                "      }",
                "    }",
                "  }",
                "}",
            ]
        ),
        encoding="utf-8",
    )


def test_ascii_pssm_parser_checks_header_positions_residues_and_length(tmp_path):
    path = tmp_path / "profile.pssm"
    reversed_header = AA_ORDER[::-1]
    _write_ascii_pssm(path, "AR", header=reversed_header)
    matrix = parse_ascii_pssm(path, "AR")
    assert matrix.shape == (2, 20)
    assert matrix[0].tolist() == list(reversed(range(20)))

    with pytest.raises(SequenceProfileError, match="differ"):
        parse_ascii_pssm(path, "AA")

    text = path.read_text(encoding="utf-8").replace("   2 R", "   3 R")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(SequenceProfileError, match="continuous"):
        parse_ascii_pssm(path, "AR")


def test_strict_and_hybrid_profile_stores_are_not_interchangeable(tmp_path):
    sequence = "AR"
    pssm = np.asarray([[1] * 20, [2] * 20], dtype=np.int32)
    strict_index = _write_profile_store(
        tmp_path,
        feature_type=FEATURE_PSSM,
        sequence=sequence,
        matrix=pssm,
        source=SOURCE_PSI_BLAST_PSSM,
    )
    strict = PSSMEncoder(3, strict_index)
    encoded = strict.encode_sequence(sequence, sample_id="sample")
    assert encoded.shape == (3, 20)
    assert torch.equal(encoded[:2], torch.from_numpy(pssm).float())
    with pytest.raises(SequenceProfileError, match="digest mismatch"):
        strict.encode_sequence("AA", sample_id="sample")
    with pytest.raises(SequenceProfileError, match="feature_type"):
        PSSMBlosum62HybridEncoder(3, strict_index)

    fallback = blosum62_score_profile(sequence)
    hybrid_index = _write_profile_store(
        tmp_path,
        feature_type=FEATURE_PSSM_BLOSUM62_HYBRID,
        sequence=sequence,
        matrix=fallback,
        source=SOURCE_BLOSUM62_FALLBACK,
        fallback=True,
    )
    hybrid = PSSMBlosum62HybridEncoder(2, hybrid_index)
    assert torch.equal(
        hybrid.encode_sequence(sequence, sample_id="sample"),
        torch.from_numpy(fallback).float(),
    )
    with pytest.raises(SequenceProfileError, match="feature_type"):
        PSSMEncoder(2, hybrid_index)

    psfm_fallback = blosum62_target_frequency_profile(sequence)
    psfm_hybrid_index = _write_profile_store(
        tmp_path,
        feature_type=FEATURE_PSFM_BLOSUM62_HYBRID,
        sequence=sequence,
        matrix=psfm_fallback,
        source=SOURCE_BLOSUM62_PSFM_FALLBACK,
        fallback=True,
    )
    psfm_hybrid = PSFMBlosum62HybridEncoder(2, psfm_hybrid_index)
    assert torch.equal(
        psfm_hybrid.encode_sequence(sequence, sample_id="sample"),
        torch.from_numpy(psfm_fallback).float(),
    )
    with pytest.raises(SequenceProfileError, match="feature_type"):
        PSFMEncoder(2, psfm_hybrid_index)


def test_profile_store_rejects_legacy_paths_and_matrix_tampering(tmp_path):
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"sample": "sample.pt"}), encoding="utf-8")
    with pytest.raises(SequenceProfileError) as legacy_error:
        ProfileStore(legacy, feature_type=FEATURE_PSSM)
    assert legacy_error.value.code == "profile_manifest_legacy"

    index_path = _write_profile_store(
        tmp_path,
        feature_type=FEATURE_PSSM,
        sequence="A",
        matrix=np.ones((1, 20), dtype=np.int32),
        source=SOURCE_PSI_BLAST_PSSM,
    )
    manifest = json.loads(index_path.read_text(encoding="utf-8"))
    manifest["samples"]["sample"]["path"] = "../escape.npz"
    index_path.write_text(json.dumps(manifest), encoding="utf-8")
    store = ProfileStore(index_path, feature_type=FEATURE_PSSM)
    with pytest.raises(SequenceProfileError) as path_error:
        store.load("sample", "A")
    assert path_error.value.code == "profile_manifest_path_escape"


def _asn_real(value):
    numeric = float(value)
    if numeric.is_integer():
        return f"{{ {int(numeric)}, 10, 0 }}"
    return f"{{ {int(round(numeric * 100))}, 10, -2 }}"


def test_psfm_comes_from_weighted_observed_frequencies_not_softmax(tmp_path):
    serialized = np.zeros((2, 28), dtype=np.float64)
    serialized[0, NCBISTDAA_CODES[0]] = 0.75
    serialized[0, NCBISTDAA_CODES[1]] = 0.25
    values = ",\n".join(_asn_real(value) for value in serialized.reshape(-1))
    checkpoint = tmp_path / "profile.asn1"
    checkpoint.write_text(
        "\n".join(
            [
                "PssmWithParameters ::= {",
                "  query seq {",
                "    inst {",
                "      repr raw,",
                "      mol aa,",
                "      length 2,",
                "      seq-data ncbistdaa '0110'H",
                "    }",
                "  },",
                "  pssm {",
                "    isProtein TRUE,",
                "    numRows 28,",
                "    numColumns 2,",
                "    byRow FALSE,",
                "    intermediateData {",
                "      weightedResFreqsPerPos {",
                values,
                "      }",
                "    }",
                "  }",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    matrix, zero_rows = parse_psiblast_psfm(
        checkpoint,
        expected_sequence="AR",
    )
    assert matrix[0, :2].tolist() == pytest.approx([0.75, 0.25])
    assert zero_rows == (1,)
    with pytest.raises(SequenceProfileError) as mismatch:
        parse_psiblast_psfm(checkpoint, expected_sequence="AA")
    assert mismatch.value.code == "profile_sequence_mismatch"

    duplicate = tmp_path / "duplicate.asn1"
    duplicate.write_text(
        checkpoint.read_text(encoding="utf-8").replace(
            "    numRows 28,",
            "    numRows 28,\n    numRows 28,",
        ),
        encoding="utf-8",
    )
    with pytest.raises(SequenceProfileError) as duplicate_error:
        parse_psiblast_psfm(duplicate, expected_sequence="AR")
    assert duplicate_error.value.code == "profile_checkpoint_duplicate_field"
    ascii_pssm = tmp_path / "profile.pssm"
    _write_ascii_pssm(ascii_pssm, "AR")
    with pytest.raises(SequenceProfileError) as crosscheck:
        parse_psiblast_psfm(
            checkpoint,
            expected_sequence="AR",
            ascii_pssm=ascii_pssm,
        )
    assert crosscheck.value.code == "profile_frequency_crosscheck_failed"

    index_path = _write_profile_store(
        tmp_path,
        feature_type=FEATURE_PSFM,
        sequence="AR",
        matrix=matrix,
        source=SOURCE_PSI_BLAST_WEIGHTED_OBSERVED,
        zero_positions=zero_rows,
    )
    encoded = PSFMEncoder(2, index_path).encode_sequence(
        "AR",
        sample_id="sample",
    )
    assert torch.allclose(encoded, torch.from_numpy(matrix))


def test_psfm_ascii_crosscheck_allows_only_one_percentage_point(tmp_path):
    matrix = np.zeros((2, 20), dtype=np.float64)
    matrix[0, 0] = 0.75
    matrix[0, 1] = 0.25
    checkpoint = tmp_path / "profile.asn1"
    _write_psfm_checkpoint(checkpoint, "AR", matrix)

    for label, first_row in (
        ("exact", [75, 25, *([0] * 18)]),
        ("one-point", [74, 26, *([0] * 18)]),
    ):
        ascii_pssm = tmp_path / f"{label}.pssm"
        _write_ascii_pssm(
            ascii_pssm,
            "AR",
            frequencies_by_row=[first_row, [0] * 20],
        )
        parsed, zero_rows = parse_psiblast_psfm(
            checkpoint,
            expected_sequence="AR",
            ascii_pssm=ascii_pssm,
        )
        assert np.allclose(parsed, matrix)
        assert zero_rows == (1,)

    outside_tolerance = tmp_path / "two-points.pssm"
    _write_ascii_pssm(
        outside_tolerance,
        "AR",
        frequencies_by_row=[
            [73, 27, *([0] * 18)],
            [0] * 20,
        ],
    )
    with pytest.raises(SequenceProfileError) as error:
        parse_psiblast_psfm(
            checkpoint,
            expected_sequence="AR",
            ascii_pssm=outside_tolerance,
        )
    assert error.value.code == "profile_frequency_crosscheck_failed"
    assert error.value.details["difference_percentage_points"] == 2
    assert error.value.details["tolerance_percentage_points"] == (
        PSFM_ASCII_CROSSCHECK_TOLERANCE_PERCENTAGE_POINTS
    )


def test_psfm_explicit_mask_preserves_internal_zero_observation_rows(tmp_path):
    matrix = np.zeros((3, 20), dtype=np.float32)
    matrix[0, 0] = 1.0
    matrix[2, 1] = 1.0
    index_path = _write_profile_store(
        tmp_path,
        feature_type=FEATURE_PSFM,
        sequence="ARR",
        matrix=matrix,
        source=SOURCE_PSI_BLAST_WEIGHTED_OBSERVED,
        zero_positions=(1,),
    )
    encoder = PSFMEncoder(5, index_path)
    direct = encoder.encode_sequence("ARR", sample_id="sample")
    assert isinstance(direct, torch.Tensor)
    features, mask = encoder.encode_sequence_with_mask(
        "ARR",
        sample_id="sample",
    )
    assert mask.tolist() == [True, True, True, False, False]
    assert torch.count_nonzero(features[1]) == 0

    class Tokenizer:
        max_length = 5

        @staticmethod
        def tokenize(sequence):
            return list(sequence)

    composed = ComposedFeaturizer(Tokenizer(), encoder)
    encoded = composed.encode("ARR", sample_id="sample")
    assert isinstance(encoded, ResidueModelInput)
    assert encoded.mask.tolist() == mask.tolist()


@pytest.mark.parametrize(
    "model",
    [
        FeatureTextCNN(in_dim=20, num_classes=2),
        FeatureRNN(in_dim=20, hidden_dim=8, num_classes=2),
        FeatureGRU(in_dim=20, hidden_dim=8, num_classes=2),
        FeatureLSTM(in_dim=20, hidden_dim=8, num_classes=2),
        FeatureTransformer(
            in_dim=20,
            max_length=5,
            embed_dim=8,
            num_heads=2,
            ff_dim=16,
            num_layers=1,
            num_classes=2,
        ),
        ResidueGraphClassifier(
            in_dim=20,
            hidden_dim=8,
            graph_kind="gcn",
            num_classes=2,
        ),
        ResidueGraphClassifier(
            in_dim=20,
            hidden_dim=8,
            graph_kind="gat",
            heads=2,
            num_classes=2,
        ),
    ],
)
def test_position_models_accept_valid_all_zero_residue(model):
    features = torch.zeros((2, 5, 20), dtype=torch.float32)
    features[:, 0, 0] = 1.0
    features[:, 2, 1] = 1.0
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
        ]
    )
    model.eval()
    output = model(ResidueModelInput(features, mask))
    contaminated = features.clone()
    contaminated[~mask] = 123.0
    contaminated_output = model(ResidueModelInput(contaminated, mask))
    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()
    assert torch.allclose(output, contaminated_output, atol=1e-6)


def test_pair_attention_accepts_explicit_psfm_masks():
    from scripts.train_pair_pipeline import (
        HeterogeneousPairCrossAttentionClassifier,
        PairCrossAttentionClassifier,
    )

    peptide = ResidueModelInput(
        torch.zeros((2, 3, 20)),
        torch.tensor([[True, True, False], [True, True, True]]),
    )
    partner = ResidueModelInput(
        torch.zeros((2, 3, 20)),
        torch.tensor([[True, True, True], [True, False, False]]),
    )
    peptide.features[:, 0, 0] = 1.0
    partner.features[:, 0, 1] = 1.0
    joined = ResidueModelInput(
        torch.cat([peptide.features, partner.features], dim=1),
        torch.cat([peptide.mask, partner.mask], dim=1),
    )
    homogeneous = PairCrossAttentionClassifier(
        seq_len=3,
        encoder_name="psfm",
        in_dim=20,
        hidden_dim=8,
        n_heads=2,
    )
    heterogeneous = HeterogeneousPairCrossAttentionClassifier(
        peptide_encoder_name="psfm",
        partner_encoder_name="psfm",
        peptide_in_dim=20,
        partner_in_dim=20,
        hidden_dim=8,
        n_heads=2,
    )
    assert homogeneous(joined).shape == (2, 1)
    assert heterogeneous(peptide, partner).shape == (2, 1)


def test_psfm_rejects_ambiguous_query_residues():
    assert normalize_psfm_sequence("ARND") == "ARND"
    for sequence in ("AB", "AZ", "AX", "AJ", "AU", "AO"):
        with pytest.raises(SequenceProfileError) as exc_info:
            normalize_psfm_sequence(sequence)
        assert exc_info.value.code == "psfm_query_residue_invalid"


def test_psiblast_command_fixes_scientific_parameters(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "pepplatform.profile_generation.require_blast_binary",
        lambda name: f"/usr/bin/{name}",
    )
    config = ProfileGenerationConfig(
        iterations=4,
        evalue=1e-4,
        inclusion_evalue=2e-4,
        threads=3,
    )
    command = build_psiblast_command(
        query_fasta=tmp_path / "query.fa",
        database_prefix=tmp_path / "db",
        ascii_pssm=tmp_path / "query.pssm",
        checkpoint=tmp_path / "query.asn1",
        hits=tmp_path / "query.hits",
        config=config,
        negative_seqidlist=tmp_path / "exclude.txt",
    )

    def option(name):
        return command[command.index(name) + 1]

    assert option("-num_iterations") == "4"
    assert option("-evalue") == "0.0001"
    assert option("-inclusion_ethresh") == "0.0002"
    assert option("-matrix") == "BLOSUM62"
    assert option("-gapopen") == "11"
    assert option("-gapextend") == "1"
    assert option("-word_size") == "3"
    assert option("-threshold") == "11.0"
    assert option("-pseudocount") == "0"
    assert option("-comp_based_stats") == "1"
    assert option("-seg") == "no"
    assert option("-soft_masking") == "false"
    assert option("-outfmt") == HIT_SUMMARY_OUTFMT
    assert option("-negative_seqidlist").endswith("exclude.txt")
    assert "-save_pssm_after_last_round" in command


def test_hit_summary_uses_query_coordinates_and_rejects_invalid_rows(
    tmp_path,
):
    hits = tmp_path / "hits.tsv"
    hits.write_text(
        "# Iteration: 1\n"
        "query\told\t50.0\t1\t5\t10\t1e-5\t20\n"
        "# Iteration: 2\n"
        "query\tsubject\t75.0\t2\t7\t10\t1e-20\t42.5\n"
        "query\tsubject\t70.0\t1\t5\t10\t1e-10\t30\n"
        "Search has CONVERGED!\n",
        encoding="utf-8",
    )
    summary = parse_hit_summary(
        hits,
        expected_query_id="query",
        expected_query_length=10,
    )
    assert summary == {
        "nonself_hit_count": 1,
        "max_query_coverage": 0.6,
        "max_identity": 75.0,
        "min_evalue": 1e-20,
        "zero_nonself_hits": False,
        "final_iteration": 2,
        "final_iteration_hit_rows": 2,
    }
    with pytest.raises(SequenceProfileError) as excluded:
        parse_hit_summary(
            hits,
            expected_query_id="query",
            expected_query_length=10,
            excluded_subject_ids={"subject"},
        )
    assert excluded.value.code == "profile_self_exclusion_failed"

    for row in (
        "query\tsubject\t101\t1\t5\t10\t1e-5\t20\n",
        "query\tsubject\t50\t1\t11\t10\t1e-5\t20\n",
        "query\tsubject\t50\t1\t5\t10\t-1\t20\n",
        "query\tsubject\t50\t1\t5\t10\t1e-5\tnan\n",
    ):
        hits.write_text("# Iteration: 1\n" + row, encoding="utf-8")
        with pytest.raises(SequenceProfileError) as invalid:
            parse_hit_summary(hits)
        assert invalid.value.code == "profile_hits_invalid"


def test_hybrid_fallback_after_successful_zero_hit_search(
    monkeypatch,
    tmp_path,
    capsys,
):
    subject = tmp_path / "subject.fa"
    subject.write_text(">SUBJ_0\nAR\n", encoding="ascii")
    monkeypatch.setattr(
        "pepplatform.profile_generation.psiblast_version",
        lambda: "psiblast: test",
    )
    monkeypatch.setattr(
        "pepplatform.profile_generation.makeblastdb_version",
        lambda: "makeblastdb: test",
    )
    monkeypatch.setattr(
        "pepplatform.profile_generation.make_blast_database",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "pepplatform.profile_generation.require_blast_binary",
        lambda name: f"/usr/bin/{name}",
    )

    search_commands = []

    def zero_hit_search(command, *, timeout_seconds):
        del timeout_seconds
        search_commands.append(command)
        hits = Path(command[command.index("-out") + 1])
        hits.write_text(
            "# Iteration: 1\n# 0 hits found\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "pepplatform.profile_generation.run_psiblast",
        zero_hit_search,
    )
    indexes = build_profile_caches(
        samples=[("sample", "AR")],
        feature_directories={
            FEATURE_PSSM_BLOSUM62_HYBRID: tmp_path / "hybrid",
            FEATURE_PSFM_BLOSUM62_HYBRID: tmp_path / "psfm-hybrid",
            FEATURE_AC_PSSM_BLOSUM62_HYBRID: tmp_path / "ac-hybrid",
            FEATURE_CC_PSSM_BLOSUM62_HYBRID: tmp_path / "cc-hybrid",
            FEATURE_ACC_PSSM_BLOSUM62_HYBRID: tmp_path / "acc-hybrid",
        },
        subject_fasta=subject,
        sequence_subject_ids={"AR": ["SUBJ_0"]},
        config=ProfileGenerationConfig(),
        database_scope="train_only",
        database_provenance={"train_fasta_sha256": sha256_file(subject)},
    )
    assert len(search_commands) == 1
    progress_lines = capsys.readouterr().out.splitlines()
    assert [
        (line.split()[1], line.split()[2], line.split()[3], line.split()[4])
        for line in progress_lines
    ] == [
        ("stage=database", "status=running", "current=0", "total=1"),
        ("stage=database", "status=complete", "current=1", "total=1"),
        ("stage=profiles", "status=running", "current=0", "total=1"),
        ("stage=profiles", "status=running", "current=1", "total=1"),
        ("stage=profiles", "status=complete", "current=1", "total=1"),
    ]
    loaded = ProfileStore(
        indexes[FEATURE_PSSM_BLOSUM62_HYBRID],
        feature_type=FEATURE_PSSM_BLOSUM62_HYBRID,
    ).load("sample", "AR")
    assert loaded.metadata["fallback"] is True
    assert loaded.metadata["fallback_reason"] == "profile_output_missing"
    assert loaded.metadata["quality"]["zero_nonself_hits"] is True
    assert torch.equal(
        loaded.matrix,
        torch.from_numpy(blosum62_score_profile("AR")).float(),
    )
    psfm_hybrid_store = ProfileStore(
        indexes[FEATURE_PSFM_BLOSUM62_HYBRID],
        feature_type=FEATURE_PSFM_BLOSUM62_HYBRID,
    )
    psfm_hybrid_loaded = psfm_hybrid_store.load("sample", "AR")
    assert psfm_hybrid_loaded.metadata["fallback"] is True
    assert psfm_hybrid_loaded.metadata["profile_source"] == SOURCE_BLOSUM62_PSFM_FALLBACK
    assert torch.allclose(
        psfm_hybrid_loaded.matrix,
        torch.from_numpy(blosum62_target_frequency_profile("AR")).float(),
    )
    assert psfm_hybrid_store.manifest["provenance"]["profile_semantics"] == {
        **PSFM_HYBRID_SEMANTICS,
        "aa_order": AA_ORDER,
        "scientific_reference": dict(PSFM_NAMING_REFERENCE),
    }
    tampered_psfm_manifest = json.loads(
        indexes[FEATURE_PSFM_BLOSUM62_HYBRID].read_text(encoding="utf-8")
    )
    tampered_psfm_manifest["provenance"]["profile_semantics"]["fallback_matrix"] = (
        "pssm_score_profile"
    )
    atomic_write_json(
        indexes[FEATURE_PSFM_BLOSUM62_HYBRID],
        tampered_psfm_manifest,
    )
    with pytest.raises(SequenceProfileError) as tampered_psfm_error:
        ProfileStore(
            indexes[FEATURE_PSFM_BLOSUM62_HYBRID],
            feature_type=FEATURE_PSFM_BLOSUM62_HYBRID,
        )
    assert tampered_psfm_error.value.code == "profile_manifest_provenance_invalid"
    for feature, transform in (
        (FEATURE_AC_PSSM_BLOSUM62_HYBRID, "ac_pssm"),
        (FEATURE_CC_PSSM_BLOSUM62_HYBRID, "cc_pssm"),
        (FEATURE_ACC_PSSM_BLOSUM62_HYBRID, "acc_pssm"),
    ):
        store = ProfileStore(indexes[feature], feature_type=feature)
        hybrid_loaded = store.load("sample", "AR")
        assert hybrid_loaded.metadata["fallback"] is True
        assert (
            store.manifest["provenance"]["profile_semantics"]["downstream_transform"] == transform
        )
        assert indexes[feature].parent != indexes[FEATURE_PSSM_BLOSUM62_HYBRID].parent
        with pytest.raises(SequenceProfileError, match="feature_type"):
            ProfileStore(
                indexes[feature],
                feature_type=FEATURE_PSSM_BLOSUM62_HYBRID,
            )

    tampered_index = indexes[FEATURE_AC_PSSM_BLOSUM62_HYBRID]
    tampered_manifest = json.loads(tampered_index.read_text(encoding="utf-8"))
    tampered_manifest["provenance"]["profile_semantics"]["downstream_transform"] = "cc_pssm"
    atomic_write_json(tampered_index, tampered_manifest)
    with pytest.raises(SequenceProfileError) as tampered_error:
        ProfileStore(
            tampered_index,
            feature_type=FEATURE_AC_PSSM_BLOSUM62_HYBRID,
        )
    assert tampered_error.value.code == "profile_manifest_provenance_invalid"

    with pytest.raises(SequenceProfileError) as strict_error:
        build_profile_caches(
            samples=[("sample", "AR")],
            feature_directories={FEATURE_PSSM: tmp_path / "strict"},
            subject_fasta=subject,
            sequence_subject_ids={"AR": ["SUBJ_0"]},
            config=ProfileGenerationConfig(),
            database_scope="train_only",
            database_provenance={"train_fasta_sha256": sha256_file(subject)},
        )
    assert strict_error.value.code == "profile_no_nonself_hits"
    assert strict_error.value.details == {
        "feature": FEATURE_PSSM,
        "sample_id": "sample",
        "sequence_length": 2,
        "nonself_hit_count": 0,
        "zero_nonself_hits": True,
    }
    assert "sample_id='sample'" in str(strict_error.value)

    with pytest.raises(SequenceProfileError) as psfm_error:
        build_profile_caches(
            samples=[("sample", "AR")],
            feature_directories={FEATURE_PSFM: tmp_path / "strict-psfm"},
            subject_fasta=subject,
            sequence_subject_ids={"AR": ["SUBJ_0"]},
            config=ProfileGenerationConfig(),
            database_scope="train_only",
            database_provenance={"train_fasta_sha256": sha256_file(subject)},
        )
    assert psfm_error.value.code == "profile_no_nonself_hits"
    assert psfm_error.value.details["feature"] == FEATURE_PSFM
    assert psfm_error.value.details["sample_id"] == "sample"

    def failed_search(_command, *, timeout_seconds):
        del timeout_seconds
        raise SequenceProfileError(
            "operational failure",
            code="profile_search_failed",
        )

    monkeypatch.setattr(
        "pepplatform.profile_generation.run_psiblast",
        failed_search,
    )
    with pytest.raises(SequenceProfileError) as operational_error:
        build_profile_caches(
            samples=[("other", "AR")],
            feature_directories={FEATURE_PSSM_BLOSUM62_HYBRID: tmp_path / "hybrid-error"},
            subject_fasta=subject,
            sequence_subject_ids={"AR": ["SUBJ_0"]},
            config=ProfileGenerationConfig(),
            database_scope="train_only",
            database_provenance={"train_fasta_sha256": sha256_file(subject)},
        )
    assert operational_error.value.code == "profile_search_failed"


def test_target_frequency_fallback_handles_unavailable_checkpoint_with_hits(
    monkeypatch,
    tmp_path,
):
    subject = tmp_path / "subject.fa"
    subject.write_text(">SUBJ_0\nRR\n", encoding="ascii")
    monkeypatch.setattr(
        "pepplatform.profile_generation.psiblast_version",
        lambda: "psiblast: test",
    )
    monkeypatch.setattr(
        "pepplatform.profile_generation.makeblastdb_version",
        lambda: "makeblastdb: test",
    )
    monkeypatch.setattr(
        "pepplatform.profile_generation.make_blast_database",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "pepplatform.profile_generation.require_blast_binary",
        lambda name: f"/usr/bin/{name}",
    )

    def usable_hit_with_checkpoint(command, *, timeout_seconds):
        del timeout_seconds
        query_fasta = Path(command[command.index("-query") + 1])
        query_id = query_fasta.read_text(encoding="ascii").splitlines()[0][1:]
        Path(command[command.index("-out_pssm") + 1]).write_text(
            "checkpoint",
            encoding="ascii",
        )
        Path(command[command.index("-out") + 1]).write_text(
            f"# Iteration: 1\n{query_id}\tSUBJ_0\t50\t1\t2\t2\t1e-5\t10\n",
            encoding="ascii",
        )

    def unavailable_frequency(*_args, **_kwargs):
        raise SequenceProfileError(
            "zero-sum target-frequency rows",
            code="profile_frequency_unavailable",
        )

    monkeypatch.setattr(
        "pepplatform.profile_generation.run_psiblast",
        usable_hit_with_checkpoint,
    )
    monkeypatch.setattr(
        "pepplatform.profile_generation.parse_psiblast_target_frequencies",
        unavailable_frequency,
    )
    indexes = build_profile_caches(
        samples=[("sample", "AR")],
        feature_directories={FEATURE_TARGET_FREQUENCY: tmp_path / "fallback"},
        subject_fasta=subject,
        sequence_subject_ids={"RR": ["SUBJ_0"]},
        config=ProfileGenerationConfig(),
        database_scope="train_only",
        database_provenance={"train_fasta_sha256": sha256_file(subject)},
        target_frequency_fallback=True,
    )
    manifest = json.loads(indexes[FEATURE_TARGET_FREQUENCY].read_text(encoding="utf-8"))
    entry = manifest["samples"]["sample"]
    assert entry["fallback"] is True
    assert entry["fallback_reason"] == "profile_frequency_unavailable"
    assert entry["profile_source"] == SOURCE_BLOSUM62_TARGET_FREQUENCY
    assert entry["quality"]["zero_nonself_hits"] is False

    with pytest.raises(SequenceProfileError) as strict_error:
        build_profile_caches(
            samples=[("sample", "AR")],
            feature_directories={FEATURE_TARGET_FREQUENCY: tmp_path / "strict"},
            subject_fasta=subject,
            sequence_subject_ids={"RR": ["SUBJ_0"]},
            config=ProfileGenerationConfig(),
            database_scope="train_only",
            database_provenance={"train_fasta_sha256": sha256_file(subject)},
            target_frequency_fallback=False,
        )
    assert strict_error.value.code == "profile_frequency_unavailable"

    def invalid_checkpoint(*_args, **_kwargs):
        raise SequenceProfileError(
            "negative checkpoint value",
            code="profile_checkpoint_value_invalid",
        )

    monkeypatch.setattr(
        "pepplatform.profile_generation.parse_psiblast_target_frequencies",
        invalid_checkpoint,
    )
    invalid_indexes = build_profile_caches(
        samples=[("sample", "AR")],
        feature_directories={FEATURE_TARGET_FREQUENCY: tmp_path / "invalid"},
        subject_fasta=subject,
        sequence_subject_ids={"RR": ["SUBJ_0"]},
        config=ProfileGenerationConfig(),
        database_scope="train_only",
        database_provenance={"train_fasta_sha256": sha256_file(subject)},
        target_frequency_fallback=True,
    )
    invalid_manifest = json.loads(
        invalid_indexes[FEATURE_TARGET_FREQUENCY].read_text(encoding="utf-8")
    )
    invalid_entry = invalid_manifest["samples"]["sample"]
    assert invalid_entry["fallback"] is True
    assert invalid_entry["fallback_reason"] == "profile_checkpoint_value_invalid"
    assert invalid_entry["quality"]["zero_nonself_hits"] is False

    invalid_store = ProfileStore(
        invalid_indexes[FEATURE_TARGET_FREQUENCY],
        feature_type=FEATURE_TARGET_FREQUENCY,
    )
    invalid_loaded = invalid_store.load("sample", "AR")
    assert invalid_loaded.metadata["fallback_reason"] == ("profile_checkpoint_value_invalid")
    assert invalid_store.manifest["provenance"]["audit_summary"]["fallback_reason_counts"] == {
        "profile_checkpoint_value_invalid": 1
    }


def test_psfm_hybrid_falls_back_on_invalid_ascii_frequency_rows_with_hits(
    monkeypatch,
    tmp_path,
):
    subject = tmp_path / "subject.fa"
    subject.write_text(">SUBJ_0\nRR\n", encoding="ascii")
    monkeypatch.setattr(
        "pepplatform.profile_generation.psiblast_version",
        lambda: "psiblast: test",
    )
    monkeypatch.setattr(
        "pepplatform.profile_generation.makeblastdb_version",
        lambda: "makeblastdb: test",
    )
    monkeypatch.setattr(
        "pepplatform.profile_generation.make_blast_database",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "pepplatform.profile_generation.require_blast_binary",
        lambda name: f"/usr/bin/{name}",
    )

    def usable_hit_with_invalid_ascii_frequencies(command, *, timeout_seconds):
        del timeout_seconds
        query_fasta = Path(command[command.index("-query") + 1])
        query_id = query_fasta.read_text(encoding="ascii").splitlines()[0][1:]
        _write_ascii_pssm(
            Path(command[command.index("-out_ascii_pssm") + 1]),
            "AR",
            frequencies_by_row=[[1] * 20, [1] * 20],
        )
        Path(command[command.index("-out_pssm") + 1]).write_text(
            "checkpoint",
            encoding="ascii",
        )
        Path(command[command.index("-out") + 1]).write_text(
            f"# Iteration: 1\n{query_id}\tSUBJ_0\t50\t1\t2\t2\t1e-5\t10\n",
            encoding="ascii",
        )

    monkeypatch.setattr(
        "pepplatform.profile_generation.run_psiblast",
        usable_hit_with_invalid_ascii_frequencies,
    )

    def invalid_ascii_frequency_rows(*_args, **_kwargs):
        raise SequenceProfileError(
            "implausible ASCII PSSM frequency sum",
            code="profile_frequency_not_normalized",
        )

    monkeypatch.setattr(
        "pepplatform.profile_generation.parse_psiblast_psfm",
        invalid_ascii_frequency_rows,
    )
    indexes = build_profile_caches(
        samples=[("sample", "AR")],
        feature_directories={FEATURE_PSFM_BLOSUM62_HYBRID: tmp_path / "hybrid"},
        subject_fasta=subject,
        sequence_subject_ids={"RR": ["SUBJ_0"]},
        config=ProfileGenerationConfig(),
        database_scope="train_only",
        database_provenance={"train_fasta_sha256": sha256_file(subject)},
    )
    store = ProfileStore(
        indexes[FEATURE_PSFM_BLOSUM62_HYBRID],
        feature_type=FEATURE_PSFM_BLOSUM62_HYBRID,
    )
    loaded = store.load("sample", "AR")
    assert loaded.metadata["fallback"] is True
    assert loaded.metadata["fallback_reason"] == ("profile_frequency_not_normalized")
    assert loaded.metadata["quality"]["zero_nonself_hits"] is False
    assert torch.allclose(
        loaded.matrix,
        torch.from_numpy(blosum62_target_frequency_profile("AR")).float(),
    )

    with pytest.raises(SequenceProfileError) as strict_error:
        build_profile_caches(
            samples=[("sample", "AR")],
            feature_directories={FEATURE_PSFM: tmp_path / "strict"},
            subject_fasta=subject,
            sequence_subject_ids={"RR": ["SUBJ_0"]},
            config=ProfileGenerationConfig(),
            database_scope="train_only",
            database_provenance={"train_fasta_sha256": sha256_file(subject)},
        )
    assert strict_error.value.code == "profile_frequency_not_normalized"


def test_psfm_hybrid_falls_back_when_output_is_missing_despite_usable_hits(
    monkeypatch,
    tmp_path,
):
    subject = tmp_path / "subject.fa"
    subject.write_text(">SUBJ_0\nRR\n", encoding="ascii")
    monkeypatch.setattr(
        "pepplatform.profile_generation.psiblast_version",
        lambda: "psiblast: test",
    )
    monkeypatch.setattr(
        "pepplatform.profile_generation.makeblastdb_version",
        lambda: "makeblastdb: test",
    )
    monkeypatch.setattr(
        "pepplatform.profile_generation.make_blast_database",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "pepplatform.profile_generation.require_blast_binary",
        lambda name: f"/usr/bin/{name}",
    )

    def usable_hit_search(command, *, timeout_seconds):
        del timeout_seconds
        query_fasta = Path(command[command.index("-query") + 1])
        query_id = query_fasta.read_text(encoding="ascii").splitlines()[0][1:]
        hits = Path(command[command.index("-out") + 1])
        hits.write_text(
            f"# Iteration: 1\n{query_id}\tSUBJ_0\t50\t1\t2\t2\t1e-5\t10\n",
            encoding="ascii",
        )

    monkeypatch.setattr(
        "pepplatform.profile_generation.run_psiblast",
        usable_hit_search,
    )
    hybrid_indexes = build_profile_caches(
        samples=[("sample", "AR")],
        feature_directories={FEATURE_PSFM_BLOSUM62_HYBRID: tmp_path / "hybrid"},
        subject_fasta=subject,
        sequence_subject_ids={"RR": ["SUBJ_0"]},
        config=ProfileGenerationConfig(),
        database_scope="train_only",
        database_provenance={"train_fasta_sha256": sha256_file(subject)},
    )
    hybrid_manifest = json.loads(
        hybrid_indexes[FEATURE_PSFM_BLOSUM62_HYBRID].read_text(encoding="utf-8")
    )
    hybrid_entry = hybrid_manifest["samples"]["sample"]
    assert hybrid_entry["fallback"] is True
    assert hybrid_entry["fallback_reason"] == "profile_output_missing"
    assert hybrid_entry["quality"]["zero_nonself_hits"] is False

    with pytest.raises(SequenceProfileError) as strict_error:
        build_profile_caches(
            samples=[("sample", "AR")],
            feature_directories={FEATURE_PSFM: tmp_path / "strict"},
            subject_fasta=subject,
            sequence_subject_ids={"RR": ["SUBJ_0"]},
            config=ProfileGenerationConfig(),
            database_scope="train_only",
            database_provenance={"train_fasta_sha256": sha256_file(subject)},
        )
    assert strict_error.value.code == "profile_output_missing"


def test_subject_fasta_index_excludes_all_identical_sequences(tmp_path):
    subject = tmp_path / "subject.fa"
    subject.write_text(
        ">one description\nAJU\n>two\nAXX\n>three\nRRR\n",
        encoding="utf-8",
    )
    assert index_subject_fasta(subject) == {
        "AXX": ["one", "two"],
        "RRR": ["three"],
    }

    subject.write_text(">one\nA\n>one\nR\n", encoding="utf-8")
    with pytest.raises(SequenceProfileError) as duplicate:
        index_subject_fasta(subject)
    assert duplicate.value.code == "profile_database_input_invalid"

    subject.write_text(">one\nAR\n>two\nAR\n", encoding="utf-8")
    with pytest.raises(SequenceProfileError) as mismatch:
        build_profile_caches(
            samples=[("sample", "AR")],
            feature_directories={FEATURE_PSSM: tmp_path / "strict"},
            subject_fasta=subject,
            sequence_subject_ids={"AR": ["one"]},
            config=ProfileGenerationConfig(),
            database_scope="train_only",
            database_provenance={"train_fasta_sha256": sha256_file(subject)},
        )
    assert mismatch.value.code == "profile_self_exclusion_index_mismatch"


def test_profile_manifest_requires_full_provenance(tmp_path):
    index_path = _write_profile_store(
        tmp_path,
        feature_type=FEATURE_PSSM,
        sequence="A",
        matrix=np.ones((1, 20), dtype=np.int32),
        source=SOURCE_PSI_BLAST_PSSM,
    )
    manifest = json.loads(index_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == PROFILE_SCHEMA
    assert manifest["contract_version"] == PROFILE_CONTRACT_VERSION
    del manifest["provenance"]["scientific_parameters"]
    index_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SequenceProfileError) as exc_info:
        ProfileStore(index_path, feature_type=FEATURE_PSSM)
    assert exc_info.value.code == "profile_manifest_provenance_invalid"


def test_profile_store_rejects_audit_row_that_disagrees_with_manifest(tmp_path):
    index_path = _write_profile_store(
        tmp_path,
        feature_type=FEATURE_PSSM,
        sequence="A",
        matrix=np.ones((1, 20), dtype=np.int32),
        source=SOURCE_PSI_BLAST_PSSM,
    )
    manifest = json.loads(index_path.read_text(encoding="utf-8"))
    audit_path = index_path.parent / manifest["provenance"]["audit_artifact"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["samples"][0]["max_identity"] = 12.5
    atomic_write_json(audit_path, audit)
    manifest["provenance"]["audit_sha256"] = sha256_file(audit_path)
    atomic_write_json(index_path, manifest)

    with pytest.raises(SequenceProfileError) as exc_info:
        ProfileStore(index_path, feature_type=FEATURE_PSSM)
    assert exc_info.value.code == "profile_audit_invalid"


def test_psfm_store_rejects_noncanonical_query_even_if_manifest_is_forged(
    tmp_path,
):
    matrix = np.zeros((1, 20), dtype=np.float32)
    matrix[0, 0] = 1.0
    index_path = _write_profile_store(
        tmp_path,
        feature_type=FEATURE_PSFM,
        sequence="X",
        matrix=matrix,
        source=SOURCE_PSI_BLAST_WEIGHTED_OBSERVED,
    )

    store = ProfileStore(index_path, feature_type=FEATURE_PSFM)
    with pytest.raises(SequenceProfileError) as exc_info:
        store.load("sample", "X")
    assert exc_info.value.code == "psfm_query_residue_invalid"


@pytest.mark.skipif(
    shutil.which("psiblast") is None or shutil.which("makeblastdb") is None,
    reason="NCBI BLAST+ is unavailable",
)
def test_real_blast_psfm_current_checkpoint_and_ascii_crosscheck(
    tmp_path,
    monkeypatch,
):
    query = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANL"
    homolog = "MKTAYIAKQRQISFVKSHFSRQLDERLGLIEVQANL"
    subject = tmp_path / "subject.fasta"
    subject.write_text(f">TRAIN_000000\n{homolog}\n", encoding="ascii")
    indexes = build_profile_caches(
        samples=[("query", query)],
        feature_directories={FEATURE_PSFM: tmp_path / "psfm"},
        subject_fasta=subject,
        sequence_subject_ids={homolog: ["TRAIN_000000"]},
        config=ProfileGenerationConfig(
            iterations=2,
            threads=1,
            timeout_seconds=60,
        ),
        database_scope="test_frozen",
        database_provenance={"fixture": "real_blast"},
    )
    store = ProfileStore(indexes[FEATURE_PSFM], feature_type=FEATURE_PSFM)
    loaded = store.load("query", query)
    assert loaded.matrix.shape == (len(query), 20)
    assert store.manifest["contract_version"] == PROFILE_CONTRACT_VERSION
    assert store.manifest["provenance"]["scientific_parameters"]["composition_based_stats"] == "1"
    assert (
        store.manifest["provenance"]["profile_semantics"]["checkpoint_field"]
        == "weightedResFreqsPerPos"
    )
    matrix_path = store.samples["query"]["path"]
    resumed = build_profile_caches(
        samples=[("query", query)],
        feature_directories={FEATURE_PSFM: tmp_path / "psfm"},
        subject_fasta=subject,
        sequence_subject_ids={homolog: ["TRAIN_000000"]},
        config=ProfileGenerationConfig(
            iterations=2,
            threads=1,
            timeout_seconds=60,
        ),
        database_scope="test_frozen",
        database_provenance={"fixture": "real_blast"},
    )
    resumed_store = ProfileStore(
        resumed[FEATURE_PSFM],
        feature_type=FEATURE_PSFM,
    )
    assert resumed_store.samples["query"]["path"] == matrix_path
    audit = json.loads((tmp_path / "psfm" / "profile_audit.json").read_text(encoding="utf-8"))
    assert len(audit["samples"]) == 1
    assert audit["samples"][0]["cache_reused"] is True
    # A third request reaches an already existing content-addressed reuse audit.
    repeated = build_profile_caches(
        samples=[("query", query)],
        feature_directories={FEATURE_PSFM: tmp_path / "psfm"},
        subject_fasta=subject,
        sequence_subject_ids={homolog: ["TRAIN_000000"]},
        config=ProfileGenerationConfig(iterations=2, threads=1, timeout_seconds=60),
        database_scope="test_frozen",
        database_provenance={"fixture": "real_blast"},
    )
    repeated_store = ProfileStore(repeated[FEATURE_PSFM], feature_type=FEATURE_PSFM)
    assert np.array_equal(repeated_store.load("query", query).matrix, loaded.matrix)
    index_digest = sha256_file(resumed[FEATURE_PSFM])

    def fail_search(*_args, **_kwargs):
        raise SequenceProfileError(
            "forced search failure",
            code="profile_search_failed",
        )

    monkeypatch.setattr(
        "pepplatform.profile_generation.run_psiblast",
        fail_search,
    )
    with pytest.raises(SequenceProfileError):
        build_profile_caches(
            samples=[("query", query)],
            feature_directories={FEATURE_PSFM: tmp_path / "psfm"},
            subject_fasta=subject,
            sequence_subject_ids={homolog: ["TRAIN_000000"]},
            config=ProfileGenerationConfig(
                iterations=2,
                threads=1,
                timeout_seconds=60,
            ),
            database_scope="test_frozen",
            database_provenance={"fixture": "real_blast"},
            force=True,
        )
    assert sha256_file(resumed[FEATURE_PSFM]) == index_digest
    ProfileStore(
        resumed[FEATURE_PSFM],
        feature_type=FEATURE_PSFM,
    ).load("query", query)


def test_content_addressed_matrix_rejects_existing_wrong_payload(tmp_path):
    matrix = np.full((2, 20), 0.05, dtype=np.float32)
    relative_path, _ = profile_generation._content_addressed_matrix_path(
        tmp_path,
        matrix,
    )
    atomic_save_profile_matrix(
        tmp_path / relative_path,
        np.eye(2, 20, dtype=np.float32),
    )

    with pytest.raises(SequenceProfileError) as exc_info:
        profile_generation._content_addressed_matrix_path(tmp_path, matrix)
    assert exc_info.value.code == "profile_matrix_content_collision"
