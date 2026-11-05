"""Resolve HLA-I column aliases and reject unresolved receptor sequences."""


class HLAIInputError(ValueError):
    """An HLA-I input could not supply a partner sequence."""


def hla_allele(row, default=""):
    from pepplatform.task_compat import LEGACY_HLAI_COLUMN

    current = str(row.get("hla", "") or "").strip()
    historical = str(row.get(LEGACY_HLAI_COLUMN, "") or "").strip()
    if "hla" in row and LEGACY_HLAI_COLUMN in row and current != historical:
        raise HLAIInputError("conflicting hla and historical allele columns")
    if "hla" in row or LEGACY_HLAI_COLUMN in row:
        return current or historical
    return default


def require_hlai_partner_sequence(
    partner, *, allele, resolved, source_path, row_number=None, pseudoseq_path=None
):
    """Validate the caller's selected sequence without changing its precedence."""
    location = str(source_path)
    if row_number is not None:
        location += f":row {row_number}"
    mapping = str(pseudoseq_path or "pseudoseqs_decoded.csv")
    if not resolved:
        raise HLAIInputError(
            f"{location}: HLA-I allele {allele!r} has no usable pseudosequence; "
            f"check {mapping} (missing/empty mapping, absent allele, or non-ok status). "
            "Supply an exact mapping or an explicit partner sequence."
        )
    sequence = "".join(str(partner or "").split())
    if not sequence or not sequence.isascii() or not sequence.isalpha():
        raise HLAIInputError(
            f"{location}: HLA-I allele {allele!r} has an invalid partner sequence "
            f"{partner!r}; allele identifiers are not sequence inputs."
        )
    return partner


def canonical_hlai_columns(row):
    """Compare historical CSV rows after their original bytes were verified."""
    from pepplatform.task_compat import LEGACY_HLAI_COLUMN

    hla_allele(row)  # Reject conflicting aliases before choosing a view.
    result = dict(row)
    if LEGACY_HLAI_COLUMN in result:
        result.setdefault("hla", result.pop(LEGACY_HLAI_COLUMN))
    return result
