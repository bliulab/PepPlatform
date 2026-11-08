"""Canonical task names and read compatibility for historical experiments.

Keep legacy spellings here. Resolve names on copies at runtime; never rewrite
sealed metadata, recipes, manifests, data files, or their digests.
"""

from pathlib import Path

TASK_ALIASES = {
    "pep_function_multilabel": "pep_function",
    "pep_hlai_interaction": "pep_hlai",
    "peptide_hlai_interaction": "pep_hlai",
    "pep_mhci": "pep_hlai",
    "pep_mhci_interaction": "pep_hlai",
    "peptide_mhci_interaction": "pep_hlai",
    "pep_hlaii_interaction": "pep_hlaii",
    "peptide_hlaii_interaction": "pep_hlaii",
}
TASK_TYPES = ("pep_function", "pep_protein_interaction", "pep_hlai", "pep_hlaii")
PAIR_TASK_TYPES = frozenset(TASK_TYPES[1:])
LEGACY_HLAI_ROOT = "task_peptide_MHCI_interaction"
LEGACY_HLAI_COLUMN = "mhc"
LEGACY_HLAI_BUILTIN = "builtin-mhci"
LEGACY_INPUT_SCHEMA = "peptide_mhc_pairs"
REMOVED_SEMANTIC_MODES = frozenset({"hla_context", "mhci_anchor", "hlai_anchor"})
REMOVED_SEMANTIC_FIELDS = (
    "hla_context_semantic_margin",
    "mhci_semantic_smoothing_tau",
    "hlai_semantic_smoothing_tau",
)


def normalize_task_type(value):
    task = str(value or "").strip().lower()
    return TASK_ALIASES.get(task, task)


def same_task_type(left, right):
    return normalize_task_type(left) == normalize_task_type(right)


def task_spellings(value):
    canonical = normalize_task_type(value)
    return (canonical, *(key for key, target in TASK_ALIASES.items() if target == canonical))


def normalize_input_schema(value):
    return "peptide_hlai_pairs" if value == LEGACY_INPUT_SCHEMA else value


def builtin_id_spellings(value):
    if value in {"builtin-hlai", LEGACY_HLAI_BUILTIN}:
        other = LEGACY_HLAI_BUILTIN if value == "builtin-hlai" else "builtin-hlai"
        return (value, other)
    return (value,)


def resolve_hlai_root(project_root):
    """Choose the sole available layout; explicit paths bypass this helper."""
    parent = Path(project_root) / "datasets"
    current, historical = parent / "task_peptide_HLAI_interaction", parent / LEGACY_HLAI_ROOT
    if current.exists() and historical.exists():
        raise ValueError(
            "Both current and historical HLA-I dataset roots exist; specify task_root explicitly"
        )
    return historical if historical.exists() else current


def legacy_keyword(kwargs, canonical, value):
    """Accept old keyword arguments without spreading old names into callers."""
    aliases = {
        "hla_pseudo_map": "mhc_pseudo_map",
        "hlai_pseudosequences": "mhci_pseudosequences",
        "require_hlai_pseudoseq": "require_mhci_pseudoseq",
    }
    old = aliases[canonical]
    if old in kwargs:
        legacy = kwargs.pop(old)
        if value is not None and value is not False and legacy is not value and legacy != value:
            raise TypeError(f"conflicting {canonical} and historical keyword")
        value = legacy
    return value


def install_legacy_exports(namespace):
    """Thin aliases for frozen Python scripts and imports."""
    for name, value in tuple(namespace.items()):
        if "hlai" in name and "hlaii" not in name:
            namespace.setdefault(name.replace("hlai", "mhci"), value)
        if "HLAI" in name and "HLAII" not in name:
            namespace.setdefault(name.replace("HLAI", "MHCI"), value)


def add_legacy_semantic_argument(parser):
    import argparse

    parser.add_argument(
        "--mhci_semantic_smoothing_tau", type=float, default=None, help=argparse.SUPPRESS
    )


def normalize_pair_tool_key(value):
    return "hlai" if value == "mhci" else value


def canonical_task_config(config):
    result = dict(config)
    if "task_type" in result:
        result["task_type"] = normalize_task_type(result["task_type"])
    return result


def builtin_name_for_display(name, source, task_type):
    if source == "builtin" and normalize_task_type(task_type) == "pep_hlai":
        import re

        return re.sub(r"MHC[ -]I(?!I)\b", "HLA-I", str(name))
    return name
