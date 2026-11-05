"""Shared policy for the standard amino-acid K-mer tokenizer."""

from numbers import Integral


KMER_MIN_K = 1
KMER_MAX_K = 3
KMER_MAX_LENGTH_UNIT = "tokens"


def validate_kmer_k(k) -> int:
    """Return a validated standard K-mer window size.

    The standard tokenizer eagerly enumerates its full vocabulary, so this
    check must run before any ``itertools.product`` call.
    """
    if isinstance(k, bool) or not isinstance(k, Integral):
        raise ValueError(
            f"K-mer tokenizer k must be an integer in [{KMER_MIN_K}, {KMER_MAX_K}], got {k!r}"
        )
    value = int(k)
    if not KMER_MIN_K <= value <= KMER_MAX_K:
        raise ValueError(
            f"K-mer tokenizer k must be an integer in [{KMER_MIN_K}, {KMER_MAX_K}], got {k!r}"
        )
    return value
