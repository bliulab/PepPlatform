"""Shared semantic-relation and semantic-representation utilities.

The module deliberately keeps relation construction separate from model
forward passes.  A relation source and its fixed partner table are built only
after the data split has been frozen, then training code may look up the
relations for anchors in the unchanged main mini-batch.  A trained projection
may also be exported explicitly as a semantic representation; it never replaces
the ordinary predictor embedding or changes classification logits implicitly.
"""

from __future__ import annotations
from pepplatform.task_compat import (
    REMOVED_SEMANTIC_MODES,
    REMOVED_SEMANTIC_FIELDS,
    normalize_task_type,
)


import hashlib
import inspect
import json
import math
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import RandomSampler


SEMANTIC_SCHEMA_VERSION = 2
SEMANTIC_REPRESENTATION_CONTRACT_VERSION = 2
INTRINSIC_LABEL_JACCARD = "intrinsic_label_jaccard"
INTERACTION_PROFILE_JACCARD = "interaction_profile_jaccard"


class SemanticRepresentationUnavailableError(RuntimeError):
    """Raised when an artifact cannot provide the requested semantic vector."""


class RecordingRandomSampler(RandomSampler):
    """RandomSampler with an audit hash for every fully consumed epoch.

    Iteration is delegated to PyTorch's implementation, so adding the audit
    trail does not alter the baseline sampling distribution or RNG calls.
    """

    def __init__(self, data_source, *, generator: torch.Generator, audit_log=None):
        super().__init__(data_source, replacement=False, generator=generator)
        self.audit_log = audit_log if audit_log is not None else []

    def __iter__(self):
        digest = hashlib.sha256()
        count = 0
        for row_index in super().__iter__():
            digest.update(struct.pack("<q", int(row_index)))
            count += 1
            yield row_index
        self.audit_log.append(
            {
                "epoch": len(self.audit_log) + 1,
                "num_rows": count,
                "sha256": digest.hexdigest(),
            }
        )


def _stable_hash_int(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _affine_indices(length: int, *seed_parts: Any) -> Iterator[int]:
    """Yield a deterministic full permutation without allocating it."""

    if length <= 0:
        return
    start = _stable_hash_int(*seed_parts, "start") % length
    if length == 1:
        yield 0
        return
    step = 1 + (_stable_hash_int(*seed_parts, "step") % (length - 1))
    while math.gcd(step, length) != 1:
        step += 1
        if step >= length:
            step = 1
    for offset in range(length):
        yield (start + offset * step) % length


@dataclass(frozen=True)
class SemanticRelation:
    anchor_index: int
    partner_index: int
    similarity: float
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in {"positive", "zero"}:
            raise ValueError(f"unsupported semantic relation kind: {self.kind}")
        if self.anchor_index == self.partner_index:
            raise ValueError("semantic relation cannot point to the anchor itself")
        if not 0.0 <= float(self.similarity) <= 1.0:
            raise ValueError("semantic similarity must be in [0, 1]")
        if self.kind == "positive" and float(self.similarity) <= 0.0:
            raise ValueError("positive semantic relation must have similarity > 0")
        if self.kind == "zero" and float(self.similarity) != 0.0:
            raise ValueError("zero semantic relation must have similarity == 0")


@dataclass(frozen=True)
class SemanticPartnerTable:
    relation_type: str
    num_rows: int
    seed: int
    relations: tuple[SemanticRelation, ...]
    source_summary: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.relation_type not in {
            INTRINSIC_LABEL_JACCARD,
            INTERACTION_PROFILE_JACCARD,
        }:
            raise ValueError(f"unsupported semantic relation type: {self.relation_type}")
        for relation in self.relations:
            if not 0 <= relation.anchor_index < self.num_rows:
                raise ValueError("semantic anchor index is outside the training split")
            if not 0 <= relation.partner_index < self.num_rows:
                raise ValueError("semantic partner index is outside the training split")

    @cached_property
    def relations_by_anchor(self) -> dict[int, tuple[SemanticRelation, ...]]:
        grouped: dict[int, list[SemanticRelation]] = {}
        for relation in self.relations:
            grouped.setdefault(relation.anchor_index, []).append(relation)
        return {anchor: tuple(items) for anchor, items in grouped.items()}

    def relations_for(self, anchor_indices: Iterable[int]) -> tuple[SemanticRelation, ...]:
        grouped = self.relations_by_anchor
        out: list[SemanticRelation] = []
        for anchor in anchor_indices:
            out.extend(grouped.get(int(anchor), ()))
        return tuple(out)

    def unique_row_indices(self, relations: Sequence[SemanticRelation]) -> tuple[int, ...]:
        rows = {relation.anchor_index for relation in relations}
        rows.update(relation.partner_index for relation in relations)
        return tuple(sorted(rows))


class IntrinsicLabelRelationSource:
    relation_type = INTRINSIC_LABEL_JACCARD

    def __init__(self, labels: torch.Tensor | np.ndarray | Sequence[Sequence[float]]):
        array = np.asarray(labels, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2:
            raise ValueError("intrinsic semantic labels must be a 2D matrix")
        self.labels = array > 0.5
        self.label_sets = tuple(frozenset(np.flatnonzero(row).tolist()) for row in self.labels)
        label_to_rows: list[list[int]] = [[] for _ in range(array.shape[1])]
        for row_index, label_set in enumerate(self.label_sets):
            for label_index in label_set:
                label_to_rows[label_index].append(row_index)
        self.label_to_rows = tuple(tuple(rows) for rows in label_to_rows)

    @property
    def num_rows(self) -> int:
        return len(self.label_sets)

    def similarity(self, left_index: int, right_index: int) -> tuple[float, bool]:
        left = self.label_sets[int(left_index)]
        right = self.label_sets[int(right_index)]
        if not left or not right:
            return 0.0, False
        union = left | right
        return float(len(left & right) / len(union)), True

    def positive_partner(self, anchor_index: int, seed: int) -> int | None:
        label_set = self.label_sets[int(anchor_index)]
        if not label_set:
            return None
        labels = tuple(sorted(label_set))
        for label_pos in _affine_indices(
            len(labels), seed, self.relation_type, anchor_index, "label"
        ):
            candidates = self.label_to_rows[labels[label_pos]]
            for candidate_pos in _affine_indices(
                len(candidates), seed, self.relation_type, anchor_index, labels[label_pos], "row"
            ):
                candidate = candidates[candidate_pos]
                if candidate != anchor_index:
                    similarity, valid = self.similarity(anchor_index, candidate)
                    if valid and similarity > 0.0:
                        return candidate
        return None

    def zero_partner(self, anchor_index: int, seed: int) -> int | None:
        anchor_index = int(anchor_index)
        for candidate in _affine_indices(
            self.num_rows,
            seed,
            self.relation_type,
            anchor_index,
            "zero-partner",
        ):
            if candidate == anchor_index:
                continue
            similarity, valid = self.similarity(anchor_index, candidate)
            if valid and similarity == 0.0:
                return candidate
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "relation_type": self.relation_type,
            "num_rows": self.num_rows,
            "num_labels": int(self.labels.shape[1]),
            "rows_with_labels": int(sum(bool(items) for items in self.label_sets)),
        }

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        return {"labels": self.labels.astype(np.uint8, copy=False)}


class EntityProfileRelationSource:
    """Unique-entity interaction-profile relation source.

    Each row is one unique peptide or one unique context. Similarity is the
    Jaccard overlap of that entity's positive interaction neighbors in the
    frozen training split.
    """

    relation_type = INTERACTION_PROFILE_JACCARD

    def __init__(
        self,
        *,
        entity_ids: Sequence[str],
        neighbors: Sequence[Sequence[int]],
        role: str,
    ):
        self.entity_ids = tuple(str(value) for value in entity_ids)
        self.role = str(role)
        self.neighbors = tuple(frozenset(int(value) for value in items) for items in neighbors)
        if len(self.entity_ids) != len(self.neighbors):
            raise ValueError("entity_ids and neighbors must have the same length")

        neighbor_to_entities: dict[int, list[int]] = {}
        for entity_index, neighbor_set in enumerate(self.neighbors):
            for neighbor in neighbor_set:
                neighbor_to_entities.setdefault(int(neighbor), []).append(entity_index)
        self.neighbor_to_entities = {
            int(neighbor): tuple(items) for neighbor, items in neighbor_to_entities.items()
        }

    @property
    def num_rows(self) -> int:
        return len(self.entity_ids)

    @staticmethod
    def _jaccard(left: frozenset[int], right: frozenset[int]) -> tuple[float, bool]:
        if not left or not right:
            return 0.0, False
        return float(len(left & right) / len(left | right)), True

    def similarity(self, left_index: int, right_index: int) -> tuple[float, bool]:
        left_index = int(left_index)
        right_index = int(right_index)
        return self._jaccard(self.neighbors[left_index], self.neighbors[right_index])

    def positive_partner(self, anchor_index: int, seed: int) -> int | None:
        anchor_index = int(anchor_index)
        neighbor_set = self.neighbors[anchor_index]
        if not neighbor_set:
            return None
        sorted_neighbors = tuple(sorted(neighbor_set))
        for neighbor_pos in _affine_indices(
            len(sorted_neighbors),
            seed,
            self.relation_type,
            self.role,
            anchor_index,
            "entity-neighbor",
        ):
            neighbor = sorted_neighbors[neighbor_pos]
            candidates = self.neighbor_to_entities[neighbor]
            for candidate_pos in _affine_indices(
                len(candidates),
                seed,
                self.relation_type,
                self.role,
                anchor_index,
                neighbor,
                "entity-row",
            ):
                candidate = candidates[candidate_pos]
                if candidate == anchor_index:
                    continue
                similarity, valid = self.similarity(anchor_index, candidate)
                if valid and similarity > 0.0:
                    return candidate
        return None

    def zero_partner(self, anchor_index: int, seed: int) -> int | None:
        anchor_index = int(anchor_index)
        for candidate in _affine_indices(
            self.num_rows,
            seed,
            self.relation_type,
            self.role,
            anchor_index,
            "entity-zero",
        ):
            if candidate == anchor_index:
                continue
            similarity, valid = self.similarity(anchor_index, candidate)
            if valid and similarity == 0.0:
                return candidate
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "relation_type": self.relation_type,
            "role": self.role,
            "num_rows": self.num_rows,
            "entities_with_positive_neighbors": int(sum(bool(items) for items in self.neighbors)),
        }

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        neighbor_ids = np.empty(self.num_rows, dtype=object)
        for index, items in enumerate(self.neighbors):
            neighbor_ids[index] = np.asarray(sorted(items), dtype=np.int64)
        return {
            "entity_ids": np.asarray(self.entity_ids, dtype=np.str_),
            "neighbor_ids": neighbor_ids,
            "role": np.asarray(self.role, dtype=np.str_),
        }


class InteractionProfileRelationSource:
    relation_type = INTERACTION_PROFILE_JACCARD

    def __init__(
        self,
        rows: Sequence[Sequence[Any]],
        context_ids: Sequence[str] | None = None,
    ):
        if context_ids is not None and len(context_ids) != len(rows):
            raise ValueError("context_ids must have the same number of rows as the training split")
        peptides = [str(row[1]).strip().upper() for row in rows]
        contexts = [
            str(context_ids[index] if context_ids is not None else row[2]).strip()
            for index, row in enumerate(rows)
        ]
        peptide_values = tuple(sorted(set(peptides)))
        context_values = tuple(sorted(set(contexts)))
        self.peptide_values = peptide_values
        self.context_values = context_values
        peptide_to_id = {value: index for index, value in enumerate(peptide_values)}
        context_to_id = {value: index for index, value in enumerate(context_values)}
        self.row_peptide_ids = tuple(peptide_to_id[value] for value in peptides)
        self.row_context_ids = tuple(context_to_id[value] for value in contexts)

        peptide_neighbors: list[set[int]] = [set() for _ in peptide_values]
        context_neighbors: list[set[int]] = [set() for _ in context_values]
        positive_edges: set[tuple[int, int]] = set()
        pair_labels: dict[tuple[int, int], int] = {}
        for row_index, row in enumerate(rows):
            label = int(float(row[3]))
            edge = (self.row_peptide_ids[row_index], self.row_context_ids[row_index])
            previous = pair_labels.get(edge)
            if previous is not None and previous != label:
                raise ValueError("conflicting labels found for the same peptide-context pair")
            pair_labels[edge] = label
            if label == 1:
                positive_edges.add(edge)
        for peptide_id, context_id in positive_edges:
            peptide_neighbors[peptide_id].add(context_id)
            context_neighbors[context_id].add(peptide_id)
        self.peptide_neighbors = tuple(frozenset(items) for items in peptide_neighbors)
        self.context_neighbors = tuple(frozenset(items) for items in context_neighbors)
        self.positive_edges = tuple(sorted(positive_edges))

        rows_by_peptide: list[list[int]] = [[] for _ in peptide_values]
        rows_by_context: list[list[int]] = [[] for _ in context_values]
        for row_index, (peptide_id, context_id) in enumerate(
            zip(self.row_peptide_ids, self.row_context_ids)
        ):
            rows_by_peptide[peptide_id].append(row_index)
            rows_by_context[context_id].append(row_index)
        self.rows_by_peptide = tuple(tuple(items) for items in rows_by_peptide)
        self.rows_by_context = tuple(tuple(items) for items in rows_by_context)

    @property
    def num_rows(self) -> int:
        return len(self.row_peptide_ids)

    @staticmethod
    def _jaccard(left: frozenset[int], right: frozenset[int]) -> tuple[float, bool]:
        if not left or not right:
            return 0.0, False
        return float(len(left & right) / len(left | right)), True

    def similarity(self, left_index: int, right_index: int) -> tuple[float, bool]:
        left_index = int(left_index)
        right_index = int(right_index)
        peptide_similarity, peptide_valid = self._jaccard(
            self.peptide_neighbors[self.row_peptide_ids[left_index]],
            self.peptide_neighbors[self.row_peptide_ids[right_index]],
        )
        context_similarity, context_valid = self._jaccard(
            self.context_neighbors[self.row_context_ids[left_index]],
            self.context_neighbors[self.row_context_ids[right_index]],
        )
        values = []
        if peptide_valid:
            values.append(peptide_similarity)
        if context_valid:
            values.append(context_similarity)
        if not values:
            return 0.0, False
        return float(sum(values) / len(values)), True

    def _positive_via_peptide(self, anchor_index: int, seed: int) -> int | None:
        peptide_id = self.row_peptide_ids[anchor_index]
        common_contexts = tuple(sorted(self.peptide_neighbors[peptide_id]))
        for common_pos in _affine_indices(
            len(common_contexts), seed, self.relation_type, anchor_index, "pep-common"
        ):
            common_context = common_contexts[common_pos]
            candidate_peptides = tuple(sorted(self.context_neighbors[common_context]))
            for peptide_pos in _affine_indices(
                len(candidate_peptides),
                seed,
                self.relation_type,
                anchor_index,
                common_context,
                "pep-node",
            ):
                candidate_peptide = candidate_peptides[peptide_pos]
                candidate_rows = self.rows_by_peptide[candidate_peptide]
                for row_pos in _affine_indices(
                    len(candidate_rows),
                    seed,
                    self.relation_type,
                    anchor_index,
                    candidate_peptide,
                    "pep-row",
                ):
                    candidate = candidate_rows[row_pos]
                    if candidate == anchor_index:
                        continue
                    similarity, valid = self.similarity(anchor_index, candidate)
                    if valid and similarity > 0.0:
                        return candidate
        return None

    def _positive_via_context(self, anchor_index: int, seed: int) -> int | None:
        context_id = self.row_context_ids[anchor_index]
        common_peptides = tuple(sorted(self.context_neighbors[context_id]))
        for common_pos in _affine_indices(
            len(common_peptides), seed, self.relation_type, anchor_index, "ctx-common"
        ):
            common_peptide = common_peptides[common_pos]
            candidate_contexts = tuple(sorted(self.peptide_neighbors[common_peptide]))
            for context_pos in _affine_indices(
                len(candidate_contexts),
                seed,
                self.relation_type,
                anchor_index,
                common_peptide,
                "ctx-node",
            ):
                candidate_context = candidate_contexts[context_pos]
                candidate_rows = self.rows_by_context[candidate_context]
                for row_pos in _affine_indices(
                    len(candidate_rows),
                    seed,
                    self.relation_type,
                    anchor_index,
                    candidate_context,
                    "ctx-row",
                ):
                    candidate = candidate_rows[row_pos]
                    if candidate == anchor_index:
                        continue
                    similarity, valid = self.similarity(anchor_index, candidate)
                    if valid and similarity > 0.0:
                        return candidate
        return None

    def positive_partner(self, anchor_index: int, seed: int) -> int | None:
        routes = (self._positive_via_peptide, self._positive_via_context)
        if _stable_hash_int(seed, self.relation_type, anchor_index, "route") % 2:
            routes = tuple(reversed(routes))
        for route in routes:
            candidate = route(int(anchor_index), int(seed))
            if candidate is not None:
                return candidate
        return None

    def zero_partner(self, anchor_index: int, seed: int) -> int | None:
        anchor_index = int(anchor_index)
        for candidate in _affine_indices(
            self.num_rows,
            seed,
            self.relation_type,
            anchor_index,
            "zero-partner",
        ):
            if candidate == anchor_index:
                continue
            similarity, valid = self.similarity(anchor_index, candidate)
            if valid and similarity == 0.0:
                return candidate
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "relation_type": self.relation_type,
            "num_rows": self.num_rows,
            "num_peptides": len(self.peptide_values),
            "num_contexts": len(self.context_values),
            "positive_edges": len(self.positive_edges),
            "peptides_with_positive_neighbors": int(
                sum(bool(items) for items in self.peptide_neighbors)
            ),
            "contexts_with_positive_neighbors": int(
                sum(bool(items) for items in self.context_neighbors)
            ),
        }

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        edges = np.asarray(self.positive_edges, dtype=np.int64).reshape(-1, 2)
        return {
            "positive_edges": edges,
            "row_peptide_ids": np.asarray(self.row_peptide_ids, dtype=np.int64),
            "row_context_ids": np.asarray(self.row_context_ids, dtype=np.int64),
            "peptide_values": np.asarray(self.peptide_values, dtype=np.str_),
            "context_values": np.asarray(self.context_values, dtype=np.str_),
        }

    def peptide_entity_source(self) -> EntityProfileRelationSource:
        return EntityProfileRelationSource(
            entity_ids=self.peptide_values,
            neighbors=self.peptide_neighbors,
            role="peptide",
        )

    def context_entity_source(self) -> EntityProfileRelationSource:
        return EntityProfileRelationSource(
            entity_ids=self.context_values,
            neighbors=self.context_neighbors,
            role="context",
        )


def build_partner_table(
    source: IntrinsicLabelRelationSource
    | InteractionProfileRelationSource
    | EntityProfileRelationSource,
    *,
    seed: int,
    positive_partners_per_anchor: int = 1,
    zero_partners_per_anchor: int = 8,
) -> SemanticPartnerTable:
    if int(positive_partners_per_anchor) < 1:
        raise ValueError("positive_partners_per_anchor must be >= 1")
    if int(zero_partners_per_anchor) < 0:
        raise ValueError("zero_partners_per_anchor must be >= 0")
    if isinstance(
        source,
        (InteractionProfileRelationSource, EntityProfileRelationSource),
    ):
        return _build_interaction_partner_table(
            source,
            seed=seed,
            positive_partners_per_anchor=positive_partners_per_anchor,
            zero_partners_per_anchor=zero_partners_per_anchor,
        )
    relations: list[SemanticRelation] = []
    for anchor_index in range(source.num_rows):
        positive_partners: list[int] = []
        for positive_pos in range(int(positive_partners_per_anchor)):
            candidate = source.positive_partner(
                anchor_index,
                seed + 7919 * (positive_pos + 1),
            )
            if candidate is None or candidate in positive_partners:
                continue
            positive_similarity, positive_valid = source.similarity(anchor_index, candidate)
            if not positive_valid or positive_similarity <= 0.0:
                raise RuntimeError(
                    "semantic positive-partner selection produced an invalid relation"
                )
            positive_partners.append(candidate)
            relations.append(
                SemanticRelation(
                    anchor_index,
                    candidate,
                    positive_similarity,
                    "positive",
                )
            )
        if not positive_partners:
            continue

        zero_partners: list[int] = []
        for zero_pos in range(int(zero_partners_per_anchor)):
            candidate = source.zero_partner(
                anchor_index,
                seed + 104729 * (zero_pos + 1),
            )
            if candidate is None or candidate in zero_partners or candidate in positive_partners:
                continue
            similarity, valid = source.similarity(anchor_index, candidate)
            if valid and similarity == 0.0:
                zero_partners.append(candidate)
                relations.append(SemanticRelation(anchor_index, candidate, 0.0, "zero"))

    return SemanticPartnerTable(
        relation_type=source.relation_type,
        num_rows=source.num_rows,
        seed=int(seed),
        relations=tuple(relations),
        source_summary=source.summary(),
    )


def _interaction_positive_unions(
    source: InteractionProfileRelationSource | EntityProfileRelationSource,
) -> list[set[int]]:
    """Precompute rows/entities that share a positive neighbor with each anchor."""

    unions: list[set[int]] = []
    if isinstance(source, InteractionProfileRelationSource):
        for anchor_index in range(source.num_rows):
            rows: set[int] = set()
            peptide_id = source.row_peptide_ids[anchor_index]
            context_id = source.row_context_ids[anchor_index]
            for context in source.peptide_neighbors[peptide_id]:
                rows.update(source.rows_by_context[context])
            for peptide in source.context_neighbors[context_id]:
                rows.update(source.rows_by_peptide[peptide])
            rows.discard(anchor_index)
            unions.append(rows)
        return unions
    for anchor_index in range(source.num_rows):
        rows = set()
        for neighbor in source.neighbors[anchor_index]:
            rows.update(source.neighbor_to_entities[neighbor])
        rows.discard(anchor_index)
        unions.append(rows)
    return unions


def _interaction_zero_eligible(
    source: InteractionProfileRelationSource | EntityProfileRelationSource,
    anchor_index: int,
    candidate_index: int,
    positive_unions: Sequence[set[int]],
) -> bool:
    """Whether a candidate has a valid, exactly-zero interaction-profile Jaccard."""

    if candidate_index == anchor_index:
        return False
    if candidate_index in positive_unions[anchor_index]:
        return False
    if isinstance(source, InteractionProfileRelationSource):
        peptide_valid = bool(
            source.peptide_neighbors[source.row_peptide_ids[anchor_index]]
            and source.peptide_neighbors[source.row_peptide_ids[candidate_index]]
        )
        context_valid = bool(
            source.context_neighbors[source.row_context_ids[anchor_index]]
            and source.context_neighbors[source.row_context_ids[candidate_index]]
        )
        return peptide_valid or context_valid
    return bool(source.neighbors[anchor_index] and source.neighbors[candidate_index])


def _build_interaction_partner_table(
    source: InteractionProfileRelationSource | EntityProfileRelationSource,
    *,
    seed: int,
    positive_partners_per_anchor: int = 1,
    zero_partners_per_anchor: int = 8,
) -> SemanticPartnerTable:
    """Vectorized-equivalent partner table for interaction-profile sources.

    Positive partners use the exact deterministic ``positive_partner`` logic.
    Zero partners use the same ``_affine_indices`` scan order as the generic
    builder, but zero-eligibility is answered with precomputed positive-neighbor
    unions instead of an O(num_rows) Python similarity scan per candidate.
    """

    positive_unions = _interaction_positive_unions(source)
    num_rows = source.num_rows
    relations: list[SemanticRelation] = []
    for anchor_index in range(num_rows):
        positive_partners: list[int] = []
        for positive_pos in range(int(positive_partners_per_anchor)):
            candidate = source.positive_partner(
                anchor_index,
                seed + 7919 * (positive_pos + 1),
            )
            if candidate is None or candidate in positive_partners:
                continue
            similarity, valid = source.similarity(anchor_index, candidate)
            if not valid or similarity <= 0.0:
                raise RuntimeError(
                    "semantic positive-partner selection produced an invalid relation"
                )
            positive_partners.append(candidate)
            relations.append(
                SemanticRelation(
                    anchor_index,
                    candidate,
                    similarity,
                    "positive",
                )
            )
        if not positive_partners:
            continue
        selected = set(positive_partners)
        for zero_pos in range(int(zero_partners_per_anchor)):
            for candidate in _affine_indices(
                num_rows,
                seed + 104729 * (zero_pos + 1),
                source.relation_type,
                anchor_index,
                "zero-partner",
            ):
                candidate = int(candidate)
                if candidate in selected:
                    continue
                if _interaction_zero_eligible(
                    source,
                    anchor_index,
                    candidate,
                    positive_unions,
                ):
                    selected.add(candidate)
                    relations.append(
                        SemanticRelation(
                            anchor_index,
                            candidate,
                            0.0,
                            "zero",
                        )
                    )
                    break

    return SemanticPartnerTable(
        relation_type=source.relation_type,
        num_rows=num_rows,
        seed=int(seed),
        relations=tuple(relations),
        source_summary=source.summary(),
    )


def placebo_shuffle_table(
    table: SemanticPartnerTable,
    *,
    seed: int,
    tag: str = "placebo",
) -> SemanticPartnerTable:
    """Deterministically replace partner identities while preserving loss shape.

    The returned table keeps the same anchors, relation counts, kinds and
    similarity values; only partner identities are replaced by a deterministic
    seed-based permutation, so any observed effect can be attributed to the
    semantic content of the relations rather than to the auxiliary loss alone.
    """

    true_by_anchor = table.relations_by_anchor
    permutation = list(_affine_indices(table.num_rows, seed, f"{tag}-placebo"))
    if len(permutation) != table.num_rows:
        raise RuntimeError("placebo permutation failed")
    relations: list[SemanticRelation] = []
    for anchor_index in sorted(true_by_anchor):
        offset = 0
        for relation in true_by_anchor[anchor_index]:
            while True:
                candidate = permutation[(anchor_index + offset) % table.num_rows]
                offset += 1
                if candidate != anchor_index:
                    break
            relations.append(
                SemanticRelation(
                    anchor_index,
                    int(candidate),
                    relation.similarity,
                    relation.kind,
                )
            )
    return SemanticPartnerTable(
        relation_type=table.relation_type,
        num_rows=table.num_rows,
        seed=table.seed,
        relations=tuple(relations),
        source_summary=dict(table.source_summary),
    )


def build_dense_hard_partner_table(
    source: IntrinsicLabelRelationSource,
    *,
    seed: int,
    embeddings: np.ndarray,
    positive_cap: int = 16,
    negative_cap: int = 8,
    placebo: bool = False,
) -> SemanticPartnerTable:
    """Build a dense, hard-negative semantic partner table for sequence tasks.

    Positives are all training anchors sharing at least one label with the
    anchor, capped at ``positive_cap`` and ordered by descending label-Jaccard
    (ties broken by ascending row index); each relation keeps its Jaccard
    similarity as the InfoNCE weight.  Negatives are zero-overlap samples with
    the highest cosine similarity in the supplied frozen pooled embeddings,
    capped at ``negative_cap``.  Only the frozen training split participates.

    ``placebo`` keeps the exact loss machinery (counts, kinds and similarity
    values per anchor) but replaces partner identities with a deterministic
    seed-based permutation, so any observed effect can be attributed to the
    semantic content of the relations rather than to the auxiliary loss alone.
    """

    if not isinstance(source, IntrinsicLabelRelationSource):
        raise ValueError("dense-hard partner tables support intrinsic-label relations")
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != source.num_rows:
        raise ValueError("dense-hard embeddings must be a 2D array aligned with relation rows")
    if not np.isfinite(array).all():
        raise ValueError("dense-hard embeddings must be finite")
    if int(positive_cap) < 1:
        raise ValueError("semantic_positive_cap must be >= 1")
    if int(negative_cap) < 0:
        raise ValueError("semantic_negative_cap must be >= 0")

    labels = np.asarray(source.labels, dtype=np.float32)
    row_sums = labels.sum(axis=1, dtype=np.float32)
    normalized = array / (np.linalg.norm(array, axis=1, keepdims=True) + 1e-8)
    num_rows = source.num_rows
    relations: list[SemanticRelation] = []
    chunk = 512
    for start in range(0, num_rows, chunk):
        end = min(start + chunk, num_rows)
        block_labels = labels[start:end]
        intersection = block_labels @ labels.T
        union = row_sums[start:end, None] + row_sums[None, :] - intersection
        jaccard = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float32),
            where=union > 0,
        )
        cosine = normalized[start:end] @ normalized.T
        for local_index, anchor_index in enumerate(range(start, end)):
            positive_mask = jaccard[local_index] > 0
            positive_mask[anchor_index] = False
            positive_candidates = np.flatnonzero(positive_mask)
            if positive_candidates.size == 0:
                continue
            positive_order = positive_candidates[
                np.lexsort(
                    (
                        positive_candidates,
                        -jaccard[local_index, positive_candidates],
                    )
                )
            ][: int(positive_cap)]
            for partner_index in positive_order.tolist():
                relations.append(
                    SemanticRelation(
                        anchor_index,
                        int(partner_index),
                        float(jaccard[local_index, int(partner_index)]),
                        "positive",
                    )
                )
            if int(negative_cap) <= 0:
                continue
            zero_mask = (jaccard[local_index] == 0.0) & (union[local_index] > 0)
            zero_mask[anchor_index] = False
            zero_candidates = np.flatnonzero(zero_mask)
            if zero_candidates.size == 0:
                continue
            negative_order = zero_candidates[
                np.lexsort(
                    (
                        zero_candidates,
                        -cosine[local_index, zero_candidates],
                    )
                )
            ][: int(negative_cap)]
            for partner_index in negative_order.tolist():
                relations.append(
                    SemanticRelation(
                        anchor_index,
                        int(partner_index),
                        0.0,
                        "zero",
                    )
                )

    table = SemanticPartnerTable(
        relation_type=source.relation_type,
        num_rows=num_rows,
        seed=int(seed),
        relations=tuple(relations),
        source_summary=source.summary(),
    )
    if placebo:
        table = placebo_shuffle_table(table, seed=seed, tag="dense-hard-placebo")
    return table


def build_interaction_dense_hard_partner_table(
    source: InteractionProfileRelationSource,
    *,
    seed: int,
    embeddings: np.ndarray,
    positive_cap: int = 16,
    negative_cap: int = 8,
    placebo: bool = False,
) -> SemanticPartnerTable:
    """Build a dense, hard-negative partner table for interaction tasks.

    Positives are training rows sharing at least one positive interaction
    neighbor with the anchor (peptide-side or context-side), capped at
    ``positive_cap`` and ordered by descending exact profile-Jaccard (ties
    broken by ascending row index).  Negatives are zero-overlap rows with the
    highest cosine similarity in the supplied frozen pooled embeddings,
    capped at ``negative_cap``.  Only the frozen training split participates.

    ``placebo`` keeps the exact loss machinery (counts, kinds and similarity
    values per anchor) but replaces partner identities with a deterministic
    seed-based permutation, so any observed effect can be attributed to the
    semantic content of the relations rather than to the auxiliary loss alone.
    """

    if not isinstance(source, InteractionProfileRelationSource):
        raise ValueError(
            "interaction dense-hard partner tables require an InteractionProfileRelationSource"
        )
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != source.num_rows:
        raise ValueError("dense-hard embeddings must be a 2D array aligned with relation rows")
    if not np.isfinite(array).all():
        raise ValueError("dense-hard embeddings must be finite")
    if int(positive_cap) < 1:
        raise ValueError("semantic_positive_cap must be >= 1")
    if int(negative_cap) < 0:
        raise ValueError("semantic_negative_cap must be >= 0")

    normalized = array / (np.linalg.norm(array, axis=1, keepdims=True) + 1e-8)
    num_rows = source.num_rows
    rows_by_peptide = source.rows_by_peptide
    rows_by_context = source.rows_by_context
    peptide_neighbors = source.peptide_neighbors
    context_neighbors = source.context_neighbors
    peptide_nonempty = np.asarray(
        [bool(peptide_neighbors[source.row_peptide_ids[index]]) for index in range(num_rows)],
        dtype=bool,
    )
    context_nonempty = np.asarray(
        [bool(context_neighbors[source.row_context_ids[index]]) for index in range(num_rows)],
        dtype=bool,
    )

    def overlapping_rows(anchor_index: int) -> set[int]:
        """Rows sharing a positive neighbor with the anchor on either role."""

        found: set[int] = set()
        peptide_id = source.row_peptide_ids[anchor_index]
        for context_id in peptide_neighbors[peptide_id]:
            for other_peptide in context_neighbors[context_id]:
                for row_index in rows_by_peptide[other_peptide]:
                    if row_index != anchor_index:
                        found.add(int(row_index))
        context_id = source.row_context_ids[anchor_index]
        for peptide_id in context_neighbors[context_id]:
            for other_context in peptide_neighbors[peptide_id]:
                for row_index in rows_by_context[other_context]:
                    if row_index != anchor_index:
                        found.add(int(row_index))
        return found

    all_rows = np.arange(num_rows, dtype=np.int64)
    relations: list[SemanticRelation] = []
    for anchor_index in range(num_rows):
        candidates = overlapping_rows(anchor_index)
        positives: list[tuple[int, float]] = []
        for candidate_index in sorted(candidates):
            similarity, valid = source.similarity(anchor_index, candidate_index)
            if valid and similarity > 0.0:
                positives.append((int(candidate_index), float(similarity)))
        positives.sort(key=lambda item: (-item[1], item[0]))
        positives = positives[: int(positive_cap)]
        if not positives:
            continue
        for partner_index, similarity in positives:
            relations.append(
                SemanticRelation(
                    anchor_index,
                    partner_index,
                    similarity,
                    "positive",
                )
            )
        if int(negative_cap) <= 0:
            continue

        valid_rows = (peptide_nonempty[anchor_index] & peptide_nonempty) | (
            context_nonempty[anchor_index] & context_nonempty
        )
        excluded = candidates | {anchor_index}
        negative_mask = valid_rows.copy()
        for excluded_index in excluded:
            negative_mask[int(excluded_index)] = False
        negative_indices = all_rows[negative_mask]
        if negative_indices.size == 0:
            continue

        anchor_embedding = normalized[anchor_index]
        scores = anchor_embedding @ normalized.T
        selected = negative_indices[
            np.lexsort(
                (
                    negative_indices,
                    -scores[negative_indices],
                )
            )
        ][: int(negative_cap)]
        for partner_index in selected.tolist():
            relations.append(
                SemanticRelation(
                    anchor_index,
                    int(partner_index),
                    0.0,
                    "zero",
                )
            )

    table = SemanticPartnerTable(
        relation_type=source.relation_type,
        num_rows=num_rows,
        seed=int(seed),
        relations=tuple(relations),
        source_summary=source.summary(),
    )
    if placebo:
        table = placebo_shuffle_table(
            table,
            seed=seed,
            tag="interaction-dense-hard-placebo",
        )
    return table


@dataclass
class SemanticLossOutput:
    loss: torch.Tensor
    anchors: int
    positive_relations: int
    negative_relations: int
    unique_rows: int
    positive_similarity_mean: float
    negative_similarity_mean: float


def semantic_batch_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor | None,
    *,
    temperature: float = 0.07,
    theta_w: float = 0.1,
    mode: str = "multilabel",
) -> tuple[torch.Tensor, dict[str, float]]:
    """In-batch supervised contrastive loss using training samples only.

    ``mode="multilabel"``: for every anchor, positive samples are all other rows
    whose label-set Jaccard similarity ``w = |y_i and y_j| / |y_i or y_j|`` exceeds
    ``theta_w``; each positive pair contributes with weight ``w`` (MultiSupCon).
    Anchors without any positive are excluded from the mean.

    ``mode="simclr"``: the batch must be laid out as ``[v1_batch; v2_batch]``
    (2B rows); the only positive of row ``i`` is ``i + B (mod 2B)`` (the paired
    augmented view of the same source sequence).  Labels are unused.

    Returns ``(mean_loss, stats)`` where stats carries anchor/pair counts and
    mean positive/negative cosine similarity (pre-softmax, unnormalized).
    """
    if mode not in ("multilabel", "simclr"):
        raise ValueError(f"unknown semantic contrastive mode: {mode!r}")
    embeddings = embeddings.float().reshape(embeddings.shape[0], -1)
    normed = F.normalize(embeddings, p=2, dim=1, eps=1e-8)
    batch = normed.shape[0]
    device = normed.device
    sim = normed @ normed.t() / max(float(temperature), 1e-6)
    eye = torch.eye(batch, device=device, dtype=torch.bool)
    sim_masked = sim.masked_fill(eye, float("-inf"))

    if mode == "simclr":
        half = batch // 2
        if 2 * half != batch:
            raise ValueError("simclr mode requires an even batch (paired augmented views)")
        rows = torch.arange(batch, device=device)
        partners = (rows + half) % batch
        pos_logits = sim_masked[rows, partners]
        log_denom = torch.logsumexp(sim_masked, dim=1)
        per_anchor = -(pos_logits - log_denom)
        used = torch.isfinite(per_anchor)
        pos_mask = torch.zeros_like(sim_masked, dtype=torch.bool)
        pos_mask[rows, partners] = True
    else:
        if labels is None or labels.ndim != 2 or labels.shape[0] != batch:
            raise ValueError("multilabel mode requires labels of shape (B, C)")
        y = (labels.detach().float() > 0.5).float()
        inter = y @ y.t()
        union = y.sum(dim=1, keepdim=True) + y.sum(dim=1, keepdim=True).t() - inter
        w = inter / union.clamp(min=1e-6)
        pos_mask = (w > float(theta_w)) & ~eye
        counts = pos_mask.sum(dim=1)
        used = counts > 0
        if not bool(used.any()):
            return (
                embeddings.sum() * 0.0,
                {
                    "anchors": 0,
                    "positive_pairs": 0,
                    "mean_pos_sim": 0.0,
                    "mean_neg_sim": 0.0,
                },
            )
        log_denom = torch.logsumexp(sim_masked, dim=1, keepdim=True)
        logits = sim_masked - log_denom  # diag already -inf
        weighted = (pos_mask.float() * w) * logits
        weighted = weighted.masked_fill(eye, 0.0)  # 0 * (-inf) = NaN on the diagonal
        per_anchor = -(weighted.sum(dim=1) / counts.float().clamp(min=1.0))
        per_anchor = per_anchor.masked_fill(~used, 0.0)

    loss = per_anchor[used].mean()
    pos_sim = (sim.masked_fill(~pos_mask, 0.0)).sum() / max(float(pos_mask.sum()), 1.0)
    neg_mask = (~pos_mask) & (~eye)
    neg_sim = (sim.masked_fill(~neg_mask, 0.0)).sum() / max(float(neg_mask.sum()), 1.0)
    stats = {
        "anchors": int(used.sum()),
        "positive_pairs": int(pos_mask.sum()),
        "mean_pos_sim": float(pos_sim.item()) * float(temperature),
        "mean_neg_sim": float(neg_sim.item()) * float(temperature),
    }
    return loss, stats


def semantic_contrastive_loss(
    embeddings: torch.Tensor,
    row_indices: Sequence[int] | torch.Tensor,
    relations: Sequence[SemanticRelation],
    *,
    temperature: float = 0.1,
    min_negatives: int = 1,
    weight_positives: bool = False,
    loss_type: str = "legacy",
) -> SemanticLossOutput:
    # Preserve the historical objective, including scheme-specific exponent
    # weighting, unless the new objective is explicitly requested.
    if loss_type == "jaccard_supcon":
        return semantic_jaccard_supcon_loss(
            embeddings,
            row_indices,
            relations,
            temperature=temperature,
            min_negatives=min_negatives,
        )
    if loss_type != "legacy":
        raise ValueError(f"unknown semantic loss type: {loss_type!r}")
    if float(temperature) <= 0.0:
        raise ValueError("semantic contrastive temperature must be > 0")
    if embeddings.ndim < 2:
        raise ValueError("semantic embeddings must include a sample and feature dimension")
    embeddings = embeddings.float().reshape(embeddings.shape[0], -1)
    normalized = F.normalize(embeddings, p=2, dim=1, eps=1e-8)
    if isinstance(row_indices, torch.Tensor):
        row_values = [int(value) for value in row_indices.detach().cpu().view(-1).tolist()]
    else:
        row_values = [int(value) for value in row_indices]
    if len(row_values) != normalized.shape[0]:
        raise ValueError("row_indices must align with semantic embeddings")
    row_to_position = {row_index: position for position, row_index in enumerate(row_values)}

    grouped: dict[int, dict[str, list[tuple[int, float]]]] = {}
    for relation in relations:
        if (
            relation.anchor_index not in row_to_position
            or relation.partner_index not in row_to_position
        ):
            raise ValueError("semantic relation references a row that was not encoded")
        bucket = grouped.setdefault(relation.anchor_index, {"positive": [], "negative": []})
        if relation.kind == "positive":
            weight = float(relation.similarity) if weight_positives else 1.0
            if weight > 0.0:
                bucket["positive"].append((relation.partner_index, weight))
        else:
            bucket["negative"].append((relation.partner_index, 1.0))

    anchor_losses: list[torch.Tensor] = []
    positive_similarities: list[float] = []
    negative_similarities: list[float] = []
    used_negative_count = 0
    for anchor_index in sorted(grouped):
        positive = grouped[anchor_index]["positive"]
        if not positive:
            continue
        anchor_emb = normalized[row_to_position[anchor_index]]
        positive_pairs = list(dict.fromkeys((partner, weight) for partner, weight in positive))
        positive_indices = [partner for partner, _ in positive_pairs]
        positive_weights = [weight for _, weight in positive_pairs]
        positive_embs = torch.stack(
            [normalized[row_to_position[index]] for index in positive_indices]
        )
        negative_indices = list(
            dict.fromkeys(partner for partner, _ in grouped[anchor_index]["negative"])
        )
        if len(negative_indices) < int(min_negatives):
            available = [
                index
                for index in row_values
                if index != anchor_index and index not in positive_indices
            ]
            for index in available:
                if len(negative_indices) >= int(min_negatives):
                    break
                if index not in negative_indices:
                    negative_indices.append(index)
        if not negative_indices:
            negative_indices = [
                index
                for index in row_values
                if index != anchor_index and index not in positive_indices
            ]
        if not negative_indices:
            continue

        used_negative_count += len(negative_indices)
        negative_embs = torch.stack(
            [normalized[row_to_position[index]] for index in negative_indices]
        )
        positive_logits = (anchor_emb @ positive_embs.T) / float(temperature)
        negative_logits = (anchor_emb @ negative_embs.T) / float(temperature)
        positive_exp = torch.exp(positive_logits)
        if weight_positives:
            weights = torch.as_tensor(
                positive_weights,
                dtype=positive_exp.dtype,
                device=positive_exp.device,
            )
            positive_exp = positive_exp * weights
        denominator = positive_exp.sum() + torch.exp(negative_logits).sum()
        anchor_loss = -torch.log(positive_exp.sum() / denominator)
        anchor_losses.append(anchor_loss)
        positive_similarities.extend(
            float(value) for value in (anchor_emb @ positive_embs.T).detach().cpu().tolist()
        )
        negative_similarities.extend(
            float(value) for value in (anchor_emb @ negative_embs.T).detach().cpu().tolist()
        )

    if anchor_losses:
        loss = torch.stack(anchor_losses).mean()
    else:
        loss = normalized.sum() * 0.0

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return SemanticLossOutput(
        loss=loss,
        anchors=len(anchor_losses),
        positive_relations=sum(len(grouped[anchor]["positive"]) for anchor in sorted(grouped)),
        negative_relations=used_negative_count,
        unique_rows=len(row_values),
        positive_similarity_mean=mean(positive_similarities),
        negative_similarity_mean=mean(negative_similarities),
    )


def semantic_jaccard_supcon_loss(
    embeddings: torch.Tensor,
    row_indices: Sequence[int] | torch.Tensor,
    relations: Sequence[SemanticRelation],
    *,
    temperature: float = 0.1,
    min_negatives: int = 1,
) -> SemanticLossOutput:
    """Jaccard-weighted per-positive log probabilities over existing partners.

    Divide by the number of positives, never by the sum of their weights.
    Candidate selection, negative fallback and anchor reduction deliberately
    match the legacy loss. No label/profile similarities are recomputed here.
    """
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("semantic contrastive temperature must be > 0")
    if embeddings.ndim < 2:
        raise ValueError("semantic embeddings must include a sample and feature dimension")
    embeddings = embeddings.float().reshape(embeddings.shape[0], -1)
    normalized = F.normalize(embeddings, p=2, dim=1, eps=1e-8)
    if isinstance(row_indices, torch.Tensor):
        row_values = [int(value) for value in row_indices.detach().cpu().view(-1).tolist()]
    else:
        row_values = [int(value) for value in row_indices]
    if len(row_values) != normalized.shape[0]:
        raise ValueError("row_indices must align with semantic embeddings")
    row_to_position = {row_index: position for position, row_index in enumerate(row_values)}

    grouped: dict[int, dict[str, list[tuple[int, float]]]] = {}
    for relation in relations:
        if (
            relation.anchor_index not in row_to_position
            or relation.partner_index not in row_to_position
        ):
            raise ValueError("semantic relation references a row that was not encoded")
        bucket = grouped.setdefault(relation.anchor_index, {"positive": [], "negative": []})
        if relation.kind == "positive":
            weight = float(relation.similarity)
            if weight > 0.0:
                bucket["positive"].append((relation.partner_index, weight))
        else:
            bucket["negative"].append((relation.partner_index, 1.0))

    anchor_losses: list[torch.Tensor] = []
    positive_similarities: list[float] = []
    negative_similarities: list[float] = []
    used_negative_count = 0
    for anchor_index in sorted(grouped):
        positive = grouped[anchor_index]["positive"]
        if not positive:
            continue
        anchor_emb = normalized[row_to_position[anchor_index]]
        positive_pairs = list(dict.fromkeys((partner, weight) for partner, weight in positive))
        positive_indices = [partner for partner, _ in positive_pairs]
        positive_weights = [weight for _, weight in positive_pairs]
        positive_embs = torch.stack(
            [normalized[row_to_position[index]] for index in positive_indices]
        )
        negative_indices = list(
            dict.fromkeys(partner for partner, _ in grouped[anchor_index]["negative"])
        )
        if len(negative_indices) < int(min_negatives):
            available = [
                index
                for index in row_values
                if index != anchor_index and index not in positive_indices
            ]
            for index in available:
                if len(negative_indices) >= int(min_negatives):
                    break
                if index not in negative_indices:
                    negative_indices.append(index)
        if not negative_indices:
            negative_indices = [
                index
                for index in row_values
                if index != anchor_index and index not in positive_indices
            ]
        if not negative_indices:
            continue

        used_negative_count += len(negative_indices)
        negative_embs = torch.stack(
            [normalized[row_to_position[index]] for index in negative_indices]
        )
        positive_logits = (anchor_emb @ positive_embs.T) / float(temperature)
        negative_logits = (anchor_emb @ negative_embs.T) / float(temperature)
        log_denom = torch.logsumexp(torch.cat((positive_logits, negative_logits)), dim=0)
        weights = torch.as_tensor(
            positive_weights, dtype=positive_logits.dtype, device=positive_logits.device
        )
        anchor_loss = -(weights * (positive_logits - log_denom)).mean()
        anchor_losses.append(anchor_loss)
        positive_similarities.extend(
            float(value) for value in (anchor_emb @ positive_embs.T).detach().cpu().tolist()
        )
        negative_similarities.extend(
            float(value) for value in (anchor_emb @ negative_embs.T).detach().cpu().tolist()
        )

    if anchor_losses:
        loss = torch.stack(anchor_losses).mean()
    else:
        loss = normalized.sum() * 0.0

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return SemanticLossOutput(
        loss=loss,
        anchors=len(anchor_losses),
        positive_relations=sum(len(grouped[anchor]["positive"]) for anchor in sorted(grouped)),
        negative_relations=used_negative_count,
        unique_rows=len(row_values),
        positive_similarity_mean=mean(positive_similarities),
        negative_similarity_mean=mean(negative_similarities),
    )


class SemanticProjection(torch.nn.Module):
    """Small projection used for semantic contrastive loss and explicit export."""

    def __init__(self, input_dim: int, output_dim: int = 64):
        super().__init__()
        if int(input_dim) <= 0 or int(output_dim) <= 0:
            raise ValueError("semantic projection dimensions must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.output_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.output_dim, self.output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs.float())


def _semantic_projection_owner(model: torch.nn.Module) -> torch.nn.Module:
    """Find the module that owns semantic projectors through transparent wrappers."""

    current = model
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        if any(
            hasattr(current, name)
            for name in (
                "semantic_projector",
                "peptide_semantic_projector",
                "partner_semantic_projector",
            )
        ):
            return current
        nested = getattr(current, "base", None)
        if not isinstance(nested, torch.nn.Module):
            break
        current = nested
    return model


def semantic_representation_contract(
    model: torch.nn.Module,
    *,
    relation_type: str = "",
    pair_mode: str = "fused",
    normalization: str = "l2",
) -> dict[str, Any]:
    """Return the versioned contract for explicitly exportable semantic vectors."""

    if normalization != "l2":
        raise ValueError("semantic representation normalization must be l2")
    owner = _semantic_projection_owner(model)
    projectors = {
        "fused": getattr(owner, "semantic_projector", None),
        "peptide": getattr(owner, "peptide_semantic_projector", None),
        "partner": getattr(owner, "partner_semantic_projector", None),
    }
    dimensions = {
        role: int(projector.output_dim)
        for role, projector in projectors.items()
        if isinstance(projector, SemanticProjection)
    }
    if "fused" not in dimensions:
        raise SemanticRepresentationUnavailableError(
            "model does not contain a fused semantic projection"
        )
    modules = getattr(model, "_modules", {})
    parameters = getattr(model, "_parameters", {})
    residual_enabled = bool(
        isinstance(modules, Mapping)
        and isinstance(modules.get("base"), torch.nn.Module)
        and isinstance(modules.get("residual_projection"), torch.nn.Module)
    )
    learned_gate = bool(residual_enabled and getattr(model, "learned_gate", False))
    configured_weight = float(getattr(model, "weight", 0.0)) if residual_enabled else 0.0
    gate_parameter = parameters.get("gate") if isinstance(parameters, Mapping) else None
    if learned_gate:
        if not isinstance(gate_parameter, torch.Tensor) or gate_parameter.numel() != 1:
            raise SemanticRepresentationUnavailableError(
                "learned semantic residual is missing its scalar gate"
            )
        effective_gate = float(gate_parameter.detach().cpu().item())
        gate_kind = "learned"
    elif residual_enabled:
        effective_gate = configured_weight
        gate_kind = "fixed"
    else:
        effective_gate = 0.0
        gate_kind = "none"

    classification_usage = {
        role: (
            "consumed_via_semantic_residual"
            if role == "fused" and residual_enabled
            else "not_directly_consumed_by_prediction_head"
        )
        for role in sorted(dimensions)
    }
    return {
        "contract_version": SEMANTIC_REPRESENTATION_CONTRACT_VERSION,
        "kind": "supervised_semantic_projection",
        "relation_type": str(relation_type or ""),
        "pair_mode": str(pair_mode or "fused"),
        "roles": sorted(dimensions),
        "dimensions": dimensions,
        "normalization": normalization,
        "classification_usage": classification_usage,
        "residual": {
            "enabled": residual_enabled,
            "adapter": type(model).__name__ if residual_enabled else "",
            "gate_kind": gate_kind,
            "learned_gate": learned_gate,
            "configured_weight": configured_weight,
            "effective_gate": effective_gate,
        },
        "inference_available": True,
    }


def semantic_encode(
    model: torch.nn.Module,
    inputs: Any,
    *,
    role: str = "fused",
    normalize: bool = True,
) -> torch.Tensor:
    """Export a semantic representation without changing prediction behavior.

    The model is evaluated with deterministic BatchNorm/Dropout behavior and its
    original per-module training flags are restored afterwards.  Returned
    vectors remain on the model/input device and are detached from autograd.
    """

    requested_role = str(role or "fused").strip().lower()
    if requested_role not in {"fused", "peptide", "partner"}:
        raise ValueError("semantic representation role must be fused, peptide or partner")
    owner = _semantic_projection_owner(model)
    projector_name = {
        "fused": "semantic_projector",
        "peptide": "peptide_semantic_projector",
        "partner": "partner_semantic_projector",
    }[requested_role]
    projector = getattr(owner, projector_name, None)
    if not isinstance(projector, SemanticProjection):
        raise SemanticRepresentationUnavailableError(
            f"model does not contain a {requested_role} semantic projection"
        )

    with torch.no_grad():
        with auxiliary_eval_forward(model):
            if requested_role == "fused":
                if not hasattr(model, "forward_with_embedding"):
                    raise SemanticRepresentationUnavailableError(
                        "model cannot export a fused predictor embedding"
                    )
                if _embedding_forward_unpacks(model, inputs, "forward_with_embedding"):
                    _logits, embedding = model.forward_with_embedding(*inputs)
                else:
                    _logits, embedding = model.forward_with_embedding(inputs)
            else:
                if not hasattr(model, "forward_with_role_embeddings"):
                    raise SemanticRepresentationUnavailableError(
                        "model cannot export role-specific predictor embeddings"
                    )
                if _embedding_forward_unpacks(model, inputs, "forward_with_role_embeddings"):
                    role_outputs = model.forward_with_role_embeddings(*inputs)
                else:
                    role_outputs = model.forward_with_role_embeddings(inputs)
                if not isinstance(role_outputs, (tuple, list)) or len(role_outputs) < 4:
                    raise SemanticRepresentationUnavailableError(
                        "role-specific embedding forward must return fused, peptide and partner vectors"
                    )
                embedding = role_outputs[2 if requested_role == "peptide" else 3]
            vectors = projector(embedding.float())
            if normalize:
                vectors = F.normalize(vectors, p=2, dim=-1, eps=1e-8)
    return vectors.detach()


def _embedding_forward_unpacks(
    model: torch.nn.Module,
    sample_inputs: Any,
    method_name: str,
) -> bool:
    """Whether a semantic embedding forward should receive sample_inputs unpacked.

    Pair models accept several positional inputs (peptide, partner), while
    sequence models accept a single input that may itself be a tuple-like
    carrier (e.g. ResidueModelInput is a NamedTuple, hence a tuple subclass).
    Decide from the model signature instead of the Python type of sample_inputs.
    """
    if not isinstance(sample_inputs, tuple):
        return False
    method = getattr(model, method_name, None)
    if method is None:
        return False
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return True
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and parameter.default is inspect.Parameter.empty
    ]
    return len(positional) > 1


def infer_semantic_embedding_dims(
    model: torch.nn.Module,
    sample_inputs: Any,
) -> tuple[int, tuple[int, int] | None]:
    """Infer fused and optional role embedding widths from one small batch."""

    with torch.no_grad():
        with auxiliary_eval_forward(model):
            if _embedding_forward_unpacks(model, sample_inputs, "forward_with_embedding"):
                _, fused = model.forward_with_embedding(*sample_inputs)
            else:
                _, fused = model.forward_with_embedding(sample_inputs)
    fused_dim = int(fused.shape[-1])

    role_dims: tuple[int, int] | None = None
    if hasattr(model, "forward_with_role_embeddings"):
        with torch.no_grad():
            with auxiliary_eval_forward(model):
                if _embedding_forward_unpacks(model, sample_inputs, "forward_with_role_embeddings"):
                    _, _, peptide, partner = model.forward_with_role_embeddings(*sample_inputs)
                else:
                    _, _, peptide, partner = model.forward_with_role_embeddings(sample_inputs)
        role_dims = (int(peptide.shape[-1]), int(partner.shape[-1]))
    return fused_dim, role_dims


def prepare_semantic_projectors(
    model: torch.nn.Module,
    args: Any,
    *,
    sample_inputs: Any = None,
) -> None:
    """Attach train-only semantic projection heads before optimizer creation."""

    if not semantic_enabled(args):
        return
    projection_dim = int(getattr(args, "semantic_projection_dim", 64))
    pair_mode = str(getattr(args, "semantic_pair_mode", "role_fused")).strip().lower()
    fused_dim = getattr(model, "semantic_embedding_dim", None)
    role_dims = getattr(model, "semantic_role_embedding_dims", None)
    needs_inference = fused_dim is None or (
        pair_mode in {"role", "role_fused"}
        and role_dims is None
        and hasattr(model, "forward_with_role_embeddings")
    )
    if needs_inference:
        if sample_inputs is None:
            raise RuntimeError(
                "semantic projection dimensions require sample inputs for this model"
            )
        inferred_fused, inferred_role = infer_semantic_embedding_dims(model, sample_inputs)
        if fused_dim is None:
            fused_dim = inferred_fused
        if role_dims is None and hasattr(model, "forward_with_role_embeddings"):
            role_dims = inferred_role
    model.semantic_projector = SemanticProjection(int(fused_dim), projection_dim)
    if pair_mode in {"role", "role_fused"} and role_dims is not None:
        model.peptide_semantic_projector = SemanticProjection(int(role_dims[0]), projection_dim)
        model.partner_semantic_projector = SemanticProjection(int(role_dims[1]), projection_dim)
    # The projection heads are created on the default (CPU) device and attached
    # after model.to(device); move them to the model's device so the auxiliary
    # contrastive forward never mixes devices.
    device = next(model.parameters()).device
    model.semantic_projector = model.semantic_projector.to(device)
    if pair_mode in {"role", "role_fused"} and role_dims is not None:
        model.peptide_semantic_projector = model.peptide_semantic_projector.to(device)
        model.partner_semantic_projector = model.partner_semantic_projector.to(device)


@dataclass(frozen=True)
class SemanticTrainingProfile:
    relation_type: str
    seed: int
    fused_table: SemanticPartnerTable | None
    peptide_table: SemanticPartnerTable | None
    context_table: SemanticPartnerTable | None
    row_peptide_ids: tuple[int, ...]
    row_context_ids: tuple[int, ...]
    peptide_example_rows: tuple[int, ...]
    context_example_rows: tuple[int, ...]
    source_summary: Mapping[str, Any]
    config: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.fused_table and not self.peptide_table and not self.context_table:
            raise ValueError("semantic training profile requires at least one table")

    def active_table_count(self) -> int:
        return (
            int(bool(self.fused_table))
            + int(bool(self.peptide_table))
            + int(bool(self.context_table))
        )


@contextmanager
def auxiliary_eval_forward(module: torch.nn.Module) -> Iterator[None]:
    """Run a differentiable auxiliary forward with eval BN/dropout behavior.

    Per-module training flags and Torch RNG state are restored even when the
    forward raises.  This context never changes autograd enablement.
    """

    training_states = {child: bool(child.training) for child in module.modules()}
    rnn_dropouts = {
        child: float(child.dropout)
        for child in module.modules()
        if isinstance(child, torch.nn.RNNBase)
    }
    cuda_devices = sorted(
        {
            parameter.device.index
            for parameter in module.parameters()
            if parameter.is_cuda and parameter.device.index is not None
        }
    )
    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            module.eval()
            # cuDNN refuses to backpropagate through an RNN graph constructed
            # in eval mode. Keep only RNNBase modules in training mode while
            # neutralizing their internal dropout; BN and ordinary dropout
            # modules retain deterministic eval behavior.
            for child in rnn_dropouts:
                child.training = True
                child.dropout = 0.0
            yield
    finally:
        for child, dropout in rnn_dropouts.items():
            child.dropout = dropout
        for child, training in training_states.items():
            child.training = training


def save_semantic_artifacts(
    output_dir: str | Path,
    *,
    source: IntrinsicLabelRelationSource | InteractionProfileRelationSource,
    table: SemanticPartnerTable,
    split_sample_ids: Mapping[str, Sequence[str]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_path = output / "data_split_indices.npz"
    source_path = output / "semantic_relation_source.npz"
    table_path = output / "semantic_partner_table.npz"
    provenance_path = output / "semantic_provenance.json"

    np.savez_compressed(
        split_path,
        **{
            str(split): np.asarray([str(value) for value in values], dtype=np.str_)
            for split, values in sorted(split_sample_ids.items())
        },
    )
    np.savez_compressed(source_path, **source.artifact_arrays())
    np.savez_compressed(
        table_path,
        anchor_index=np.asarray([item.anchor_index for item in table.relations], dtype=np.int64),
        partner_index=np.asarray([item.partner_index for item in table.relations], dtype=np.int64),
        similarity=np.asarray([item.similarity for item in table.relations], dtype=np.float32),
        kind=np.asarray([item.kind for item in table.relations], dtype=np.str_),
    )
    files = {
        split_path.name: _sha256_file(split_path),
        source_path.name: _sha256_file(source_path),
        table_path.name: _sha256_file(table_path),
    }
    payload = {
        "schema_version": 1,
        "relation_type": table.relation_type,
        "source_split": "train",
        "seed": table.seed,
        "positive_partners_per_anchor": 1,
        "zero_partners_per_anchor": 1,
        "source_summary": dict(table.source_summary),
        "relations": {
            "positive": sum(item.kind == "positive" for item in table.relations),
            "zero": sum(item.kind == "zero" for item in table.relations),
        },
        "config": dict(config),
        "files": files,
    }
    provenance_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["files"][provenance_path.name] = _sha256_file(provenance_path)
    return payload


def _save_partner_table(
    path: Path,
    table: SemanticPartnerTable,
) -> None:
    np.savez_compressed(
        path,
        anchor_index=np.asarray([item.anchor_index for item in table.relations], dtype=np.int64),
        partner_index=np.asarray([item.partner_index for item in table.relations], dtype=np.int64),
        similarity=np.asarray([item.similarity for item in table.relations], dtype=np.float32),
        kind=np.asarray([item.kind for item in table.relations], dtype=np.str_),
    )


def save_semantic_training_profile(
    output_dir: str | Path,
    *,
    profile: SemanticTrainingProfile,
    source: IntrinsicLabelRelationSource | InteractionProfileRelationSource,
    split_sample_ids: Mapping[str, Sequence[str]],
    placebo_reference_table: SemanticPartnerTable | None = None,
    placebo_reference_tables: Mapping[str, SemanticPartnerTable | None] | None = None,
) -> dict[str, Any]:
    """Persist legacy schema-v2 artifacts or explicit new-objective schema v3."""

    output = Path(output_dir)
    if profile.config.get("semantic_loss_type", "legacy") == "jaccard_supcon":
        existing_provenance = output / "semantic_provenance.json"
        if existing_provenance.exists():
            previous = json.loads(existing_provenance.read_text(encoding="utf-8"))
            if previous.get("loss_contract", {}).get("type") != "jaccard_supcon":
                raise ValueError(
                    "jaccard_supcon requires a new artifact directory; "
                    "refusing to overwrite historical semantic artifacts"
                )
    output.mkdir(parents=True, exist_ok=True)
    split_path = output / "data_split_indices.npz"
    source_path = output / "semantic_relation_source.npz"
    fused_table_path = output / "semantic_partner_table.npz"
    placebo_reference_paths = {
        "fused": output / "semantic_placebo_reference_table.npz",
        "peptide": output / "semantic_peptide_placebo_reference_table.npz",
        "context": output / "semantic_context_placebo_reference_table.npz",
    }
    peptide_table_path = output / "semantic_peptide_partner_table.npz"
    context_table_path = output / "semantic_context_partner_table.npz"
    role_mapping_path = output / "semantic_role_mapping.npz"
    provenance_path = output / "semantic_provenance.json"

    np.savez_compressed(
        split_path,
        **{
            str(split): np.asarray([str(value) for value in values], dtype=np.str_)
            for split, values in sorted(split_sample_ids.items())
        },
    )
    np.savez_compressed(source_path, **source.artifact_arrays())
    if profile.fused_table is not None:
        _save_partner_table(fused_table_path, profile.fused_table)
    reference_tables = dict(placebo_reference_tables or {})
    if placebo_reference_table is not None:
        existing_fused = reference_tables.get("fused")
        if existing_fused is not None and existing_fused != placebo_reference_table:
            raise ValueError("conflicting fused placebo reference tables")
        reference_tables["fused"] = placebo_reference_table
    unknown_reference_roles = sorted(set(reference_tables) - set(placebo_reference_paths))
    if unknown_reference_roles:
        raise ValueError("unknown placebo reference roles: " + ", ".join(unknown_reference_roles))
    for role, table in reference_tables.items():
        if table is not None:
            _save_partner_table(placebo_reference_paths[role], table)
    if profile.peptide_table is not None:
        _save_partner_table(peptide_table_path, profile.peptide_table)
    if profile.context_table is not None:
        _save_partner_table(context_table_path, profile.context_table)
    np.savez_compressed(
        role_mapping_path,
        row_peptide_ids=np.asarray(profile.row_peptide_ids, dtype=np.int64),
        row_context_ids=np.asarray(profile.row_context_ids, dtype=np.int64),
        peptide_values=np.asarray(getattr(source, "peptide_values", ()), dtype=np.str_),
        context_values=np.asarray(getattr(source, "context_values", ()), dtype=np.str_),
    )

    def table_summary(table: SemanticPartnerTable | None) -> dict[str, Any] | None:
        if table is None:
            return None
        return {
            "num_rows": table.num_rows,
            "positive": sum(item.kind == "positive" for item in table.relations),
            "zero": sum(item.kind == "zero" for item in table.relations),
        }

    def placebo_structure_audit(
        observed: SemanticPartnerTable | None,
        reference: SemanticPartnerTable | None,
    ) -> dict[str, Any] | None:
        if reference is None:
            return None
        if observed is None:
            return {"passed": False, "reason": "observed table is missing"}
        observed_structure = [
            (item.anchor_index, item.kind, float(item.similarity)) for item in observed.relations
        ]
        reference_structure = [
            (item.anchor_index, item.kind, float(item.similarity)) for item in reference.relations
        ]
        changed = sum(
            left.partner_index != right.partner_index
            for left, right in zip(observed.relations, reference.relations)
        )
        passed = bool(
            observed.num_rows == reference.num_rows
            and len(observed.relations) == len(reference.relations)
            and observed_structure == reference_structure
            and all(item.anchor_index != item.partner_index for item in observed.relations)
        )
        return {
            "passed": passed,
            "num_rows_match": observed.num_rows == reference.num_rows,
            "relation_count_match": len(observed.relations) == len(reference.relations),
            "anchor_kind_weight_match": observed_structure == reference_structure,
            "non_self_partners": all(
                item.anchor_index != item.partner_index for item in observed.relations
            ),
            "changed_partner_count": changed,
            "changed_partner_fraction": (
                float(changed / len(observed.relations)) if observed.relations else 0.0
            ),
        }

    files = {
        split_path.name: _sha256_file(split_path),
        source_path.name: _sha256_file(source_path),
        role_mapping_path.name: _sha256_file(role_mapping_path),
    }
    if profile.fused_table is not None:
        files[fused_table_path.name] = _sha256_file(fused_table_path)
    for role, table in reference_tables.items():
        if table is not None:
            path = placebo_reference_paths[role]
            files[path.name] = _sha256_file(path)
    if profile.peptide_table is not None:
        files[peptide_table_path.name] = _sha256_file(peptide_table_path)
    if profile.context_table is not None:
        files[context_table_path.name] = _sha256_file(context_table_path)
    payload = {
        "schema_version": SEMANTIC_SCHEMA_VERSION,
        "loss_mode": "contrastive",
        "relation_type": profile.relation_type,
        "source_split": "train",
        "seed": profile.seed,
        "pair_mode": str(profile.config.get("semantic_pair_mode", "")),
        "loss_space": str(profile.config.get("semantic_loss_space", "projection")),
        "residual_weight": float(profile.config.get("semantic_residual_weight", 0.0)),
        "partner_scheme": str(profile.config.get("semantic_partner_scheme", "fixed")),
        "positive_cap": profile.config.get("semantic_positive_cap"),
        "negative_cap": profile.config.get("semantic_negative_cap"),
        "hard_negative_source": str(profile.config.get("semantic_hard_negative_source", "")),
        "placebo": bool(
            str(profile.config.get("semantic_partner_scheme", "fixed"))
            in {"placebo", "fixed_placebo"}
        ),
        "placebo_source_scheme": (
            "fixed"
            if str(profile.config.get("semantic_partner_scheme", "fixed")) == "fixed_placebo"
            else "dense_hard"
            if str(profile.config.get("semantic_partner_scheme", "fixed")) == "placebo"
            else ""
        ),
        "projection_dim": profile.config.get("semantic_projection_dim"),
        "temperature": profile.config.get("semantic_contrastive_temperature"),
        "positive_partners_per_anchor": profile.config.get("semantic_positive_partners"),
        "zero_partners_per_anchor": profile.config.get("semantic_zero_partners"),
        "source_summary": dict(profile.source_summary),
        "tables": {
            "fused": table_summary(profile.fused_table),
            "peptide": table_summary(profile.peptide_table),
            "context": table_summary(profile.context_table),
            "placebo_reference": table_summary(reference_tables.get("fused")),
            "placebo_references": {
                role: table_summary(reference_tables.get(role))
                for role in ("fused", "peptide", "context")
            },
        },
        "placebo_structure_audit": {
            "fused": placebo_structure_audit(profile.fused_table, reference_tables.get("fused")),
            "peptide": placebo_structure_audit(
                profile.peptide_table, reference_tables.get("peptide")
            ),
            "context": placebo_structure_audit(
                profile.context_table, reference_tables.get("context")
            ),
        },
        "config": dict(profile.config),
        "files": files,
    }
    if profile.config.get("semantic_loss_type", "legacy") == "jaccard_supcon":
        payload["schema_version"] = 3
        payload["loss_contract"] = {
            "type": "jaccard_supcon",
            "version": 1,
            "positive_weight": "relation_similarity",
            "weight_placement": "outside_log",
            "positive_reduction": "count_mean",
            "anchor_reduction": "valid_anchor_mean",
            "denominator": "unweighted_positive_and_negative_exp_sum",
            "candidate_set": "per_anchor_partner_table_with_legacy_negative_fallback",
            "negative_fallback": "auxiliary_rows_excluding_anchor_and_selected_positives",
            "skip_anchor": "no_positive_or_no_available_negative",
            "scope": "semantic_auxiliary_only",
        }
    provenance_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["files"][provenance_path.name] = _sha256_file(provenance_path)
    return payload


def semantic_enabled(args: Any) -> bool:
    return (
        bool(getattr(args, "enable_semantic_similarity", False))
        and float(getattr(args, "semantic_similarity_weight", 0.1)) > 0.0
    )


def normalize_semantic_aliases(args: Any) -> Any:
    """Normalize canonical semantic flags while accepting legacy pair aliases."""

    # Do not materialize a new default in historical serialized namespaces.
    if hasattr(args, "semantic_loss_type"):
        loss_type = str(args.semantic_loss_type).strip().lower()
        if loss_type not in {"legacy", "jaccard_supcon"}:
            raise SystemExit("semantic_loss_type must be one of legacy, jaccard_supcon")
        args.semantic_loss_type = loss_type

    canonical_enable = getattr(args, "enable_semantic_similarity", None)
    legacy_enable = getattr(args, "enable_interaction_semantic_similarity", None)
    if (
        canonical_enable is not None
        and legacy_enable is not None
        and bool(canonical_enable) != bool(legacy_enable)
    ):
        raise SystemExit(
            "enable_semantic_similarity conflicts with legacy enable_interaction_semantic_similarity"
        )
    enabled = bool(canonical_enable if canonical_enable is not None else legacy_enable or False)

    canonical_weight = getattr(args, "semantic_similarity_weight", None)
    legacy_weight = getattr(args, "interaction_semantic_similarity_weight", None)
    if (
        canonical_weight is not None
        and legacy_weight is not None
        and float(canonical_weight) != float(legacy_weight)
    ):
        raise SystemExit(
            "semantic_similarity_weight conflicts with legacy interaction_semantic_similarity_weight"
        )
    weight = float(
        canonical_weight
        if canonical_weight is not None
        else legacy_weight
        if legacy_weight is not None
        else 0.1
    )
    zero_weight = float(getattr(args, "semantic_zero_weight", 0.25))
    if weight < 0.0:
        raise SystemExit("semantic_similarity_weight must be >= 0.0")
    if zero_weight < 0.0:
        raise SystemExit("semantic_zero_weight must be >= 0.0")

    pair_mode = str(getattr(args, "semantic_pair_mode", "role_fused")).strip().lower()
    if pair_mode not in {"fused", "role", "role_fused"}:
        raise SystemExit("semantic_pair_mode must be one of fused, role, role_fused")
    projection_dim = int(getattr(args, "semantic_projection_dim", 64))
    if projection_dim <= 0:
        raise SystemExit("semantic_projection_dim must be > 0")
    temperature = float(getattr(args, "semantic_contrastive_temperature", 0.1))
    if temperature <= 0.0:
        raise SystemExit("semantic_contrastive_temperature must be > 0")
    positive_partners = int(getattr(args, "semantic_positive_partners", 1))
    if positive_partners < 1:
        raise SystemExit("semantic_positive_partners must be >= 1")
    zero_partners = int(getattr(args, "semantic_zero_partners", 8))
    if zero_partners < 0:
        raise SystemExit("semantic_zero_partners must be >= 0")
    loss_space = str(getattr(args, "semantic_loss_space", "projection")).strip().lower()
    if loss_space not in {"projection", "shared"}:
        raise SystemExit("semantic_loss_space must be one of projection, shared")
    residual_weight = float(getattr(args, "semantic_residual_weight", 0.0))
    if residual_weight < 0.0:
        raise SystemExit("semantic_residual_weight must be >= 0.0")
    learned_gate = bool(getattr(args, "semantic_learned_gate", False))
    if learned_gate and residual_weight <= 0.0:
        raise SystemExit("semantic_learned_gate requires semantic_residual_weight > 0")
    partner_scheme = str(getattr(args, "semantic_partner_scheme", "fixed")).strip().lower()
    if partner_scheme not in {"fixed", "fixed_placebo", "dense_hard", "placebo"}:
        raise SystemExit(
            "semantic_partner_scheme must be one of fixed, fixed_placebo, dense_hard, placebo"
        )
    positive_cap = int(getattr(args, "semantic_positive_cap", 16))
    negative_cap = int(getattr(args, "semantic_negative_cap", 8))
    if positive_cap < 1:
        raise SystemExit("semantic_positive_cap must be >= 1")
    if negative_cap < 0:
        raise SystemExit("semantic_negative_cap must be >= 0")
    hard_negative_source = (
        str(getattr(args, "semantic_hard_negative_source", "input_embedding")).strip().lower()
    )
    if hard_negative_source not in {"input_embedding"}:
        raise SystemExit("semantic_hard_negative_source only supports input_embedding")

    legacy_mode = str(getattr(args, "interaction_semantic_mode", "") or "").strip().lower()
    if legacy_mode in REMOVED_SEMANTIC_MODES:
        raise SystemExit(
            f"interaction_semantic_mode={legacy_mode} was removed; use the automatic "
            "interaction_profile_jaccard semantic relation"
        )
    if legacy_mode not in {"", "profile_jaccard"}:
        raise SystemExit("interaction_semantic_mode only accepts the legacy value profile_jaccard")
    for removed in REMOVED_SEMANTIC_FIELDS:
        value = getattr(args, removed, None)
        if value is not None:
            raise SystemExit(f"{removed} was removed with the task-specific semantic modes")

    args.enable_semantic_similarity = enabled
    args.semantic_similarity_weight = weight
    args.semantic_zero_weight = zero_weight
    args.semantic_pair_mode = pair_mode
    args.semantic_projection_dim = projection_dim
    args.semantic_contrastive_temperature = temperature
    args.semantic_positive_partners = positive_partners
    args.semantic_zero_partners = zero_partners
    args.semantic_loss_space = loss_space
    args.semantic_residual_weight = residual_weight
    args.semantic_learned_gate = learned_gate
    args.semantic_partner_scheme = partner_scheme
    args.semantic_positive_cap = positive_cap
    args.semantic_negative_cap = negative_cap
    args.semantic_hard_negative_source = hard_negative_source
    for legacy_field in (
        "enable_interaction_semantic_similarity",
        "interaction_semantic_similarity_weight",
        "interaction_semantic_mode",
        "hla_context_semantic_margin",
        *REMOVED_SEMANTIC_FIELDS,
    ):
        if hasattr(args, legacy_field):
            delattr(args, legacy_field)
    return args


def semantic_relation_type(task_type: str) -> str:
    task_type = normalize_task_type(task_type)
    task = normalize_task_type(task_type or "")
    if normalize_task_type(task) == "pep_function":
        return INTRINSIC_LABEL_JACCARD
    if normalize_task_type(task) in {"pep_protein_interaction", "pep_hlaii", "pep_hlai"}:
        return INTERACTION_PROFILE_JACCARD
    raise ValueError(f"semantic similarity is not defined for task_type={task_type}")
