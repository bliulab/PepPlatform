"""A standards-conformant PyTorch implementation of the GloVe objective."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from pepplatform.static_embeddings import (
    STATIC_EMBEDDING_CONTRACT_VERSION,
    validate_seed,
)


GLOVE_ALGORITHM = "glove_weighted_least_squares"
GLOVE_IMPLEMENTATION = "pepplatform_pytorch"
GLOVE_IMPLEMENTATION_VERSION = 2


def glove_cache_signature(
    *,
    x_max: float,
    alpha: float,
    learning_rate: float,
    batch_size: int,
) -> dict[str, object]:
    """Return the semantic fields that identify a reusable GloVe cache."""
    return {
        "contract_version": STATIC_EMBEDDING_CONTRACT_VERSION,
        "algorithm": GLOVE_ALGORITHM,
        "implementation": GLOVE_IMPLEMENTATION,
        "implementation_version": GLOVE_IMPLEMENTATION_VERSION,
        "cooccurrence_weighting": "inverse_distance",
        "vector_combination": "word_plus_context",
        "optimizer": "adagrad",
        "optimizer_initial_accumulator": 1.0,
        "x_max": float(x_max),
        "alpha": float(alpha),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "device": "cpu",
        "workers": 1,
        "deterministic": True,
        "artifact_format": "glove_text",
        "oov_policy": "provided_unk_else_null_vector",
    }


@dataclass(frozen=True)
class GloveCorpus:
    vocab: tuple[str, ...]
    token_counts: tuple[int, ...]
    word_ids: torch.Tensor
    context_ids: torch.Tensor
    cooccurrences: torch.Tensor


@dataclass(frozen=True)
class GloveTrainingResult:
    vectors: dict[str, np.ndarray]
    initial_loss: float
    final_loss: float
    nonzero_cooccurrences: int
    vocab_size: int


class GloveModel(nn.Module):
    def __init__(self, vocab_size: int, vector_size: int):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, vector_size)
        self.context_embeddings = nn.Embedding(vocab_size, vector_size)
        self.word_biases = nn.Embedding(vocab_size, 1)
        self.context_biases = nn.Embedding(vocab_size, 1)

        bound = 0.5 / float(vector_size)
        nn.init.uniform_(self.word_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.context_embeddings.weight, -bound, bound)
        nn.init.zeros_(self.word_biases.weight)
        nn.init.zeros_(self.context_biases.weight)

    def forward(self, word_ids: torch.Tensor, context_ids: torch.Tensor):
        word = self.word_embeddings(word_ids)
        context = self.context_embeddings(context_ids)
        prediction = (word * context).sum(dim=1)
        prediction = prediction + self.word_biases(word_ids).squeeze(1)
        prediction = prediction + self.context_biases(context_ids).squeeze(1)
        return prediction


def _validate_glove_parameters(
    *,
    vector_size: int,
    window: int,
    min_count: int,
    epochs: int,
    x_max: float,
    alpha: float,
    learning_rate: float,
    batch_size: int,
) -> None:
    integer_parameters = {
        "vector_size": vector_size,
        "window": window,
        "min_count": min_count,
        "epochs": epochs,
        "batch_size": batch_size,
    }
    for name, value in integer_parameters.items():
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
            raise ValueError(f"GloVe {name} must be a positive integer.")
    real_parameters = {
        "x_max": x_max,
        "alpha": alpha,
        "learning_rate": learning_rate,
    }
    for name, value in real_parameters.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"GloVe {name} must be a finite number.")
    if int(vector_size) < 1:
        raise ValueError("GloVe vector_size must be >= 1.")
    if int(window) < 1:
        raise ValueError("GloVe window must be >= 1.")
    if int(min_count) < 1:
        raise ValueError("GloVe min_count must be >= 1.")
    if int(epochs) < 1:
        raise ValueError("GloVe epochs must be >= 1.")
    if float(x_max) <= 0.0:
        raise ValueError("GloVe x_max must be > 0.")
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("GloVe alpha must be in (0, 1].")
    if float(learning_rate) <= 0.0:
        raise ValueError("GloVe learning_rate must be > 0.")


def build_glove_corpus(
    sentences: Sequence[Sequence[str]],
    *,
    window: int,
    min_count: int,
) -> GloveCorpus:
    """Build the non-zero, inverse-distance-weighted co-occurrence entries."""
    if isinstance(window, bool) or not isinstance(window, Integral) or int(window) < 1:
        raise ValueError("GloVe window must be >= 1.")
    if isinstance(min_count, bool) or not isinstance(min_count, Integral) or int(min_count) < 1:
        raise ValueError("GloVe min_count must be >= 1.")

    normalized_sentences: list[tuple[str, ...]] = []
    for sentence_index, sentence in enumerate(sentences):
        if isinstance(sentence, (str, bytes)):
            raise ValueError(
                f"GloVe sentence {sentence_index} must be a sequence of tokens, not text"
            )
        normalized = []
        for token_index, token in enumerate(sentence):
            if (
                not isinstance(token, str)
                or not token
                or any(character.isspace() for character in token)
            ):
                raise ValueError(
                    "GloVe tokens must be non-empty strings without whitespace; "
                    f"got {token!r} at sentence {sentence_index}, token {token_index}"
                )
            normalized.append(token)
        normalized_sentences.append(tuple(normalized))

    frequencies = Counter(token for sentence in normalized_sentences for token in sentence)
    vocab = tuple(
        token
        for token, count in sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
        if count >= int(min_count)
    )
    if not vocab:
        raise ValueError("No tokens left after min_count filtering for GloVe.")
    token_to_id = {token: index for index, token in enumerate(vocab)}

    cooccurrences: defaultdict[tuple[int, int], float] = defaultdict(float)
    for sentence in normalized_sentences:
        sentence_length = len(sentence)
        for center_index, word in enumerate(sentence):
            word_id = token_to_id.get(word)
            if word_id is None:
                continue
            left = max(0, center_index - int(window))
            right = min(sentence_length, center_index + int(window) + 1)
            for context_index in range(left, right):
                if context_index == center_index:
                    continue
                context_id = token_to_id.get(sentence[context_index])
                if context_id is None:
                    continue
                distance = abs(center_index - context_index)
                cooccurrences[(word_id, context_id)] += 1.0 / float(distance)

    if not cooccurrences:
        raise ValueError("Co-occurrence data is empty for GloVe.")
    entries = sorted(cooccurrences.items())
    word_ids = torch.tensor([pair[0][0] for pair in entries], dtype=torch.long)
    context_ids = torch.tensor([pair[0][1] for pair in entries], dtype=torch.long)
    counts = torch.tensor([pair[1] for pair in entries], dtype=torch.float32)
    token_counts = tuple(int(frequencies[token]) for token in vocab)
    return GloveCorpus(vocab, token_counts, word_ids, context_ids, counts)


def glove_weighting(cooccurrences: torch.Tensor, *, x_max: float, alpha: float) -> torch.Tensor:
    scaled = (cooccurrences / float(x_max)).pow(float(alpha))
    return torch.clamp(scaled, max=1.0)


def glove_loss(
    predictions: torch.Tensor,
    cooccurrences: torch.Tensor,
    *,
    x_max: float,
    alpha: float,
    reduction: str = "sum",
) -> torch.Tensor:
    weights = glove_weighting(cooccurrences, x_max=x_max, alpha=alpha)
    losses = weights * (predictions - cooccurrences.log()).pow(2)
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    if reduction == "none":
        return losses
    raise ValueError(f"Unsupported GloVe loss reduction: {reduction}")


def train_glove(
    sentences: Sequence[Sequence[str]],
    *,
    vector_size: int,
    window: int,
    min_count: int,
    epochs: int,
    seed: int,
    x_max: float = 100.0,
    alpha: float = 0.75,
    learning_rate: float = 0.05,
    batch_size: int = 4096,
    epoch_callback: Callable[[int, float], None] | None = None,
) -> GloveTrainingResult:
    """Train GloVe on CPU with sparse mini-batches and AdaGrad."""
    _validate_glove_parameters(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        x_max=x_max,
        alpha=alpha,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
    seed = validate_seed(seed)
    corpus = build_glove_corpus(sentences, window=window, min_count=min_count)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = GloveModel(len(corpus.vocab), int(vector_size)).cpu()
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=float(learning_rate),
            initial_accumulator_value=1.0,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        def average_loss() -> float:
            total = 0.0
            count = int(corpus.cooccurrences.numel())
            with torch.no_grad():
                for start in range(0, count, int(batch_size)):
                    stop = min(count, start + int(batch_size))
                    prediction = model(corpus.word_ids[start:stop], corpus.context_ids[start:stop])
                    total += float(
                        glove_loss(
                            prediction,
                            corpus.cooccurrences[start:stop],
                            x_max=x_max,
                            alpha=alpha,
                            reduction="sum",
                        ).item()
                    )
            return total / float(count)

        initial_loss = average_loss()
        entry_count = int(corpus.cooccurrences.numel())
        for epoch in range(int(epochs)):
            order = torch.randperm(entry_count, generator=generator)
            epoch_loss = 0.0
            for start in range(0, entry_count, int(batch_size)):
                batch_indices = order[start : start + int(batch_size)]
                prediction = model(
                    corpus.word_ids[batch_indices],
                    corpus.context_ids[batch_indices],
                )
                loss = glove_loss(
                    prediction,
                    corpus.cooccurrences[batch_indices],
                    x_max=x_max,
                    alpha=alpha,
                    reduction="mean",
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * int(batch_indices.numel())
            if epoch_callback is not None:
                epoch_callback(epoch + 1, epoch_loss / float(entry_count))

        final_loss = average_loss()
        if not math.isfinite(initial_loss) or not math.isfinite(final_loss):
            raise RuntimeError("GloVe training produced a non-finite loss.")
        combined = (
            model.word_embeddings.weight.detach().cpu()
            + model.context_embeddings.weight.detach().cpu()
        ).numpy()
        if not np.isfinite(combined).all():
            raise RuntimeError("GloVe training produced non-finite vectors.")

    vectors = {token: combined[index].copy() for index, token in enumerate(corpus.vocab)}
    return GloveTrainingResult(
        vectors=vectors,
        initial_loss=initial_loss,
        final_loss=final_loss,
        nonzero_cooccurrences=entry_count,
        vocab_size=len(corpus.vocab),
    )
