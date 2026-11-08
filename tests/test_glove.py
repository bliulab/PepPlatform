import math

import numpy as np
import pytest
import torch

from pepplatform.glove import GloveModel, build_glove_corpus, glove_loss, train_glove


def test_glove_corpus_uses_symmetric_inverse_distance_counts():
    corpus = build_glove_corpus([["A", "B", "C"]], window=2, min_count=1)
    entries = {
        (corpus.vocab[word], corpus.vocab[context]): float(count)
        for word, context, count in zip(
            corpus.word_ids.tolist(),
            corpus.context_ids.tolist(),
            corpus.cooccurrences.tolist(),
        )
    }

    assert entries == {
        ("A", "B"): 1.0,
        ("A", "C"): 0.5,
        ("B", "A"): 1.0,
        ("B", "C"): 1.0,
        ("C", "A"): 0.5,
        ("C", "B"): 1.0,
    }


def test_glove_loss_matches_weighted_least_squares_formula():
    cooccurrences = torch.tensor([1.0, 4.0, 100.0])
    predictions = cooccurrences.log() + torch.tensor([1.0, 2.0, 3.0])

    loss = glove_loss(
        predictions,
        cooccurrences,
        x_max=100.0,
        alpha=0.5,
        reduction="sum",
    )

    assert float(loss) == pytest.approx(0.1 * 1.0 + 0.2 * 4.0 + 1.0 * 9.0)


def test_glove_objective_updates_embeddings_and_biases():
    torch.manual_seed(7)
    model = GloveModel(vocab_size=2, vector_size=3)
    prediction = model(torch.tensor([0, 1]), torch.tensor([1, 0]))
    loss = glove_loss(
        prediction,
        torch.tensor([2.0, 3.0]),
        x_max=100.0,
        alpha=0.75,
        reduction="sum",
    )
    loss.backward()

    assert torch.count_nonzero(model.word_embeddings.weight.grad) > 0
    assert torch.count_nonzero(model.context_embeddings.weight.grad) > 0
    assert torch.count_nonzero(model.word_biases.weight.grad) > 0
    assert torch.count_nonzero(model.context_biases.weight.grad) > 0


def test_glove_training_is_deterministic_reduces_loss_and_preserves_dimension():
    sentences = [["A", "B", "A", "C", "B", "A"] for _ in range(8)]
    kwargs = {
        "vector_size": 7,
        "window": 2,
        "min_count": 1,
        "epochs": 60,
        "seed": 42,
        "x_max": 10.0,
        "alpha": 0.75,
        "learning_rate": 0.1,
        "batch_size": 4,
    }

    first = train_glove(sentences, **kwargs)
    second = train_glove(sentences, **kwargs)

    assert first.final_loss < first.initial_loss
    assert set(first.vectors) == {"A", "B", "C"}
    assert all(vector.shape == (7,) for vector in first.vectors.values())
    assert all(
        np.array_equal(first.vectors[token], second.vectors[token]) for token in first.vectors
    )
    assert math.isfinite(first.final_loss)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"x_max": 0.0}, "x_max"),
        ({"alpha": 1.1}, "alpha"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"batch_size": 0}, "batch_size"),
    ],
)
def test_glove_rejects_invalid_hyperparameters(overrides, message):
    kwargs = {
        "vector_size": 2,
        "window": 1,
        "min_count": 1,
        "epochs": 1,
        "seed": 42,
        "x_max": 100.0,
        "alpha": 0.75,
        "learning_rate": 0.05,
        "batch_size": 4,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        train_glove([["A", "B"]], **kwargs)
