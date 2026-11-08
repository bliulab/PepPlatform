import pytest
import torch

from pepplatform.encoders import IndexEncoder, OneHotEncoder


def test_index_encoder_normalizes_native_ids_to_fixed_length():
    encoder = IndexEncoder(max_length=4, vocab_size=5)

    assert encoder.encode([]).tolist() == [0, 0, 0, 0]
    assert encoder.encode([2, 1]).tolist() == [2, 1, 0, 0]
    assert encoder.encode([1, 2, 3, 4, 99]).tolist() == [1, 2, 3, 4]
    assert encoder.encode([2]).dtype == torch.long


def test_index_and_onehot_share_string_token_id_mapping():
    token_ids = {"known": 2}

    def to_indices(tokens):
        return [token_ids.get(token, 1) for token in tokens]

    index = IndexEncoder(max_length=4, vocab_size=3, to_indices=to_indices)
    onehot = OneHotEncoder(max_length=4, vocab_size=3, to_indices=to_indices)

    ids = index.encode(["known", "unknown"])
    features = onehot.encode(["known", "unknown"])

    assert ids.tolist() == [2, 1, 0, 0]
    assert features.dtype == torch.float32
    assert features.shape == (4, 3)
    assert features[:2].argmax(dim=1).tolist() == ids[:2].tolist()
    assert torch.equal(features[:2].sum(dim=1), torch.ones(2))
    assert torch.count_nonzero(features[2:]) == 0


@pytest.mark.parametrize("invalid_id", [-1, 3, 1.5, "1", True])
def test_encoders_reject_invalid_mapped_ids(invalid_id):
    to_indices = lambda _tokens: [invalid_id]

    with pytest.raises(ValueError, match="token id at position 0"):
        IndexEncoder(max_length=2, vocab_size=3, to_indices=to_indices).encode(["x"])
    with pytest.raises(ValueError, match="token id at position 0"):
        OneHotEncoder(max_length=2, vocab_size=3, to_indices=to_indices).encode(["x"])


@pytest.mark.parametrize("field,value", [("max_length", 0), ("vocab_size", 0)])
def test_encoders_require_positive_dimensions(field, value):
    kwargs = {"max_length": 2, "vocab_size": 3}
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        IndexEncoder(**kwargs)
    with pytest.raises(ValueError, match=field):
        OneHotEncoder(**kwargs, to_indices=lambda tokens: tokens)
