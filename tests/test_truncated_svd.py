from pepplatform.lsa import lsa_primitive_params
from pepplatform.truncated_svd import new_truncated_svd, truncated_svd_params


def test_lsa_and_generic_tsvd_share_the_same_primitive_parameters():
    assert lsa_primitive_params(17) == truncated_svd_params(17)
    assert new_truncated_svd(3, 17).get_params(deep=False) == {
        "n_components": 3,
        **truncated_svd_params(17),
    }
