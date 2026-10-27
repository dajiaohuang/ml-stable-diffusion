import torch

from python_coreml_stable_diffusion.attention import split_einsum, split_einsum_v2


def test_split_einsum_v2_preserves_non_multiple_query_tail():
    torch.manual_seed(0)
    query_length = 513
    q = torch.randn(1, 2, 1, query_length)
    k = torch.randn(1, 2, 1, query_length)
    v = torch.randn(1, 2, 1, query_length)

    expected = split_einsum(q, k, v, None, heads=1, dim_head=2)
    actual = split_einsum_v2(q, k, v, None, heads=1, dim_head=2)

    assert actual.shape == q.shape
    torch.testing.assert_close(actual, expected)
