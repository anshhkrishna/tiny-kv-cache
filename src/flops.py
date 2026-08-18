"""Analytic FLOP counts for one generation step, mirroring the matmul shapes actually
executed in attention.py / uncached.py / cached.py. A multiply-add is counted as 2
FLOPs, matching the usual `2 * m * n * k` convention for an (m, k) @ (k, n) matmul.
"""


def _matmul_flops(batch, m, n, k):
    return 2 * batch * m * n * k


def full_forward_flops(seq_len, config, batch):
    """FLOPs for one from-scratch forward pass over `seq_len` positions.

    This is what one uncached generation step costs (attention.forward over the whole
    current prefix), and also what cached.forward_prefill costs over the prompt.
    """
    d_model = config["d_model"]
    d_ff = config["d_ff"]
    n_layers = config["n_layers"]
    vocab_size = config["vocab_size"]

    per_layer = (
        3 * _matmul_flops(batch, seq_len, d_model, d_model)  # Q, K, V projections
        + 2 * _matmul_flops(batch, seq_len, seq_len, d_model)  # scores + weighted sum
        + _matmul_flops(batch, seq_len, d_model, d_model)  # output projection
        + _matmul_flops(batch, seq_len, d_ff, d_model)  # feed-forward layer 1
        + _matmul_flops(batch, seq_len, d_model, d_ff)  # feed-forward layer 2
    )
    unembed = _matmul_flops(batch, seq_len, vocab_size, d_model)
    return n_layers * per_layer + unembed


def incremental_step_flops(cache_len, config, batch):
    """FLOPs for one cached decoding step: one new token attended against a cache of
    `cache_len` already-computed keys/values (so `cache_len + 1` keys total).
    """
    d_model = config["d_model"]
    d_ff = config["d_ff"]
    n_layers = config["n_layers"]
    vocab_size = config["vocab_size"]
    kv_len = cache_len + 1

    per_layer = (
        3 * _matmul_flops(batch, 1, d_model, d_model)  # Q, K, V for the new token
        + 2 * _matmul_flops(batch, 1, kv_len, d_model)  # scores + weighted sum
        + _matmul_flops(batch, 1, d_model, d_model)  # output projection
        + _matmul_flops(batch, 1, d_ff, d_model)
        + _matmul_flops(batch, 1, d_model, d_ff)
    )
    unembed = _matmul_flops(batch, 1, vocab_size, d_model)
    return n_layers * per_layer + unembed
