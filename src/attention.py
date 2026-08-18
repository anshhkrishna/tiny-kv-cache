"""Causal multi-head self-attention forward pass for a decoder-only transformer.

No layernorm, matching the fixed-random-weight design in data.py: nothing here
depends on learned statistics, so the extra normalization would only complicate the
FLOP/timing accounting without changing what's being measured.
"""
import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def relu(x):
    return np.maximum(x, 0.0)


def split_heads(x, n_heads):
    """(batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_head)."""
    batch, seq_len, d_model = x.shape
    d_head = d_model // n_heads
    return x.reshape(batch, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)


def merge_heads(x):
    """(batch, n_heads, seq_len, d_head) -> (batch, seq_len, d_model)."""
    batch, n_heads, seq_len, d_head = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, seq_len, n_heads * d_head)


def project_qkv(x, layer):
    return x @ layer["W_q"], x @ layer["W_k"], x @ layer["W_v"]


def causal_mask(q_len, kv_len):
    """Boolean (q_len, kv_len) mask: True where query i may attend to key j.

    Queries are assumed to be the last q_len positions of a kv_len-length prefix, so
    query i sits at absolute position (kv_len - q_len + i). This covers both full
    recomputation (q_len == kv_len) and single-new-token generation against a growing
    KV cache (q_len == 1, kv_len == prefix length so far).
    """
    offset = kv_len - q_len
    q_idx = np.arange(q_len) + offset
    k_idx = np.arange(kv_len)
    return k_idx[None, :] <= q_idx[:, None]


def scaled_dot_product_attention(Q, K, V, n_heads):
    """Q: (batch, q_len, d_model); K, V: (batch, kv_len, d_model). Causal-masked."""
    Qh, Kh, Vh = split_heads(Q, n_heads), split_heads(K, n_heads), split_heads(V, n_heads)
    d_head = Qh.shape[-1]
    scores = Qh @ Kh.transpose(0, 1, 3, 2) / np.sqrt(d_head)
    mask = causal_mask(Qh.shape[2], Kh.shape[2])
    scores = np.where(mask[None, None, :, :], scores, -np.inf)
    weights = softmax(scores, axis=-1)
    out = weights @ Vh
    return merge_heads(out)


def self_attention_block(x, layer, n_heads):
    Q, K, V = project_qkv(x, layer)
    attn_out = scaled_dot_product_attention(Q, K, V, n_heads)
    return attn_out @ layer["W_o"]


def feed_forward_block(x, layer):
    h = relu(x @ layer["W1"] + layer["b1"])
    return h @ layer["W2"] + layer["b2"]


def embed(weights, tokens, pos_offset=0):
    """tokens: (batch, seq_len) int ids -> (batch, seq_len, d_model)."""
    seq_len = tokens.shape[1]
    tok = weights["token_emb"][tokens]
    pos = weights["pos_emb"][pos_offset:pos_offset + seq_len]
    return tok + pos[None, :, :]


def forward(weights, tokens):
    """Full forward pass over `tokens` from scratch.

    (batch, seq_len) token ids -> (batch, seq_len, vocab_size) logits. Recomputes
    every position's attention over the whole sequence; this is the "rerun everything"
    half of the baseline-vs-cache comparison.
    """
    n_heads = weights["config"]["n_heads"]
    x = embed(weights, tokens, pos_offset=0)
    for layer in weights["layers"]:
        x = x + self_attention_block(x, layer, n_heads)
        x = x + feed_forward_block(x, layer)
    return x @ weights["W_out"] + weights["b_out"]
