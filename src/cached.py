"""Autoregressive generation with a KV cache.

The prompt is processed once (`forward_prefill`), caching each layer's keys and
values. Every subsequent token only computes Q/K/V for the newly generated token
(`forward_step`) and attends it against the growing cache, instead of recomputing the
whole prefix from scratch like `uncached.py` does.
"""
import time

import numpy as np

from attention import embed, feed_forward_block, project_qkv, scaled_dot_product_attention
from data import REFERENCE_SEED, WEIGHT_SEED, make_sequences, make_weights


def forward_prefill(weights, tokens):
    """Process the prompt once: (batch, prompt_len) token ids -> (logits, cache).

    `logits` is (batch, prompt_len, vocab_size). `cache` is a list, one entry per
    layer, of {"K": (batch, prompt_len, d_model), "V": (batch, prompt_len, d_model)}.
    """
    n_heads = weights["config"]["n_heads"]
    x = embed(weights, tokens, pos_offset=0)
    cache = []
    for layer in weights["layers"]:
        Q, K, V = project_qkv(x, layer)
        attn_out = scaled_dot_product_attention(Q, K, V, n_heads)
        x = x + attn_out @ layer["W_o"]
        x = x + feed_forward_block(x, layer)
        cache.append({"K": K, "V": V})
    logits = x @ weights["W_out"] + weights["b_out"]
    return logits, cache


def forward_step(weights, cache, token, pos):
    """One incremental decoding step.

    `token` is (batch, 1) ids for the single new token at absolute position `pos`.
    Only that token's Q/K/V are projected; its K/V are appended to `cache` (returned
    as a new list, `cache` itself is left untouched) before attending. Returns
    (logits, new_cache) where `logits` is (batch, 1, vocab_size).
    """
    n_heads = weights["config"]["n_heads"]
    x = embed(weights, token, pos_offset=pos)
    new_cache = []
    for layer, layer_cache in zip(weights["layers"], cache):
        Q, K_new, V_new = project_qkv(x, layer)
        K = np.concatenate([layer_cache["K"], K_new], axis=1)
        V = np.concatenate([layer_cache["V"], V_new], axis=1)
        attn_out = scaled_dot_product_attention(Q, K, V, n_heads)
        x = x + attn_out @ layer["W_o"]
        x = x + feed_forward_block(x, layer)
        new_cache.append({"K": K, "V": V})
    logits = x @ weights["W_out"] + weights["b_out"]
    return logits, new_cache


def generate_cached(weights, prompt, n_new_tokens):
    """Greedily generate `n_new_tokens` tokens after `prompt` (batch, prompt_len).

    Returns (tokens, step_times, prefill_time_s): the full
    (batch, prompt_len + n_new_tokens) sequence, a list of per-step wall-clock
    seconds for every generated token after the first (each an incremental,
    cached step), and the wall-clock seconds spent processing the prompt once. The
    first new token comes from the prefill pass itself, so `step_times` has
    `n_new_tokens - 1` entries where `generate_uncached`'s has `n_new_tokens`.
    """
    start = time.perf_counter()
    logits, cache = forward_prefill(weights, prompt)
    prefill_time = time.perf_counter() - start

    tokens = prompt.copy()
    next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    tokens = np.concatenate([tokens, next_token], axis=1)

    step_times = []
    pos = prompt.shape[1]
    for _ in range(n_new_tokens - 1):
        start = time.perf_counter()
        logits, cache = forward_step(weights, cache, next_token, pos)
        step_times.append(time.perf_counter() - start)
        next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        tokens = np.concatenate([tokens, next_token], axis=1)
        pos += 1

    return tokens, step_times, prefill_time


if __name__ == "__main__":
    weights = make_weights(seed=WEIGHT_SEED)
    prompt = make_sequences(batch_size=2, seq_len=8, seed=REFERENCE_SEED)
    n_new_tokens = 16

    tokens, step_times, prefill_time = generate_cached(weights, prompt, n_new_tokens)

    print(f"prompt_len={prompt.shape[1]} n_new_tokens={n_new_tokens} batch_size={prompt.shape[0]}")
    print(f"config={weights['config']}")
    print(f"final_sequence_shape={tokens.shape}")
    print(f"prefill_time_s={prefill_time:.6f}")
    for i, t in enumerate(step_times):
        prefix_len = prompt.shape[1] + 1 + i
        print(f"step={i} prefix_len={prefix_len} time_s={t:.6f}")
    print(f"total_time_s={prefill_time + sum(step_times):.6f}")
