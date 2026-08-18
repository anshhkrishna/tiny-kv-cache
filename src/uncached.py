"""Naive autoregressive generation: no KV cache.

At every new token, the entire prefix generated so far is re-run through the full
forward pass from scratch. This is the baseline the cached implementation is compared
against.
"""
import time

import numpy as np

from attention import forward
from data import REFERENCE_SEED, WEIGHT_SEED, make_sequences, make_weights


def generate_uncached(weights, prompt, n_new_tokens):
    """Greedily generate `n_new_tokens` tokens after `prompt` (batch, prompt_len).

    Returns (tokens, step_times): the full (batch, prompt_len + n_new_tokens) sequence
    and a list of per-step wall-clock seconds, one per generated token, each timing a
    full forward-pass recomputation over the prefix generated so far.
    """
    tokens = prompt.copy()
    step_times = []
    for _ in range(n_new_tokens):
        start = time.perf_counter()
        logits = forward(weights, tokens)
        step_times.append(time.perf_counter() - start)
        next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        tokens = np.concatenate([tokens, next_token], axis=1)
    return tokens, step_times


if __name__ == "__main__":
    weights = make_weights(seed=WEIGHT_SEED)
    prompt = make_sequences(batch_size=2, seq_len=8, seed=REFERENCE_SEED)
    n_new_tokens = 16

    tokens, step_times = generate_uncached(weights, prompt, n_new_tokens)

    print(f"prompt_len={prompt.shape[1]} n_new_tokens={n_new_tokens} batch_size={prompt.shape[0]}")
    print(f"config={weights['config']}")
    print(f"final_sequence_shape={tokens.shape}")
    for i, t in enumerate(step_times):
        prefix_len = prompt.shape[1] + i
        print(f"step={i} prefix_len={prefix_len} time_s={t:.6f}")
    print(f"total_time_s={sum(step_times):.6f}")
