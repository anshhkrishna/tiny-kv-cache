"""Synthetic causal token sequences and fixed random transformer weights.

Both are generated in-process from a seeded NumPy RNG: no download, no external
service, no learned parameters. The weights parameterize a small decoder-only,
causal, multi-head self-attention transformer (token embedding, positional
embedding, per-layer Q/K/V/output projections and a two-layer feed-forward
block, final unembedding).
"""
import numpy as np

VOCAB_SIZE = 64
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
D_FF = 64
MAX_SEQ_LEN = 256

DEFAULT_LENGTHS = (4, 8, 16, 32, 64, 128)

REFERENCE_SEED = 0
WEIGHT_SEED = 1


def make_sequences(batch_size, seq_len, seed, vocab_size=VOCAB_SIZE):
    """Return an (batch_size, seq_len) int array of token ids in [0, vocab_size)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab_size, size=(batch_size, seq_len))


def make_sequences_at_lengths(lengths, batch_size, seed, vocab_size=VOCAB_SIZE):
    """Return {length: (batch_size, length) int array}, one seeded draw per length."""
    return {
        length: make_sequences(batch_size, length, seed=seed + length, vocab_size=vocab_size)
        for length in lengths
    }


def _linear_weight(rng, shape):
    fan_in = shape[0]
    return rng.standard_normal(shape) * (fan_in ** -0.5)


def make_weights(
    seed,
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    d_ff=D_FF,
    max_seq_len=MAX_SEQ_LEN,
):
    """Return a fixed random weight dict for a decoder-only transformer.

    Keys: token_emb (vocab_size, d_model), pos_emb (max_seq_len, d_model),
    layers (list of n_layers dicts with W_q/W_k/W_v/W_o (d_model, d_model),
    W1 (d_model, d_ff), b1 (d_ff,), W2 (d_ff, d_model), b2 (d_model,)),
    W_out (d_model, vocab_size), b_out (vocab_size,). d_model must be divisible
    by n_heads.
    """
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    rng = np.random.default_rng(seed)

    layers = []
    for _ in range(n_layers):
        layers.append({
            "W_q": _linear_weight(rng, (d_model, d_model)),
            "W_k": _linear_weight(rng, (d_model, d_model)),
            "W_v": _linear_weight(rng, (d_model, d_model)),
            "W_o": _linear_weight(rng, (d_model, d_model)),
            "W1": _linear_weight(rng, (d_model, d_ff)),
            "b1": np.zeros(d_ff),
            "W2": _linear_weight(rng, (d_ff, d_model)),
            "b2": np.zeros(d_model),
        })

    return {
        "token_emb": _linear_weight(rng, (vocab_size, d_model)),
        "pos_emb": _linear_weight(rng, (max_seq_len, d_model)),
        "layers": layers,
        "W_out": _linear_weight(rng, (d_model, vocab_size)),
        "b_out": np.zeros(vocab_size),
        "config": {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "max_seq_len": max_seq_len,
        },
    }


if __name__ == "__main__":
    seqs = make_sequences_at_lengths(DEFAULT_LENGTHS, batch_size=4, seed=REFERENCE_SEED)
    for length, batch in seqs.items():
        print(f"length={length}: shape={batch.shape} vocab=[0, {VOCAB_SIZE})")
    weights = make_weights(seed=WEIGHT_SEED)
    print(f"weights: {weights['config']}, {len(weights['layers'])} layers")
