import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import (
    D_FF,
    D_MODEL,
    DEFAULT_LENGTHS,
    MAX_SEQ_LEN,
    N_HEADS,
    N_LAYERS,
    REFERENCE_SEED,
    VOCAB_SIZE,
    WEIGHT_SEED,
    make_sequences,
    make_sequences_at_lengths,
    make_weights,
)


def test_sequence_shape_dtype_and_range():
    seqs = make_sequences(batch_size=8, seq_len=32, seed=REFERENCE_SEED)
    assert seqs.shape == (8, 32)
    assert np.issubdtype(seqs.dtype, np.integer)
    assert seqs.min() >= 0
    assert seqs.max() < VOCAB_SIZE


def test_sequence_determinism():
    a = make_sequences(batch_size=8, seq_len=16, seed=42)
    b = make_sequences(batch_size=8, seq_len=16, seed=42)
    assert np.array_equal(a, b)


def test_sequence_different_seeds_differ():
    a = make_sequences(batch_size=8, seq_len=16, seed=1)
    b = make_sequences(batch_size=8, seq_len=16, seed=2)
    assert not np.array_equal(a, b)


def test_sequences_at_lengths_shapes():
    seqs = make_sequences_at_lengths(DEFAULT_LENGTHS, batch_size=4, seed=REFERENCE_SEED)
    assert set(seqs.keys()) == set(DEFAULT_LENGTHS)
    for length, batch in seqs.items():
        assert batch.shape == (4, length)
        assert batch.min() >= 0
        assert batch.max() < VOCAB_SIZE


def test_sequences_at_lengths_determinism():
    a = make_sequences_at_lengths(DEFAULT_LENGTHS, batch_size=4, seed=7)
    b = make_sequences_at_lengths(DEFAULT_LENGTHS, batch_size=4, seed=7)
    for length in DEFAULT_LENGTHS:
        assert np.array_equal(a[length], b[length])


def test_weights_shapes():
    w = make_weights(seed=WEIGHT_SEED)
    assert w["token_emb"].shape == (VOCAB_SIZE, D_MODEL)
    assert w["pos_emb"].shape == (MAX_SEQ_LEN, D_MODEL)
    assert w["W_out"].shape == (D_MODEL, VOCAB_SIZE)
    assert w["b_out"].shape == (VOCAB_SIZE,)
    assert len(w["layers"]) == N_LAYERS
    for layer in w["layers"]:
        assert layer["W_q"].shape == (D_MODEL, D_MODEL)
        assert layer["W_k"].shape == (D_MODEL, D_MODEL)
        assert layer["W_v"].shape == (D_MODEL, D_MODEL)
        assert layer["W_o"].shape == (D_MODEL, D_MODEL)
        assert layer["W1"].shape == (D_MODEL, D_FF)
        assert layer["b1"].shape == (D_FF,)
        assert layer["W2"].shape == (D_FF, D_MODEL)
        assert layer["b2"].shape == (D_MODEL,)
    assert w["config"]["n_heads"] == N_HEADS


def test_weights_dtype_and_finite():
    w = make_weights(seed=WEIGHT_SEED)
    for arr in (w["token_emb"], w["pos_emb"], w["W_out"], w["b_out"]):
        assert np.issubdtype(arr.dtype, np.floating)
        assert np.isfinite(arr).all()
    for layer in w["layers"]:
        for arr in layer.values():
            assert np.isfinite(arr).all()


def test_weights_determinism():
    a = make_weights(seed=3)
    b = make_weights(seed=3)
    assert np.array_equal(a["token_emb"], b["token_emb"])
    assert np.array_equal(a["pos_emb"], b["pos_emb"])
    for la, lb in zip(a["layers"], b["layers"]):
        for key in la:
            assert np.array_equal(la[key], lb[key])


def test_weights_different_seeds_differ():
    a = make_weights(seed=3)
    b = make_weights(seed=4)
    assert not np.array_equal(a["token_emb"], b["token_emb"])


def test_weights_reject_bad_head_count():
    try:
        make_weights(seed=0, d_model=10, n_heads=3)
        assert False, "expected an assertion error for d_model not divisible by n_heads"
    except AssertionError:
        pass
