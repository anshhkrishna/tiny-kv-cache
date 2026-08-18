import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attention import forward
from cached import forward_prefill, forward_step
from data import REFERENCE_SEED, WEIGHT_SEED, make_sequences, make_weights
from flops import full_forward_flops, incremental_step_flops
from rigor import ALT_CONFIG, DEFAULT_CONFIG, HIGH_FLOPS, LOW_FLOPS, flops_ratio_exponent


def test_cached_matches_uncached_logits():
    weights = make_weights(seed=WEIGHT_SEED)
    tokens = make_sequences(batch_size=2, seq_len=9, seed=REFERENCE_SEED)

    full_logits = forward(weights, tokens)

    prefill_logits, cache = forward_prefill(weights, tokens[:, :6])
    assert np.allclose(prefill_logits, full_logits[:, :6, :], atol=1e-8)

    step_logits = None
    for pos in range(6, 9):
        step_logits, cache = forward_step(weights, cache, tokens[:, pos:pos + 1], pos=pos)

    assert np.allclose(step_logits[:, -1, :], full_logits[:, -1, :], atol=1e-8)


@pytest.mark.parametrize("config", [DEFAULT_CONFIG, ALT_CONFIG])
def test_flops_ratio_uncached_closer_to_quadratic_than_linear(config):
    exponent = flops_ratio_exponent(full_forward_flops, LOW_FLOPS, HIGH_FLOPS, config)
    assert abs(exponent - 2) < abs(exponent - 1)


@pytest.mark.parametrize("config", [DEFAULT_CONFIG, ALT_CONFIG])
def test_flops_ratio_cached_closer_to_linear_than_quadratic(config):
    exponent = flops_ratio_exponent(incremental_step_flops, LOW_FLOPS, HIGH_FLOPS, config)
    assert abs(exponent - 1) < abs(exponent - 2)
