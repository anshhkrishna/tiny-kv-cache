"""Rigor pass: repeat the uncached-vs-cached scaling comparison across multiple
seeds and model shapes, and check the asymptotic scaling claim directly from FLOP
counts rather than from a single whole-range log-log fit.

The step-5 experiment already showed that fitting one slope across the whole probed
length range blends a small-length regime (where the per-token linear-in-d_model
projection cost is still comparable to the length-dependent attention cost) with the
large-length regime where attention actually dominates, so the fitted exponent
undershoots the asymptotic value in both directions. Two things follow from that here:

- The seed/config sweep below still fits over the same practically-probed range as the
  experiment step, so its mean/std exponents are expected to show the same undershoot,
  honestly, rather than being tuned to look closer to 2 and 1.
- The FLOPs-ratio check isolates the asymptotic regime instead of fitting across it:
  since `flops.py`'s counters are closed-form arithmetic (not a timed matmul), evaluating
  them at two lengths that are both already far past the constant-term-dominated regime
  costs nothing extra and gives a clean read on the claim itself.
"""
import math

import numpy as np

from data import make_weights
from experiment import loglog_fit, sweep
from flops import full_forward_flops, incremental_step_flops

SEEDS = (10, 11, 12)
CONFIGS = {
    "default": dict(d_model=32, n_heads=4, n_layers=2, d_ff=64),
    "alt": dict(d_model=64, n_heads=8, n_layers=1, d_ff=128),
}
LENGTHS = (8, 32, 128, 512)
N_PROBE_STEPS = 4
REPEATS = 3

# Config used by the FLOPs-ratio check below; matches data.py's defaults.
DEFAULT_CONFIG = dict(vocab_size=64, d_model=32, n_heads=4, n_layers=2, d_ff=64)
ALT_CONFIG = dict(vocab_size=64, **CONFIGS["alt"])

# Well-separated (128x) lengths, both far past the constant-dominated regime described
# above, so the implied exponent reads the asymptotic behavior rather than a blend.
LOW_FLOPS = 8192
HIGH_FLOPS = 1048576


def run_one(seed, config_kwargs):
    """Sweep one (seed, model shape) combination and fit its wall-clock exponents."""
    max_len = max(LENGTHS) + N_PROBE_STEPS
    weights = make_weights(seed=seed, max_seq_len=max_len, **config_kwargs)
    points = sweep(weights, lengths=LENGTHS, n_probe_steps=N_PROBE_STEPS, repeats=REPEATS)
    incremental = {
        impl: [(p[0], p[2]) for p in points[impl] if p[1] >= 1]
        for impl in ("uncached", "cached")
    }
    exponents = {}
    for impl in ("uncached", "cached"):
        b, _a, _r2 = loglog_fit(incremental[impl])
        exponents[impl] = b
    return exponents


def flops_ratio_exponent(flop_fn, low, high, config, batch=1):
    """Exponent b such that flop_fn(high) / flop_fn(low) == (high / low) ** b."""
    return math.log(flop_fn(high, config, batch) / flop_fn(low, config, batch)) / math.log(high / low)


def main():
    print(f"seeds={SEEDS} configs={list(CONFIGS)} lengths={LENGTHS} "
          f"n_probe_steps={N_PROBE_STEPS} repeats={REPEATS}")

    results = {impl: [] for impl in ("uncached", "cached")}
    for config_name, config_kwargs in CONFIGS.items():
        for seed in SEEDS:
            exponents = run_one(seed, config_kwargs)
            for impl in ("uncached", "cached"):
                results[impl].append(exponents[impl])
                print(f"config={config_name} seed={seed} impl={impl} "
                      f"exponent_b={exponents[impl]:.4f}")

    print(f"\n--- fitted wall-clock exponent, mean +/- std across "
          f"{len(SEEDS)} seeds x {len(CONFIGS)} configs (lengths={LENGTHS}) ---")
    for impl in ("uncached", "cached"):
        arr = np.array(results[impl])
        print(f"impl={impl} n={len(arr)} mean={arr.mean():.4f} std={arr.std():.4f}")

    print(f"\n--- FLOPs-ratio check, analytic, lengths {LOW_FLOPS} and {HIGH_FLOPS} ---")
    for config_name, config in (("default", DEFAULT_CONFIG), ("alt", ALT_CONFIG)):
        exp_uncached = flops_ratio_exponent(full_forward_flops, LOW_FLOPS, HIGH_FLOPS, config)
        exp_cached = flops_ratio_exponent(incremental_step_flops, LOW_FLOPS, HIGH_FLOPS, config)
        print(f"config={config_name} uncached_exponent={exp_uncached:.4f} "
              f"(closer to 2 than 1: {abs(exp_uncached - 2) < abs(exp_uncached - 1)})")
        print(f"config={config_name} cached_exponent={exp_cached:.4f} "
              f"(closer to 1 than 2: {abs(exp_cached - 1) < abs(exp_cached - 2)})")


if __name__ == "__main__":
    main()
