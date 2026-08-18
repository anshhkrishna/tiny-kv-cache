"""Sweep uncached vs. cached generation across prefix lengths.

For a range of starting prefix lengths, both implementations generate the same small
number of new tokens from the same prompt and weights. Each generation step is timed
and its analytic FLOP cost computed, giving paired (prefix_len, time, flops) samples
for both implementations across several orders of magnitude of prefix length. A
log-log linear fit on each series gives its empirical scaling exponent, and the two
fits are used to solve for the wall-clock crossover point.

Every measurement is indexed by "prefix length": the number of tokens already in
context before that step computes the next one, and by a step index within its
length's probe run. The step at index zero is the same operation on both sides:
cached.forward_prefill and one uncached.forward call both process the entire starting
prefix from scratch to produce the first new token, so their cost is expected to match
and is reported as a sanity check. It is excluded from the scaling fits and the
crossover: comparing it would compare identical work against itself rather than the
two implementations' differing approach to the tokens after it, so the fits and the
crossover point use only steps at index one and above, where cached does an
incremental step and uncached repeats the full recomputation.
"""
import numpy as np

from cached import generate_cached
from data import REFERENCE_SEED, WEIGHT_SEED, make_sequences, make_weights
from flops import full_forward_flops, incremental_step_flops
from uncached import generate_uncached

PREFIX_LENGTHS = (8, 16, 32, 64, 128, 256, 512, 1024)
N_PROBE_STEPS = 6
REPEATS = 5
BATCH_SIZE = 1


def _median_step_times(step_time_lists):
    """step_time_lists: REPEATS lists of the same length -> per-index median."""
    return np.median(np.array(step_time_lists), axis=0)


def sweep(weights, lengths=PREFIX_LENGTHS, n_probe_steps=N_PROBE_STEPS, repeats=REPEATS):
    """Return {"uncached": [...], "cached": [...]}, each a list of
    (prefix_len, step_index, time_s, flops) tuples, one per probed step at every
    starting length. step_index 0 is the initial full-prefix pass (forward_prefill on
    the cached side, the first forward call on the uncached side); step_index >= 1 is
    a genuine per-token generation step on both sides.
    """
    config = weights["config"]
    points = {"uncached": [], "cached": []}

    for length in lengths:
        prompt = make_sequences(BATCH_SIZE, length, seed=REFERENCE_SEED + length,
                                 vocab_size=config["vocab_size"])

        uncached_trials = []
        for _ in range(repeats):
            _, step_times = generate_uncached(weights, prompt, n_probe_steps)
            uncached_trials.append(step_times)
        uncached_times = _median_step_times(uncached_trials)
        for i, t in enumerate(uncached_times):
            prefix_len = length + i
            flops = full_forward_flops(prefix_len, config, BATCH_SIZE)
            points["uncached"].append((prefix_len, i, float(t), flops))

        cached_trials = []
        for _ in range(repeats):
            _, step_times, prefill_time = generate_cached(weights, prompt, n_probe_steps)
            cached_trials.append([prefill_time] + step_times)
        cached_times = _median_step_times(cached_trials)
        for i, t in enumerate(cached_times):
            prefix_len = length + i
            if i == 0:
                flops = full_forward_flops(prefix_len, config, BATCH_SIZE)
            else:
                flops = incremental_step_flops(prefix_len - 1, config, BATCH_SIZE)
            points["cached"].append((prefix_len, i, float(t), flops))

    return points


def loglog_fit(xy_pairs):
    """xy_pairs: list of (x, y). Fit y = a * x^b via linear regression in log-log
    space. Returns (exponent_b, coefficient_a, r_squared).
    """
    x = np.log(np.array([p[0] for p in xy_pairs], dtype=float))
    y = np.log(np.array([p[1] for p in xy_pairs], dtype=float))
    b, log_a = np.polyfit(x, y, 1)
    y_pred = b * x + log_a
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot
    return float(b), float(np.exp(log_a)), float(r_squared)


def fitted_crossover(uncached_fit, cached_fit):
    """Solve a_u * x^b_u = a_c * x^b_c for x, given each series' power-law fit."""
    b_u, a_u, _ = uncached_fit
    b_c, a_c, _ = cached_fit
    if b_u == b_c:
        return None
    return float((a_c / a_u) ** (1.0 / (b_u - b_c)))


def empirical_crossover(uncached_points, cached_points):
    """Smallest sampled prefix_len, common to both series, at and beyond which cached
    measures faster than uncached at every sampled length. None if no such length was
    sampled. Both point lists are expected pre-filtered to step >= 1.
    """
    uncached_by_len = {p[0]: p[1] for p in uncached_points}
    cached_by_len = {p[0]: p[1] for p in cached_points}
    common_lengths = sorted(set(uncached_by_len) & set(cached_by_len))

    crossover = None
    for length in reversed(common_lengths):
        if cached_by_len[length] < uncached_by_len[length]:
            crossover = length
        else:
            break
    return crossover


def main():
    max_prefix_len = max(PREFIX_LENGTHS) + N_PROBE_STEPS
    weights = make_weights(seed=WEIGHT_SEED, max_seq_len=max_prefix_len)
    print(f"config={weights['config']}")
    print(f"prefix_lengths={PREFIX_LENGTHS} n_probe_steps={N_PROBE_STEPS} "
          f"repeats={REPEATS} batch_size={BATCH_SIZE}")

    points = sweep(weights)

    print("\n--- per-step measurements (median over repeats) ---")
    for impl in ("uncached", "cached"):
        for prefix_len, step_index, t, flops in points[impl]:
            print(f"impl={impl} prefix_len={prefix_len} step_index={step_index} "
                  f"time_s={t:.8f} flops={flops:.0f}")

    incremental = {
        impl: [(p[0], p[2], p[3]) for p in points[impl] if p[1] >= 1]
        for impl in ("uncached", "cached")
    }

    print("\n--- step_index=0 sanity check: prefill cost matches an uncached step ---")
    for length in PREFIX_LENGTHS:
        u = next(p for p in points["uncached"] if p[0] == length and p[1] == 0)
        c = next(p for p in points["cached"] if p[0] == length and p[1] == 0)
        print(f"prefix_len={length} uncached_time_s={u[2]:.8f} "
              f"cached_prefill_time_s={c[2]:.8f} flops_match={u[3] == c[3]}")

    print("\n--- scaling fits over step_index >= 1 (value = a * prefix_len^b) ---")
    fits = {}
    for impl in ("uncached", "cached"):
        time_fit = loglog_fit([(x, t) for x, t, _ in incremental[impl]])
        flops_fit = loglog_fit([(x, f) for x, _, f in incremental[impl]])
        fits[impl] = {"time": time_fit, "flops": flops_fit}
        print(f"impl={impl} metric=wall_clock exponent_b={time_fit[0]:.4f} "
              f"a={time_fit[1]:.6e} r_squared={time_fit[2]:.4f}")
        print(f"impl={impl} metric=flops exponent_b={flops_fit[0]:.4f} "
              f"a={flops_fit[1]:.6e} r_squared={flops_fit[2]:.4f}")

    print("\n--- wall-clock crossover (step_index >= 1 only) ---")
    fitted = fitted_crossover(fits["uncached"]["time"], fits["cached"]["time"])
    empirical = empirical_crossover(
        [(x, t) for x, t, _ in incremental["uncached"]],
        [(x, t) for x, t, _ in incremental["cached"]],
    )
    print(f"fitted_crossover_prefix_len={fitted:.2f}" if fitted is not None
          else "fitted_crossover_prefix_len=undefined (equal exponents)")
    print(f"empirical_crossover_prefix_len={empirical}" if empirical is not None
          else "empirical_crossover_prefix_len=none_sampled (cached never wins "
               "outright in the measured range)")


if __name__ == "__main__":
    main()
