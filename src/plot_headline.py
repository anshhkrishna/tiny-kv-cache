"""Regenerates results/headline.png from the committed results/run.log and
results/rigor.log -- parses the measured per-step points and fitted exponents those
scripts already printed, rather than re-running the sweep or re-deriving numbers, so
the plot always matches exactly what's checked into the logs.

Run with `python3 src/plot_headline.py`.
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RUN_LOG_PATH = RESULTS_DIR / "run.log"
RIGOR_LOG_PATH = RESULTS_DIR / "rigor.log"
OUT_PATH = RESULTS_DIR / "headline.png"

POINT_RE = re.compile(
    r"^impl=(\S+) prefix_len=(\d+) step_index=(\d+) time_s=([\d.]+) flops=(\d+)$"
)
FIT_RE = re.compile(
    r"^impl=(\S+) metric=wall_clock exponent_b=([\d.]+) a=([\d.eE+-]+) r_squared=([\d.]+)$"
)
EMPIRICAL_CROSSOVER_RE = re.compile(r"^empirical_crossover_prefix_len=(\d+)$")
RIGOR_MEAN_STD_RE = re.compile(r"^impl=(\S+) n=\d+ mean=([\d.]+) std=([\d.]+)$")


def parse_run_log(path=RUN_LOG_PATH):
    """Returns (points, fits, empirical_crossover).

    points: {impl: [(prefix_len, step_index, time_s), ...]}, all rows.
    fits: {impl: (exponent_b, a, r_squared)} for the wall_clock metric.
    empirical_crossover: int prefix_len.
    """
    points = {"uncached": [], "cached": []}
    fits = {}
    crossover = None
    for line in path.read_text().splitlines():
        m = POINT_RE.match(line)
        if m:
            impl, prefix_len, step_index, time_s, _flops = m.groups()
            points[impl].append((int(prefix_len), int(step_index), float(time_s)))
            continue
        m = FIT_RE.match(line)
        if m:
            impl, b, a, r2 = m.groups()
            fits[impl] = (float(b), float(a), float(r2))
            continue
        m = EMPIRICAL_CROSSOVER_RE.match(line)
        if m:
            crossover = int(m.group(1))
    assert points["uncached"] and points["cached"], "no per-step rows parsed"
    assert set(fits) == {"uncached", "cached"}, "missing a wall_clock fit"
    assert crossover is not None, "empirical crossover not found"
    return points, fits, crossover


def parse_rigor_log(path=RIGOR_LOG_PATH):
    """Returns {impl: (mean_exponent, std_exponent)} from the multi-seed/config
    wall-clock exponent summary.
    """
    out = {}
    for line in path.read_text().splitlines():
        m = RIGOR_MEAN_STD_RE.match(line)
        if m:
            impl, mean, std = m.groups()
            out[impl] = (float(mean), float(std))
    assert set(out) == {"uncached", "cached"}, "missing a rigor mean/std line"
    return out


def plot(points, fits, crossover, rigor_stats, out_path=OUT_PATH):
    fig, ax = plt.subplots(figsize=(1600 / 150, 900 / 150), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = {"uncached": "#d62728", "cached": "#1f77b4"}
    labels = {"uncached": "uncached (recomputes full prefix)", "cached": "cached (kv cache)"}

    for impl in ("uncached", "cached"):
        incremental = sorted(p for p in points[impl] if p[1] >= 1)
        xs = [p[0] for p in incremental]
        ys = [p[2] for p in incremental]
        color = colors[impl]
        ax.scatter(xs, ys, color=color, s=22, zorder=3, label=labels[impl])

        b, a, _r2 = fits[impl]
        x_min, x_max = min(xs), max(xs)
        fit_xs = [x_min * (x_max / x_min) ** (i / 50) for i in range(51)]
        fit_ys = [a * x ** b for x in fit_xs]
        ax.plot(fit_xs, fit_ys, color=color, linewidth=1.5, alpha=0.8)

        # Shaded band: same fitted curve, re-sloped by +/- one std of the exponent
        # measured across 3 seeds x 2 model shapes in the rigor sweep (results/rigor.log),
        # anchored at the fit's own value at x_min so the band shows exponent
        # uncertainty rather than a second, disconnected estimate.
        mean_b, std_b = rigor_stats[impl]
        y_anchor = a * x_min ** b
        lo = [y_anchor * (x / x_min) ** (mean_b - std_b) for x in fit_xs]
        hi = [y_anchor * (x / x_min) ** (mean_b + std_b) for x in fit_xs]
        ax.fill_between(fit_xs, lo, hi, color=color, alpha=0.15, zorder=1)

    ax.axvline(crossover, color="black", linestyle=":", linewidth=1.2, alpha=0.6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("prefix length (tokens already generated, log scale)", fontsize=13)
    ax.set_ylabel("wall-clock time per generation step (s, log scale)", fontsize=13)
    ax.set_title(
        "the kv cache overtakes full recomputation past prefix length 10,\n"
        "and the gap keeps widening as the prefix grows",
        fontsize=14,
    )
    ax.tick_params(labelsize=12)
    ax.grid(True, which="major", axis="both", alpha=0.2)
    ax.legend(fontsize=10, loc="upper left")

    y_low, _y_high = ax.get_ylim()
    ax.text(crossover * 1.15, y_low * 1.5, f"crossover ~ prefix len {crossover}",
            fontsize=10, alpha=0.8)

    ax.text(
        0.98, 0.03,
        "shaded band: +/- 1 std of the fitted exponent across 3 seeds x 2 model shapes\n"
        "(results/rigor.log). asymptotic FLOPs-ratio check at far-apart lengths lands at\n"
        "exponent 2.00 (uncached) / 1.00 (cached), the fits above undershoot that within\n"
        "the practically measured range (results/rigor.log).",
        transform=ax.transAxes, fontsize=8.5, ha="right", va="bottom", alpha=0.75,
    )

    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    run_points, run_fits, run_crossover = parse_run_log()
    rigor_stats = parse_rigor_log()
    plot(run_points, run_fits, run_crossover, rigor_stats)
