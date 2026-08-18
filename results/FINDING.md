A causal transformer was implemented twice from scratch in NumPy: once naively, where every
generation step recomputes attention over the entire prefix generated so far, and once with a
key/value cache, where every step only computes the new token's projections and appends them to
the cached keys and values. Both are timed generating tokens after synthetic prompts across
prefix lengths from 8 to 1024. The cache overtakes full recomputation almost immediately, at a
prefix length of 10, and by prefix length 1025 the uncached step measures 0.18423034s against
the cached step's 0.00080213s (`results/run.log`). The surprising part is how far the measured
scaling exponents sit from the textbook values of quadratic (2) for uncached and linear (1) for
cached: fitted directly on the wall-clock numbers over this practical range they land at
1.5698 and 0.2826, undershooting both, because at this model's size the per-token constant cost
is still a real fraction of the length-dependent attention cost even past a thousand tokens.
Evaluating the same complexity claim analytically at lengths far enough apart to leave that
regime, 8192 and 1048576, confirms it cleanly: 1.9964 and 0.9964 (`results/rigor.log`).
