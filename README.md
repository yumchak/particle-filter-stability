# Particle Filter Stability

**Supervised research project — Dr Mathieu Gerber, Associate Professor in Statistical Science, University of Bristol**

## Overview

This project studies how well a particle filter approximates the **exact** Kalman filter in
linear-Gaussian state-space models, and how that approximation degrades as the problem is made
harder — by demanding more particles `N`, longer time horizons `T`, more state dimensions `d`, or
more observation dimensions `M`. In a linear-Gaussian model the Kalman filter returns the exact
posterior `p(x_t | y_0:t)`, so it serves as ground truth: any gap between the two filters is Monte
Carlo error and nothing else. That makes this setting the ideal test bed for measuring particle
filter accuracy directly instead of guessing at it.

**Part 1** establishes the method in one dimension. **Part 2** extends it to many dimensions and
asks which dimension actually hurts.

## What's in this repo

| File | Contents |
|---|---|
| `kf_pf_lab.ipynb` | **Part 1.** The 1D Kalman-vs-particle-filter study: simulation, both filters, the PF–KF comparison, stability over time and `N`, ESS and resampling, and a nonlinear example. |
| `md_dimension_study.ipynb` | **Part 2.** The multidimensional study: the three `M` vs `d` regimes, side-by-side comparison, and dimension scaling. |
| `filters.py` | All reusable filter code, imported by both notebooks. NumPy only, no other dependencies. |

Both notebooks are committed **with their outputs**, so every plot and number renders on GitHub
without running anything.

---

# Part 1: The 1D study

```
State (hidden):      x_t = b * x_{t-1} + v_t,   v_t ~ N(0, Q)
Observation (seen):  y_t = a * x_t     + e_t,   e_t ~ N(0, R)
```

Parameters: `b = 0.9`, `a = 1.0`, `Q = 1.0`, `R = 1.0`, `T = 100`, seed `= 42`.

### Key findings

- **The particle filter matches the exact Kalman filter.** At `N = 1000` over `T = 100` steps the
  mean absolute gap is `0.027` and the largest single-step gap is `0.18`, around 1.5% of the scale
  of the signal itself.
- **The error follows the O(1/sqrt(N)) Monte Carlo rate.** The log–log fit gives slope `-0.493`
  against the theoretical `-0.5`: four times the computation for twice the accuracy.
- **ESS-triggered resampling controls particle degeneracy.** Resampling below `N/2` keeps the ESS
  oscillating between roughly `N/2` and `N` (mean `531`, minimum `29`) rather than collapsing.
- **The error stays flat over time**, with the second half of each run within about 10% of the
  first half — preliminary evidence of time-uniform stability.
- **The particle filter works where the Kalman filter cannot.** On `y_t = x_t^2 / 20 + e_t` no exact
  Kalman solution exists. Squaring hides the sign of the state, so the posterior is genuinely
  **bimodal** — a shape no Gaussian filter can represent. The posterior mean is consequently
  uninformative, while the estimate of `|x_t|` tracks the truth and cuts error 56% below baseline.

---

# Part 2: Dimension regimes (M vs d)

This part studies how the particle filter behaves as the observation dimension `M` moves away from
the hidden-state dimension `d`, across the over-observed (`M > d`), equal (`M = d`), and
under-observed (`M < d`) cases, benchmarked against the exact Kalman filter throughout.

```
State (hidden):      x_t = F x_{t-1} + v_t,   v_t ~ N(0, Q),   x_t in R^d
Observation (seen):  y_t = H x_t     + e_t,   e_t ~ N(0, R),   y_t in R^M
```

`F` is built from a coupled tridiagonal base and rescaled to spectral radius exactly `0.9`, so every
model is equally stable regardless of `d`. `H` selects which coordinates are measured, and its shape
is what defines the three regimes.

### Verification

Before any results, two independent checks: the multidimensional code reproduces Part 1 **to machine
precision** at `d = M = 1` (including which time steps triggered resampling), and the Kalman
benchmark itself was verified against a brute-force joint-Gaussian posterior that uses no filtering
recursion at all.

### Key findings

- **All three regimes were stable**, with no error accumulation over `T = 100`.
- **The O(1/sqrt(N)) rate survives in every regime** — fitted slopes `-0.512`, `-0.520`, `-0.507`.
  Being under-observed costs a constant factor of roughly `2x`, **not a worse convergence rate**.
  This is a critical distinction: a worse constant is merely expensive, a worse rate would be fatal.
- **Measured against the posterior's own width, all three regimes performed almost identically** at
  `N = 1000`: relative gaps of `0.050`, `0.050`, `0.057`. The regimes differ far more in how hard
  the *problem* is than in how well the particle filter *approximates* it.
- **`M < d` degrades, but not in the expected way.** The error roughly doubles and concentrates in
  the unmeasured coordinates (`2.6x` the error of measured ones), but there is **no ESS collapse** —
  the cloud is in fact *healthier* here (min ESS `83`) than in the other regimes. What makes `M < d`
  hard is irreducible uncertainty in directions the data never sees, not particle degeneracy.
- **Weight degeneracy is driven by `M`, not by `d`.** This was the clearest result. Growing `d` from
  `1` to `16` with `M = 1` left the cloud healthy throughout (min ESS never below `~90`, mean ESS
  pinned near `530`) and the relative gap crept only from `0.033` to `0.064` before flattening.
  Growing `M` from `1` to `16` with `d = 4` drove the minimum ESS from `94` down to `1.3`, forced
  resampling at every single step, and quadrupled the relative gap to `0.22`.
- **More information can make the particle filter worse.** Adding observations sharpens the
  likelihood, and a sharper likelihood is harder for particles proposed from the prior to satisfy.
  So `M > d` makes the *exact posterior* tighter while making the *particle cloud* sicker — the
  hypothesis recorded at the end of Part 1, now confirmed empirically.

### Summary table (N = 1000, T = 100)

| Case | d | M | mean gap | max gap | min ESS | resampled | posterior spread | relative gap |
|---|---|---|---|---|---|---|---|---|
| `M = d` | 2 | 2 | 0.0371 | 0.1736 | 29.3 | 69/100 | 0.7501 | 0.0495 |
| `M > d` | 2 | 4 | 0.0297 | 0.1245 | 15.6 | 97/100 | 0.5924 | 0.0502 |
| `M < d` | 4 | 2 | 0.0628 | 0.1619 | 82.7 | 77/100 | 1.1040 | 0.0568 |

"Relative gap" is the mean gap divided by the exact posterior's own spread — Monte Carlo error as a
fraction of the uncertainty that is genuinely there. Being dimensionless, it is the only one of
these numbers that compares fairly across regimes.

---

## Next steps

1. **Longer time horizons.** Everything runs to `T = 100`. Extending to `T = 1000+` would test
   whether the flat error curves really indicate time-uniform stability. Cheap, and the natural
   immediate next step.
2. **Averaging over many datasets.** The sweeps average over 10 particle-filter seeds but hold a
   single simulated dataset. Averaging over datasets too would separate "this realisation was easy"
   from genuine properties of a regime, and would put error bars on the curves.
3. **Better proposals for large `M`.** The bootstrap filter proposes from the prior, ignoring the
   observation entirely, which is the worst case exactly where Part 2 found the damage. A guided or
   auxiliary proposal would test whether the `M`-driven collapse is intrinsic to particle filtering
   or only to this filter.
4. **Sequential Quasi-Monte Carlo (SQMC).** The `O(1/sqrt(N))` rate confirmed in every regime is the
   fundamental limit of plain Monte Carlo. SQMC replaces random draws with low-discrepancy points
   and can converge faster; in more than one dimension this needs a **Hilbert space-filling curve**
   to order the multidimensional particles. The slopes measured here are the baseline any SQMC
   result should beat.
5. **Smoothing rather than filtering.** Estimating `p(x_t | y_0:T)` instead of `p(x_t | y_0:t)`
   should help most precisely where filtering struggled — the unmeasured coordinates of the `M < d`
   case, where later observations carry information about earlier unmeasured states.

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install numpy matplotlib jupyter
.venv/bin/jupyter notebook
```

## Reference

Gerber, M. & Chopin, N. (2015). *Sequential Quasi-Monte Carlo.* Journal of the Royal Statistical
Society: Series B, 77(3), 509–579.
