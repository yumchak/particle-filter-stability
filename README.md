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
asks which dimension actually hurts. **Part 3** stress-tests Part 2's conclusions — over longer
horizons, across many datasets, and against a smarter filter.

## What's in this repo

| File | Contents |
|---|---|
| `kf_pf_lab.ipynb` | **Part 1.** The 1D Kalman-vs-particle-filter study: simulation, both filters, the PF–KF comparison, stability over time and `N`, ESS and resampling, and a nonlinear example. |
| `md_dimension_study.ipynb` | **Part 2.** The multidimensional study: the three `M` vs `d` regimes, side-by-side comparison, and dimension scaling. |
| `robustness_study.ipynb` | **Part 3.** Stress-tests Part 2's claims: long horizons (`T = 2000`), error bars over 30 datasets, and whether a better proposal removes the large-`M` collapse. |
| `filters.py` | All reusable filter code, imported by every notebook. NumPy only, no other dependencies. |
| `export_figures.py` | Unpacks every plot from the notebooks into `figures/`. |
| `figures/` | All 23 figures as standalone PNGs, named `<notebook>__fig<NN>__<section>.png`. |

All three notebooks are committed **with their outputs**, so every plot and number renders on GitHub
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

# Part 3: Robustness — long horizons, error bars, and better proposals

Part 2's headline claims each rested on **one dataset**, **100 time steps**, and **one kind of
particle filter**. Part 3 attacks all three assumptions. Two claims survive; one needed correcting.

| Part 2 claim | Verdict |
|---|---|
| Error stays flat over time | **Confirmed and strengthened.** Flat to within 3% over `T = 2000`, with drift of a fraction of a percent per 1000 steps, inconsistent in sign across regimes. |
| Regimes perform near-identically relative to posterior width | **Corrected.** Over 30 datasets the relative gaps are `0.055`, `0.059`, `0.078` and every pairwise difference is statistically clear (`\|t\|` = 2.6, 12.6, 9.7). `M < d` is genuinely ~40% worse. The ordering survives; "almost identical" does not. |
| Degeneracy is driven by `M`, not `d` | **Confirmed, and explained.** Reproducible with non-overlapping error bars — but it is a property of the *bootstrap proposal*, not of particle filtering. |

### The proposal result

The bootstrap filter moves particles using the state equation alone, consulting the observation only
afterwards. The **locally optimal proposal** (Doucet et al., 2000) samples from
`p(x_t | x_{t-1}, y_t)` instead — looking at the observation *before* moving each particle — which
is available in closed form for a linear-Gaussian model. Both target the same posterior.

| M (with d = 4) | bootstrap min ESS | optimal min ESS | bootstrap rel. gap | optimal rel. gap |
|---|---|---|---|---|
| 1 | 63.1 | 257.1 | 0.0628 | 0.0401 |
| 4 | 7.2 | 162.8 | 0.0963 | 0.0427 |
| 16 | **1.4** | **223.9** | **0.2145** | **0.0387** |

At `M = 16` the bootstrap cloud collapses to about 1.4 effective particles out of 1000; the optimal
proposal holds ~224, a **160x** difference. The accuracy trend separates too: the bootstrap relative
gap degrades by a factor of 3.4 as `M` grows, while the optimal proposal's stays flat at ~0.04.

So the honest statement of Part 2's finding is not "large `M` breaks particle filters" but
**"large `M` breaks filters that propose blindly from the prior"** — a fixable problem, not a
fundamental barrier.

---

## Next steps

Items 1-3 of the original plan (long horizons, many-dataset error bars, better proposals) are **done
in Part 3**. What genuinely remains:

1. **Sequential Quasi-Monte Carlo (SQMC).** The `O(1/sqrt(N))` rate confirmed in every regime is the
   fundamental limit of plain Monte Carlo. SQMC replaces random draws with low-discrepancy points
   and can converge faster; in more than one dimension this needs a **Hilbert space-filling curve**
   to order the multidimensional particles. The slopes measured here are the baseline any SQMC
   result should beat.
2. **Smoothing rather than filtering.** Estimating `p(x_t | y_0:T)` instead of `p(x_t | y_0:t)`
   should help most precisely where filtering struggled — the unmeasured coordinates of the `M < d`
   case, where later observations carry information about earlier unmeasured states. The Kalman
   smoother is again available as an exact benchmark.
3. **Optimal proposals beyond the linear-Gaussian case.** Part 3's rescue relies on a closed form
   that only exists because the model is linear-Gaussian. In a nonlinear model the proposal must be
   approximated, so how much of the rescue survives is an open question.
4. **Error bars on the rest of Part 2.** Part 3 re-examined one single-dataset claim and it needed
   correcting; the others deserve the same treatment.

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install numpy matplotlib jupyter
.venv/bin/jupyter notebook
```

To regenerate the standalone PNGs in `figures/` from the notebooks:

```bash
.venv/bin/python export_figures.py
```

## Reference

Gerber, M. & Chopin, N. (2015). *Sequential Quasi-Monte Carlo.* Journal of the Royal Statistical
Society: Series B, 77(3), 509–579.

Doucet, A., Godsill, S. & Andrieu, C. (2000). *On sequential Monte Carlo sampling methods for
Bayesian filtering.* Statistics and Computing, 10(3), 197–208.
