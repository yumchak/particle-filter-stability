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
horizons, across many datasets, and against a smarter filter. **Part 4** returns to one dimension to
test a conjecture from Dr Gerber: that a particle filter run with Student-t state noise should be
less stable than a Gaussian one when the observations are unbounded.

## What's in this repo

| File | Contents |
|---|---|
| `01_kf_pf_1d_comparison.ipynb` | **Part 1.** The 1D Kalman-vs-particle-filter study: simulation, both filters, the PF–KF comparison, stability over time and `N`, ESS and resampling, and a nonlinear example. |
| `02_md_dimension_study.ipynb` | **Part 2.** The multidimensional study: the three `M` vs `d` regimes, side-by-side comparison, and dimension scaling. |
| `03_robustness_study.ipynb` | **Part 3.** Stress-tests Part 2's claims: long horizons (`T = 2000`), error bars over 30 datasets, and whether a better proposal removes the large-`M` collapse. |
| `04_student_t_vs_gaussian.ipynb` | **Part 4.** The 1D particle filter run with Student-t instead of Gaussian state noise, tested in ordinary data and under very large observations. |
| `filters.py` | Core filter code for Parts 1–3: Kalman filter and bootstrap particle filter in 1D and in `d` dimensions. NumPy only. |
| `student_t_filters.py` | Part 4 additions: a particle filter with Gaussian or Student-t state noise, and a **grid filter** that computes the exact posterior for either model. Builds on `filters.py` without changing it. |
| `export_figures.py` | Unpacks every plot from the notebooks into `figures/`. |
| `figures/` | All 29 figures as standalone PNGs, named `<notebook>__fig<NN>__<section>.png`, with an index in `figures/README.md`. |

The notebooks are numbered in reading order, and all four are committed **with their outputs**, so
every plot and number renders on GitHub without running anything.

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

# Part 4: Student-t vs Gaussian state noise

Dr Gerber's question: run the 1D particle filter *as if* the state noise `v_t` were Student-t, and
compare it with the ordinary Gaussian filter. His conjecture: **if the observations are unbounded,
the Student-t filter should be less stable.** The data are generated from the Gaussian model with
`a = 1`, exactly as in Part 1. Only the filter changes, and only in one line: how particles move.

### A new benchmark was needed

The Kalman filter is *not* the exact answer for a Student-t model, so comparing the Student-t filter
to it would mix up two different things: the particles failing to represent their target
(**approximation error**, which is what "unstable" means) and the Student-t model simply having a
different posterior (**model error**, which would remain even with infinitely many particles). So
`student_t_filters.py` adds a **grid filter**: a deterministic numerical filter that computes the
exact posterior of either model on a fine grid. Each particle filter is scored against the exact
filter of **its own** model. The grid filter matches the Kalman filter to about `1e-15`, and the new
particle filter reproduces Part 1 **bit for bit** when its noise is Gaussian.

### Key findings

- **On ordinary data from the model, there is no difference.** Every filter from Gaussian to Cauchy
  has an approximation error of about `0.028`, flat in time, at the `O(1/sqrt(N))` rate. If anything
  the Student-t filters look *healthier*: minimum ESS `33` at `nu = 1` against `7.6` for the
  Gaussian.
- **That is because in-model observations never get large.** The largest `|y_t|` is about `10` over
  2000 steps and only `11.7` over 100,000 — the maximum of `T` Gaussian draws grows like
  `sqrt(2 log T)`. The regime the conjecture is about is never visited.
- **Under one large observation, the conjecture holds for moderate tails.** Inserting a single
  observation `y* = 20` and measuring the accuracy of the log-likelihood estimate (200 seeds):

  | filter | variance | bias | **RMSE** |
  |---|---|---|---|
  | Gaussian | 47.7 | −39.3 | **39.9** |
  | Student-t `nu = 5` | 527.9 | −109.0 | **111.4** |
  | Student-t `nu = 3` | 1072.7 | −66.7 | **74.3** |
  | Student-t `nu = 1` (Cauchy) | 7.6 | −1.5 | **3.1** |

- **For `nu = 5` more particles do not help.** The likelihood variance is `415`, `528`, `527`, `503`
  as `N` goes from 250 to 16,000 — 64 times the computation buys nothing. That is instability in the
  strongest practical sense.
- **It is robust across datasets.** On 12 independent datasets the Student-t filters are worse than
  the Gaussian in **12 out of 12** (`nu = 5`, median `11x`; `nu = 3`, median `21x`), and the Cauchy
  is better in **12 out of 12** (median `0.08x`).
- **But it is not monotonic in `nu` — the Cauchy is the most stable of all.** Heavier tails move the
  filter's own target further out (harder to represent) but also let particles reach further (easier
  to sample). At `y* = 20` the typical furthest of 1000 particles reaches `3.4` (Gaussian), `7.4`
  (`nu = 5`), `15.9` (`nu = 3`) and about `1000` (`nu = 1`), while the targets sit at `10.6`, `19.7`,
  `19.8` and `19.9`. At `nu = 5` the target escapes beyond reach; at `nu = 1` the particles overshoot
  it by three orders of magnitude.

**Verdict: partly supported.** The conjecture holds for moderate `nu` once observations are pushed
into the large-innovation regime, but not as a monotone "heavier tails are less stable" statement,
and it is invisible on in-model data.

### Open questions and limitations

- **The scale/shape control is incomplete.** Variance-matching the Student-t rules out "it is simply
  wider", but a Gaussian filter run at the Student-t's reduced scale is still needed to separate
  scale from tail shape properly.
- **"Unbounded" may be the wrong dial.** What drives the behaviour is the size of the *innovation*,
  not the level of the observation. A random-walk variant (`b = 1`), where observations reach the
  seventies, would test this directly; it has not been run.
- **The single-observation probe is artificial.** It measures the response to an arbitrarily large
  innovation, which in-model data will not supply. Whether this is the formulation Dr Gerber
  intended is the main question for the next meeting.
- On the filtering *mean* (rather than the likelihood), the `nu = 3` result does not survive the
  multi-dataset check (`7.41` against the Gaussian's `7.35`); only `nu = 5` (worse) and `nu = 1`
  (better) separate reliably there.

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
5. **Close the Part 4 controls.** Run the shape-fixed control (a Gaussian filter at the Student-t's
   scale) and the random-walk variant, and establish whether the worst case at intermediate `nu`
   can be located analytically. See the Part 4 section above.

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
