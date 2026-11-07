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
horizons, across many datasets, and against a smarter filter. **Parts 1b, 2b, 3b and 4b** repeat Parts 1-4 at the stability
threshold `rho(F) = 1` to test whether the stable/unstable regime matters. **Part 4** returns to one
dimension to test a conjecture from Dr Gerber: that a particle filter run with Student-t state noise should be
less stable than a Gaussian one when the observations are unbounded. **Part 5** varies the state
transition matrix `F` across the stability threshold `rho(F) = 1`, where the hidden state stops
decaying and starts to explode.

## What's in this repo

| File | Contents |
|---|---|
| `01_kf_pf_1d_comparison.ipynb` | **Part 1.** The 1D Kalman-vs-particle-filter study: simulation, both filters, the PF–KF comparison, stability over time and `N`, ESS and resampling, and a nonlinear example. |
| `01b_kf_pf_1d_rho1.ipynb` | **Part 1b.** Part 1 re-run at `b = 1` (the threshold) instead of `0.9`. Shows the 1D fully-observed case is unaffected, and that Part 1's "the model forgets" explanation of its flat error curves was not the real mechanism. |
| `02_md_dimension_study.ipynb` | **Part 2.** The multidimensional study: the three `M` vs `d` regimes, side-by-side comparison, and dimension scaling. |
| `02b_md_dimension_rho1.ipynb` | **Part 2b.** Part 2 re-run at `rho(F) = 1` instead of `0.9`, with everything else identical. Answers whether the stable/unstable regime matters for the `M` vs `d` results. |
| `03_robustness_study.ipynb` | **Part 3.** Stress-tests Part 2's claims: long horizons (`T = 2000`), error bars over 30 datasets, and whether a better proposal removes the large-`M` collapse. |
| `03b_robustness_rho1.ipynb` | **Part 3b.** Part 3 re-run at `rho(F) = 1`. Confirms the error stays flat over `T = 2000`, puts the `M < d` penalty at 69% over 30 datasets, and shows the optimal proposal removes the degeneracy but not the threshold penalty. |
| `04_student_t_vs_gaussian.ipynb` | **Part 4.** The 1D particle filter run with Student-t instead of Gaussian state noise, tested in ordinary data and under very large observations. |
| `04b_student_t_rho1.ipynb` | **Part 4b.** Part 4 re-run at `b = 1`. The random-walk variant Part 4 said it had not run: observations do become large, and it still makes no difference -- the dial is the innovation, not the observation level. |
| `05_f_stability_threshold.ipynb` | **Part 5.** Varies the spectral radius of `F` across the stability threshold, in 1D and in `d` dimensions, with and without coupling: accuracy, cloud health, long-run behaviour, and what happens when an explosive direction is never observed. |
| `06_long_horizons.ipynb` | **Part 6.** How the error grows with `T`, up to `10^6`. Tracks the **maximum** error across decades rather than comparing halves of one run. Finds the maximum grows `7-9x` while the mean grows `1.1x` -- the statistic, not the horizon, was what limited Parts 1-3b. |
| `07_student_t_likelihood.ipynb` | **Part 7.** Student-t observation likelihood with `M > d`, under a control that holds total precision fixed. Extra coordinates cost nothing under any likelihood; what Part 2 saw was precision; and the `d`-dimensional projection stops being free, at up to `4.7x`. |
| `08_kolmogorov_distance.ipynb` | **Part 8.** Replaces the mean distance with the Kolmogorov distance -- bounded, scale-free, calibratable -- and shows Part 6's growing maximum is an extreme-value effect, not degradation: the per-step distribution is stationary over `10^6` steps. |
| `09_nonlinear_observations.ipynb` | **Part 9.** 39 nonlinear observation functions `f` x 4 potentials x 3 filter variants, scored against an exact grid posterior in one and two dimensions. The shape of `f` costs nothing; *assuming* the Gaussian reduction when it fails costs up to `16x`. |
| `filters.py` | Core filter code for Parts 1–3: Kalman filter and bootstrap particle filter in 1D and in `d` dimensions. NumPy only. |
| `student_t_filters.py` | Part 4 additions: a particle filter with Gaussian or Student-t state noise, and a **grid filter** that computes the exact posterior for either model. Builds on `filters.py` without changing it. |
| `observation_models.py` | Parts 7-9: arbitrary observation functions `f` (a 41-entry registry with **measured** rather than asserted structure), four potential shapes including the paper's own, the Section 5.3.2 projection, the affine surrogate, the information-equalising control, and exact grid benchmarks for `d = 1` and `d = 2`. |

The notebooks are numbered in reading order. All of them are committed **with their outputs**, so
every plot and number renders on GitHub without running anything. Part 5 needs no additions to
`filters.py`: the spectral radius is already a parameter of `build_model`, reachable through
`run_case(d, M, b=..., coupling=...)`.

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

# Part 1b: The 1D study at the threshold (`b = 1`)

Part 1 re-run with one number changed: `b = 1` instead of `0.9`. In one dimension `b` is the
spectral radius, so this is a random walk — no decay, no stationary distribution, nothing forgotten.

### Nothing of substance changes

| | `b = 0.9` | `b = 1` |
|---|---|---|
| KF steady-state variance | 0.5974 | 0.6180 |
| mean PF-KF gap | 0.02721 | 0.02688 |
| fitted `O(1/sqrt(N))` slope | -0.4934 | -0.4922 |
| half-run error ratios | 1.019 / 1.102 / 1.074 | 1.073 / 1.065 / 1.072 |
| mean ESS | 531.1 | 524.4 |
| **state standard deviation** | **1.83** | **2.98** |

The state's spread grew by 63% and lost its stationary distribution; the filter's uncertainty grew
by 3.4%.

### Part 1's explanation of its own result was wrong

Part 1 attributed its flat error curves to forgetting: `b < 1` means old states, and therefore old
errors, decay geometrically. At `b = 1` nothing decays and **the error is just as flat**. So
geometric forgetting of the state was not the mechanism.

What bounds the filter is that an observation arrives every step. The variance recursion
`P -> (1 - Ka)(b^2 P + Q)` has a bounded fixed point whether or not `b < 1`, because the update
always shrinks. The state may wander arbitrarily far; the filter's uncertainty about *where it is*
does not grow, because it is told, every step, to within `sqrt(R)`.

This is the fully-observed, one-dimensional case of the pattern in Parts 2b and 5: crossing the
threshold costs nothing when the observations constrain the state.

### A normalisation trap worth knowing

Two figures look like improvements and are not — the gap as a percentage of the signal
(1.49% -> 0.90%) and the nonlinear section's improvement over baseline (56% -> 72%). Both divide by
the state's own spread, which grew 63% while the numerators stayed put. **Any metric normalised by
the state is unreliable across the threshold.** The relative gap used from Part 2 onwards divides by
the *posterior* spread and is unaffected.

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
- **Weight degeneracy is driven by the sharpness of the likelihood, which here is controlled by
  `M`.** This was the clearest result. Growing `d` from `1` to `16` with `M = 1` left the cloud
  healthy throughout (min ESS never below `~90`, mean ESS pinned near `530`) and the relative gap
  crept only from `0.033` to `0.064` before flattening. Growing `M` from `1` to `16` with `d = 4`
  drove the minimum ESS from `94` down to `1.3`, forced resampling at every single step, and
  quadrupled the relative gap to `0.22`. A sharper likelihood is harder for particles proposed from
  the prior to satisfy, which is the mechanism behind both this and the `M > d` result above.

### Summary table (N = 1000, T = 100)

| Case | d | M | mean gap | max gap | min ESS | resampled | posterior spread | relative gap |
|---|---|---|---|---|---|---|---|---|
| `M = d` | 2 | 2 | 0.0371 | 0.1736 | 29.3 | 69/100 | 0.7501 | 0.0495 |
| `M > d` | 2 | 4 | 0.0297 | 0.1245 | 15.6 | 97/100 | 0.5924 | 0.0502 |
| `M < d` | 4 | 2 | 0.0628 | 0.1619 | 82.7 | 77/100 | 1.1040 | 0.0568 |

"Relative gap" is the mean gap divided by the exact posterior's own spread — Monte Carlo error as a
fraction of the uncertainty that is genuinely there. Being dimensionless, it is the only one of
these numbers that compares fairly across regimes.

> **Correction (added after Part 5).** Two earlier claims in this section have been withdrawn:
> that degeneracy is driven by `M` "not by `d`", and that *more information can make the particle
> filter worse*. Under a Gaussian likelihood the `M` observations reduce to a `d`-dimensional
> sufficient statistic (Del Moral & Gerber, 2026, Section 5.3.3), so growing `M` raises the
> observation's **precision** and never its effective dimension: the `M = 4` case above is the same
> model as `d = M = 2` with `R = 0.5*I` (each coordinate is measured twice, and averaging two
> independent measurements halves the noise variance). The experiment therefore separates
> *sharpness* from `d`, not *dimension* from `d`. The table also does not show the filter doing
> worse: from `M = d` to `M > d` the mean gap **falls** (`0.0371` -> `0.0297`) and the relative gap
> is flat; only the ESS drops, which is not the same thing.
>
> The ESS collapse itself is real and reproducible — only its attribution changes. Two consequences:
> the cleanest way to study the effect is to vary `r` directly at `M = d`, since the `M` sweep only
> reaches `r_eff = r*d/M`; and the original claim **should** hold under a **Student-t likelihood**,
> where the discarded component stays inside the weights instead of cancelling.

---

# Part 2b: The same study at the stability threshold (`rho(F) = 1`)

Part 2 re-run with **one parameter changed**: `rho(F) = 1` instead of `0.9`. Same `Q`, `R`,
`coupling = 0.3`, seeds, horizons and particle counts; Part 2 itself is untouched. Every section
carries a **Compared with Part 2** box giving the same metric at both spectral radii.

This answers Dr Gerber's question: does the stable versus unstable regime matter for the `M` vs `d`
results?

### Answer: yes, but only where the observations fail to constrain the state

| | `rho = 0.9` | `rho = 1` | |
|---|---|---|---|
| `M = d` relative gap | 0.0495 | 0.0515 | unchanged |
| `M > d` relative gap | 0.0502 | 0.0582 | unchanged |
| **`M < d` relative gap** | 0.0568 | **0.0875** | **+54%** |
| `M < d`, unmeasured/measured error | `2.6x` | **`3.45x`** | worse |
| rel gap at `d = 16`, `M = 1` | 0.064 | **0.110** | roughly doubles |
| rel gap at `d = 16`, `M = 1`, no coupling | saturating | **0.198** | worst case |
| min ESS, `M: 1 -> 16` | `94 -> 1.3` | `82 -> 1.5` | identical |
| fitted `O(1/sqrt(N))` slopes | `-0.512 / -0.520 / -0.507` | `-0.519 / -0.522 / -0.483` | survives |

Every effect that worsens involves a direction the observations do not constrain. At `rho < 1` an
unobserved coordinate still forgets its past, so its uncertainty settles at a finite stationary
value; at `rho = 1` it forgets nothing and the uncertainty grows without bound. Fully observed
(`M >= d`), crossing the threshold costs essentially nothing — the observations alone keep the
posterior bounded.

Two further notes:

- **`stationary_covariance` cannot be used here.** It solves `P = F P F' + Q`, which has no solution
  at `rho(F) >= 1`. Part 2's Section 4 used it as the "no data" reference; Part 2b replaces it with
  the finite-horizon prior variance, which is well defined at any spectral radius.
- **Flatness in time is less clear.** Half-run error ratios are `1.225`, `1.181`, `0.988`, against
  ratios within a few percent of 1 at `rho = 0.9`. One dataset and ten seeds is not enough to call
  this, and it is exactly what a longer horizon is needed to settle.

This matches the finding reached independently in Part 5: an explosive but *observed* direction
leaves the posterior bounded, while an explosive *unobserved* one does not. What bounds the filter
is detectability, not stability.

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

# Part 3b: Robustness at the threshold (`rho(F) = 1`)

Part 3 re-run with `rho(F) = 1`. Same three stress tests -- long horizons, error bars over 30
datasets, and a better proposal -- applied where the state is a random walk.

### The three Part 3 claims hold

| Part 3 claim | Verdict at the threshold |
|---|---|
| Error flat over `T = 2000` | **Confirmed.** Block ratios 1.034, 0.986, 1.031. |
| Regimes distinguishable, `M < d` worst | **Confirmed and amplified** -- the penalty grows from 40% to **69%**. |
| Degeneracy is a bootstrap-proposal property | **Confirmed**, with an addition below. |

### It settles a question Part 2b raised

2b's half-run error ratios at `T = 100` on one dataset were `1.225`, `1.181`, `0.988`, which looked
like growth. At `T = 2000` with 8 seeds and block fitting the ratios sit within 3.4% of 1.
**2b's apparent rise was noise.**

### The `M < d` penalty is real, and not a proposal problem

Over 30 datasets with non-overlapping intervals:

| relative gap | `rho = 0.9` | `rho = 1` |
|---|---|---|
| `M = d` | 0.0552 | 0.0555 |
| `M > d` | 0.0590 | 0.0594 |
| **`M < d`** | 0.0775 | **0.0937** |
| `M < d` penalty | 40% | **69%** |

The locally optimal proposal fixes the large-`M` collapse completely (min ESS at `M = 16`: 1.5 to
190, a factor of 129) but removes only part of the threshold penalty:

| `M` (`d = 4`) | regime | optimal, `rho = 0.9` | optimal, `rho = 1` |
|---|---|---|---|
| 1 | `M < d` | 0.0401 | **0.0535** (+33%) |
| 4 | `M = d` | 0.0427 | 0.0444 (+4%) |
| 16 | `M > d` | 0.0387 | 0.0397 (+3%) |

A better proposal places particles where the likelihood is; it cannot create information about
directions the data never see. At `rho = 1` those directions no longer forget, so there is more
missing information to be short of -- the same conclusion Part 5 Section 9 reaches by another route.

### One thing worth watching

The fitted drifts are all non-negative here (`+1.02`, `+2.12`, `+0.05`% per 1000 steps) where
Part 3's were mixed in sign. Small, and three regimes is a small sample, but only a much longer
horizon could dismiss it.


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
- **The worst case sits at `nu ~ 10`.** Sweeping `nu` from 1 to 500 at `y* = 20` and `y* = 50`
  gives an interior maximum at `nu ~ 10` in both cases -- about `3.2x` the Gaussian's log-likelihood
  RMSE -- after which the curve falls monotonically back to the Gaussian, as it must. So the honest
  statement is not "heavier tails are less stable" but **"tails heavy enough to move the filter's
  target, but not heavy enough for its particles to follow, are less stable"**, and that band is
  narrow and interior.
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
- **"Unbounded" was the wrong dial — now settled in Part 4b.** What drives the behaviour is the
  size of the *innovation*, not the level of the observation. The random-walk variant (`b = 1`) has
  now been run: observations reach `319` over `10^5` steps against `11.67` at `b = 0.9`, and the
  filters still show **no difference at all**. A random walk carries the filter along with it, so
  each observation is no more surprising than the last. See Part 4b.
- **The single-observation probe is artificial.** It measures the response to an arbitrarily large
  innovation, which in-model data will not supply. Whether this is the formulation Dr Gerber
  intended is the main question for the next meeting.
- On the filtering *mean* (rather than the likelihood), the `nu = 3` result does not survive the
  multi-dataset check (`7.41` against the Gaussian's `7.35`); only `nu = 5` (worse) and `nu = 1`
  (better) separate reliably there.


---

# Part 4b: Student-t at the threshold (`b = 1`) — the random-walk variant

Part 4 re-run with `b = 1`. This is the experiment Part 4's open questions said had not been run:
with a random-walk state the observations should become large on their own, so the conjecture's
condition ("if the observations are unbounded") might be met without an artificial outlier.

### The observations do become large

| `T` | `max|y|` at `b = 0.9` | at `b = 1` |
|---|---|---|
| 100 | 4.56 | 11.82 |
| 1,000 | 9.73 | 55.45 |
| 10,000 | 10.88 | 129.61 |
| 100,000 | 11.67 | **319.31** |

A factor of 27, and far past the artificial `y* = 20` and `y* = 50` probes.

### And it still makes no difference

Section 3 runs to `T = 2000`, where `max|y| ~ 55`, and finds every filter from Gaussian to Cauchy
with the same approximation error (`0.028`-`0.031`), flat in time.

**So the level of the observation is not the dial. The innovation is.** A random walk carries the
filter along with it, so however far the state travels each new observation is about as surprising
as the last -- the "in sd units" column sits at `3.6`-`4.2` at both spectral radii, while the raw
maximum grows 27-fold. Unbounded observations in the ordinary sense are not sufficient; an unbounded
*surprise* is needed, and in-model data will not supply one at any `rho(F) <= 1`.

This is a negative result and a useful one: Part 4's single-outlier probe is not just convenient, it
appears to be **necessary** to reach the regime the conjecture concerns.

### Everything else is unchanged, but noisier

| | `b = 0.9` | `b = 1` |
|---|---|---|
| approximation error, in-model | `~0.028` | `0.028`-`0.031` |
| error at `y* = 20`, Gaussian | 8.86 | 8.63 |
| error at `y* = 20`, `nu = 5` | 15.07 | 13.98 |
| `Var(log p)` vs Gaussian, `nu = 5` | `10.3x` | `9.7x` |
| worst case over `nu` | `nu = 10` | `nu = 7`-`10` |

The 12-dataset confidence intervals are roughly three times wider at `b = 1`: with a random-walk
state each dataset explores a different region, so the same experiment has substantially more
spread. Worth knowing before designing any longer run at this spectral radius.


# Part 5: Crossing the stability threshold (`rho(F)` vs 1)

Every result in Parts 1-4 was obtained at one setting of the transition matrix: `b = 0.9` in 1D, or
`F` rescaled to spectral radius `0.9` in `d` dimensions. Part 5 makes it the variable.

> **How does particle filter accuracy and stability change as the spectral radius of `F` crosses 1,
> and does coupling between state dimensions affect this behaviour?**

The threshold is governed by `rho(F)`, the largest absolute eigenvalue: below 1 the state forgets its
past and a stationary distribution exists; at 1 it is a knife-edge random walk; above 1 the state
variance is unbounded. Because `build_model` rescales by the spectral radius, `rho(F) = b` exactly,
so `b` *is* the stability knob and **no change to `filters.py` was needed**.

Four experiments, at `Q = R = I`, `N = 1000`, over `b = 0.5, 0.9, 0.99, 1.0, 1.01, 1.1`:
1D (`F = b`), diagonal (`F = b*I`, `coupling = 0`), coupled (`coupling = 0.2`), and under-observed
(`d = 2`, `M = 1`, so the explosive direction is never measured).

### Key findings

- **Crossing the threshold costs the particle filter almost nothing, when the state is fully
  observed.** From `b = 0.9` to `b = 1.1` the state grows by a factor of `1.3e7` while the exact
  posterior spread moves from `0.773` to `0.799` - a factor of `1.034`. The relative gap rises 16%
  (`0.0580` to `0.0675`) and the ESS is unchanged. Over 30 datasets the increase is ~4.5% and
  *marginally* statistically detectable (`|t| = 2.1`), then saturating: small and bounded, not absent.
- **The reason: the posterior, not the state, sets the scale of the problem.** The variance recursion
  has a bounded fixed point even when the state recursion does not. An unstable *model* is not a
  difficult *inference problem*.
- **What actually breaks is detectability, not stability.** With the explosive direction unobserved
  (`d = 2`, `M = 1`), the exact posterior variance in that direction reaches `1.25e13` at `b = 1.1`
  against `0.64` when it is observed - a factor of `1.9e13` - while the ESS stays healthy at 50-65.
  The divergence is in the **posterior itself**, not its particle approximation, so no filter can
  repair it. This refines Part 2: `M < d` is the regime where crossing the threshold turns large
  uncertainty into *unbounded* uncertainty.
- **Stability is a property of `F`'s eigenvalues, not its entries.** At `d = 4` with
  `coupling = 0.3` and `rho(F) = 1.1`, every entry of `F` is `0.7405` or smaller and the state still
  reaches `1.6e8`. You cannot read stability off the diagonal.
- **Coupling changes how many directions cross the threshold, not what crossing costs.** At
  `b = 1.1`, `d = 2`: diagonal `F` has 2 of 2 directions explosive, coupled `F` has 1 of 2. The
  relative-gap penalty is `1.163` versus `1.106`.
- **The error is flat in time at and just above the threshold.** Over `T = 2000` at
  `b = 0.9, 1.0, 1.01` the half-run ratios are `0.988`, `0.960`, `0.987`, with drift negative in all
  three. Stated carefully: the error is **bounded in mean over the horizons tested** - a finite-`T`,
  in-mean statement, weaker than time-uniform stability in the almost-sure sense.
- **Degeneracy and accumulation are independent failure modes.** At `d = 4`, `M = 16` the minimum ESS
  is `1.0` and resampling fires at all 2000 steps - the posterior carried by a single particle - and
  the half-run ratio is still `1.075`. A cloud running on one effective particle is inaccurate but
  not diverging. *(Closes a gap in Part 3, which ran long horizons only at small `(d, M)`.)*

### Two numerical traps, documented rather than avoided

- **`stationary_covariance` fails silently above the threshold.** It solves `P = F P F' + Q`, which
  has no solution when `rho(F) >= 1`. At `b = 1.0` it raises `LinAlgError`; at `b = 1.01` and `1.1`
  it returns a **negative variance** and raises nothing. Do not call it for `b >= 1`.
- **Apparent error growth at long horizons is floating-point cancellation, not instability.** At
  `b = 1.1`, `T = 2000` the state reaches `1e83` and the measured gap is `9.9e66` - matching the
  cancellation floor `|x|*eps = 4.1e67` almost exactly. Run naively, the experiment reports error
  growing by sixty-six orders of magnitude: a convincing false positive. The usable horizon obeys
  `b^T <~ 1e10`, so long runs near `b = 1` and short runs further out; `b = 1.1` is safe only to
  `T ~ 241`. Section 8 of the notebook measures this instead of hiding it.

### Limitations

Only `Q = R = I` was tested, so the insensitivity above depends on the observations being
informative at every step. Section 9 used a single unobserved direction at `d = 2`. Section 7b used
only 4 seeds, and its `d = 16` ratio of `1.104` is within noise of the others. And `b` and `T`
cannot both be pushed, so nothing here says what happens at `b = 1.1` beyond `T ~ 241`.

---

---

# Part 6: Long horizons — how the error grows with `T`

Dr Gerber asked for `T ~ 10^6` to be confident the error does not grow when `M < d`. The runs go to
`10^6`, but the main finding is about the **measurement**, not the horizon.

### The statistic was the binding constraint

Over `T: 1e2 -> 1e6`:

| | growth |
|---|---|
| **maximum** gap | **7.1x - 9.4x** |
| **mean** gap | 1.06x - 1.16x |

From `T = 1e4` onwards the mean is flat to three decimal places (`M = d` at `rho = 0.9` reads
`0.0404`, `0.0400`, `0.0399` at `1e4`, `1e5`, `1e6`). Parts 1, 3 and 3b reported the mean and
compared halves of a single run -- a design whose expected signal is `1.05` at `T = 2000` and
*smaller* at `1e6`, against measured noise of about +/- 3%. **Those notebooks were not limited by
running too short; they were measuring a quantity that does not move.**

### The growth is logarithmic, not polynomial

`(log T)^a` fits better than `T^b` in all six configurations (`R2 ~ 0.98-0.99` against `0.92-0.96`),
so the worst-case error is not growing like a power of the horizon. The fitted `a` is `1.74-2.03`
rather than `0.5`.

**This is not a contradiction of the theory.** Section 7.5.1 of Del Moral & Gerber (2026) bounds the
**Kolmogorov distance** between distributions, which is at most 1 by construction. What is measured
here is the distance between posterior **means**, which is unbounded -- a collapsed particle cloud
has small Kolmogorov distance however wrong its mean is. Testing the theorem itself needs the
Kolmogorov distance against the exact Gaussian posterior, which is the natural next step.

### The threshold matters only where the observations fall short

At `T = 1e6`, `rho = 1` against `rho = 0.9`:

| regime | max gap | mean gap |
|---|---|---|
| `M = d` | 1.341 -> 1.289 | +4% |
| `M > d` | 1.260 -> 1.434 | +3% |
| **`M < d`** | **1.739 -> 2.700 (+55%)** | **+37%** |

The same pattern as Parts 1b, 2b, 3b and 5, now at a million steps.

### More particles buy less in the worst case

The maximum error decays like `N^-0.22` to `N^-0.36`, against the mean's `N^-0.5`, and the exponent
weakens as `T` grows. At long horizons every configuration eventually hits a step with ESS near 1
whatever `N` is; more particles delay that step without preventing it, and the error there is set by
how far the state has travelled. A sixteenfold increase in `N` cut the mean error fourfold in
Part 1; here it cuts the worst case by less than a factor of two.

### Limitations

Two seeds at `T = 1e6` and four at `1e5`, so the longest-horizon points are the noisiest and carry
the most weight in the fitted exponent. One dataset per setting. And minimum ESS is itself an
extreme-value statistic, so its fall with `T` is partly definitional.

# Part 7: Student-t likelihood with `M > d`

Part 2 claimed that more observations make the particle filter worse. Dr Gerber pointed out that
under a Gaussian likelihood this model reduces the `M`-dimensional observation to a `d`-dimensional
one (his Section 5.3.3), so the experiment could not have been testing that; the claim was
withdrawn. This notebook builds the setting where it **is** testable -- a Student-t likelihood,
where the reduction fails -- and tests it. It also answers point 5 of his email: replacing `y` by
its `d`-dimensional projection.

Uses `observation_models.py`. Since the model is no longer linear-Gaussian, the Kalman filter is not
its exact filter; the benchmark is a **grid filter**, checked against Kalman to `1e-13` on the
Gaussian cases. The metric is the Kolmogorov distance of Part 8 -- bounded, scale-free, and
calibrated so that a score `D` means about `(1.017/D)^2` independent draws.

### Making "more observations" a fair test

With `f(x) = Ax`, `A = (1,...,1)'` and `Sigma = r I_M`, the precision is `A' Sigma^-1 A = M/r`, so
Part 2's convention **raises precision** as `M` grows. Setting `Sigma = M r0 I_M` instead holds it
at `1/r0`. The exact posterior spread is then identical at every `M` -- `0.78593` at
`M = 1, 2, 4, 8, 16`, to five decimals -- so the same problem is simply spread over more
coordinates, and dimension is isolated.

### Extra coordinates are not harmful, even under a Student-t

At fixed total precision, mean Kolmogorov distance from `M = 1` to `M = 16`:

| potential | mean KS | effective `N` |
|---|---|---|
| Gaussian | `0.0459 -> 0.0447` | `491 -> 519` |
| Student-t `nu = 5` | `0.0444 -> 0.0433` | `525 -> 553` |
| Student-t `nu = 1` | `0.0436 -> 0.0433` | `546 -> 554` |

Flat everywhere. At `nu = 1` the cloud is actually healthier at large `M` (mean ESS `531 -> 576`,
resampling rate `0.47 -> 0.40`). **The natural reading of the withdrawn sentence fails even where it
can be tested.**

### What Part 2 saw was precision, exactly as the correction said

Under Part 2's own convention the effective sample size roughly halves -- but by the same factor for
every likelihood:

| potential | KS ratio `M: 1 -> 16` | effective `N` |
|---|---|---|
| Gaussian | 1.50 | `491 -> 219` |
| Student-t `nu = 1` | 1.46 | `546 -> 256` |
| Student-t `nu = 5` | 1.53 | `525 -> 226` |
| Student-t `nu = 30` | 1.54 | `511 -> 216` |

A mechanism that behaves identically from Cauchy to near-Gaussian is not a tail mechanism, and it
vanishes once precision is held fixed. This does **not** reinstate the withdrawn claim; it sharpens
what Part 2 saw. Part 2 reported the relative gap and found it flat; on a distributional metric the
cloud's worth does fall, `491 -> 219` draws. A sharper likelihood buys a better answer (exact RMSE
`0.772 -> 0.239`) with a worse cloud.

### The real consequence: the projection stops being free

Cost of replacing `y` by its `d`-dimensional projection, as a ratio of mean Kolmogorov distances:

| `nu` | `M=2` | `M=4` | `M=8` | `M=16` |
|---|---|---|---|---|
| 1 | 3.15 | 4.17 | 4.32 | **4.73** |
| 2 | 2.40 | 3.15 | 3.49 | 4.02 |
| 5 | 1.56 | 2.04 | 2.38 | 2.70 |
| 10 | 1.20 | 1.44 | 1.65 | 1.91 |
| 30 | 1.02 | 1.08 | 1.16 | 1.25 |

Monotone in both directions across all twenty cells, and it converges to exactly the right place:
at `M = 8` the cost is `2.379`, `1.162`, `1.020`, `1.008`, `1.000` for `nu = 5, 30, 100, 1000` and
Gaussian. Run on identical random numbers, the Gaussian full and projected filters agree to
`2e-15` -- they are the same filter -- while at `nu = 5` they differ by `0.57` to `1.64`.

**That is the testable content of the original intuition, and it holds.** Not "extra coordinates
hurt", but "under a Student-t the extra coordinates carry real information, so assuming the
Gaussian reduction costs up to a factor of 4.7".

### Why: `w_t` says how far to trust `zhat_t`

At `M = 8`, `nu = 2`, split by the size of the discarded residual:

| steps | full KS | projected KS | excess |
|---|---|---|---|
| quiet (`||w||^2` below median) | 0.0486 | 0.1397 | 0.0911 |
| wild (top 10%) | 0.0349 | 0.2250 | **0.1901** |

On the wild steps the **full** filter beats its own average while the projected filter collapses. A
large `||w||^2` means the `M` coordinates disagree, which under a shared-scale Student-t is the
signature of an outlier. The full filter sees it and discounts the observation; the projected filter
has thrown the evidence away. So `w_t` carries no information about `x_t` -- that is geometry -- but
it carries information about **how much to believe** the part that does.

### Limitations

`d = 1` throughout, because the exact benchmark is a one-dimensional grid filter. `f(x) = [x,...,x]`
only, a deliberately linear image, so that the potential shape is the only thing that can break the
reduction -- curved images break it independently, which is Part 9. The joint Student-t shares one
scale across coordinates, which is what makes `||w||^2` informative; the independent-per-coordinate
version, where the mechanism should be weaker, is available in the module but not swept here. And
`rho(F) = 1` (random walk) throughout, matching the model from the meeting.

# Part 8: The Kolmogorov distance — measuring what the theorem bounds

Part 6 closed by saying the theorem and the experiment used different metrics. This notebook
measures the one the theorem uses, on the same grid, so the comparison is like for like.

The distance is `sup_z |F_cloud(z) - F_exact(z)|` along a direction, maximised over directions: a
difference of two probabilities, so it can never exceed 1. `filters.py` gains `normal_cdf`,
`unit_directions` and `weighted_ks_vs_gaussian`, plus a `stat_fn` hook on `particle_filter_md` that
computes a per-step diagnostic and discards the cloud. That hook is what makes `T = 1e6` possible:
storing the cloud would need about 16 GB, the statistic is 8 MB. With `stat_fn=None` the filter is
bit-identical to the one used in Parts 1-6.

### The metric was checked before it was used

- `normal_cdf` against `math.erf`: max error `7.45e-8` (no scipy in this repo).
- An exact sample of size `N` from the target: fitted slope `-0.5004`, with `D sqrt(N)` between
  `0.985` and `1.035`. This also **calibrates** the metric -- `D ~ 1.017 / sqrt(N)` for an exact
  sample.
- A deliberately shifted target: measured matches the analytic supremum to within Monte Carlo noise
  at all four shifts.
- A single particle carrying all the weight scores exactly `0.5` at the posterior mean. **Any step
  whose ESS collapses to ~1 scores at least 0.5**, which drives everything below.

### The per-step error distribution is stationary

Across ten blocks of a `1e6`-step run at `rho = 1` with `M < d`, last block over first:

| mean | median | q90 | q99 | q99.9 | rate > 0.25 |
|---|---|---|---|---|---|
| 1.002 | 1.000 | 1.003 | 1.002 | 0.995 | 1.001 |

Only the two rarest statistics move at all. The maximum at any horizon is then **predicted from the
`(1 - 1/T)` quantile of that distribution, with nothing fitted**, to within 15% over four decades
(`obs/pred` = 0.947, 1.149, 1.052, 0.951, 1.035). **The growth of the worst case is an
extreme-value effect, not degradation** -- longer runs do not make the filter worse, they give more
chances to observe the same rare bad step. This is the direct answer to Dr Gerber's question about
`M < d` at long horizons.

### Which statistic you report decides whether the theory appears to hold

Growth over `T: 1e2 -> 1e6`, against `sqrt(log T)` = 1.732 on this range:

| statistic | growth | fitted `(log T)^a` |
|---|---|---|
| mean | 1.00 - 1.07 | `a = -0.001 - 0.054` (no trend) |
| **q99.9** | **1.46 - 2.31** | **`a = 0.346 - 0.714`** |
| maximum | 2.95 - 4.00 | `a = 1.008 - 1.261` |
| mean gap (Parts 1-6) | 1.03 - 1.11 | — |
| max gap (Parts 1-6) | 5.71 - 8.69 | Part 6 reported 7.1 - 9.4 |

The `q99.9` row brackets both `sqrt(log T)` and the `a = 0.5` of Section 7.5.1. Part 6's `a = 1.74 -
2.03` on the maximum falls to `1.008 - 1.261` here, because at `T = 1e6` the bounded metric has
reached `0.92 - 1.00` and has nowhere left to go.

### The `sqrt(N)` collapse is recovered

The mean distance decays as `N^-0.479` to `N^-0.482` against a theoretical `-0.5`, where Part 6
measured `-0.22` to `-0.36` on the unbounded metric. Rescaled, `D sqrt(N)` is also **flat in `T`**
(2.46, 2.49, 2.49 over two decades, where `sqrt(log T)` would give 1.29x), so for the mean the
`N >= C log T` requirement is conservative. The maximum still does not collapse (`N^-0.114` to
`N^-0.434`), exactly as the extreme-value reading predicts.

### A thousand particles are worth 128-236 independent draws

Using the Section 1 calibration, at `N = 1000` and `T = 1e6`:

| regime | effective `N` | % of nominal |
|---|---|---|
| `M = d`, `rho = 0.9` | 236 | 23.6% |
| `M > d`, `rho = 0.9` | 174 | 17.4% |
| `M < d`, `rho = 0.9` | 156 | 15.6% |
| `M < d`, `rho = 1.0` | 128 | 12.8% |

No quantity in Parts 1-6 can be calibrated this way, because none of them is a probability. The
metric also tracks cloud health far better: against `log(ESS)` it correlates at `-0.654` where the
mean distance manages `-0.300`.

### Limitations

The distance is taken over the `d` coordinate axes. Adding 64 random directions raises it by a
near-constant `1.21x` across all six configurations, so growth rates are unaffected -- but this is
the **marginal** Kolmogorov distance, not the `d`-dimensional orthant version the theory uses.
`T = 1e6` used a single seed: the mean and quantiles rest on `1e6` measurements and are precise,
while the maximum there is one draw of an extreme. And Section 7.5.2 is a statement about what `N`
guarantees a fixed precision; Section 6 tests a necessary consequence of it at fixed `N`, not the
bound itself.

---

# Part 9: Nonlinear observation functions

Every part up to here observes the state through a **linear** map. Part 9 replaces it with an
arbitrary one -- the model from the supervision whiteboard,

```
x_{t+1} = b x_t + U_t,     y_t = f(x_t) + eps_t
```

-- and runs **39 distinct `f`** through the filter: `[x + cos x, x]`, `|x|`, `log(1 + x^2)`, `e^x`,
`[cos x, sin x]`, `round(x)`, `clip(x, +-3)`, a product lift, polar coordinates, range-only and
bearings-only tracking. Each is crossed with four potentials (Gaussian, joint Student-t at
`nu = 5` and `nu = 1`, coordinate-wise Student-t) and three filter variants, in one and two state
dimensions.

Once `f` is nonlinear the model is not linear-Gaussian, so **the Kalman filter is not its exact
filter and cannot be the benchmark.** The benchmark is a grid filter carrying the exact recursion
out numerically: `grid_filter_obs` for `d = 1`, and `grid_filter_obs_2d`, added here, for `d = 2`.
Both are validated against the Kalman filter where both apply -- `4e-15` at `b = 1`, where the
predict step is a pure convolution.

## The distinction the paper actually turns on

Del Moral & Gerber (2026) Section 5.3.1 needs the **image** `f(R^d)` to lie in a `d`-dimensional
affine subspace. That is not the same as `f` being linear, and the registry is built so the two can
be measured apart:

| `f` | bi-Lipschitz? | image | reduction |
|---|---|---|---|
| `[x^3, x^3]` | no -- constants span `9.8e-10` to `1.5e2` | a **line** | **holds** |
| `[x + cos x, x]` | yes -- constants `1.00` and `2.24` | a **curve** | **fails** |

The violently nonlinear one reduces; the tame one does not. The registry's own hand-set flags were
wrong in **7 of 41 entries**, all `d = 2`; the measurement overruled them. The census also found two
pairs of entries to be the same map under two names (a norm is a range; polar coordinates are
range-and-bearing), so 41 entries are 39 functions.

## Holding precision fixed, again

A nonlinear `f` sets its own observation precision pointwise: at `Sigma = I` the cubic `[x^3, x^3]`
is `18 x^4` times as informative as `[x, x]`. At `b = 1` the state reaches `|x| = 44`, and there the
exact posterior has standard deviation **`2e-18`** -- narrower than any affordable grid cell, so
both the filter and the benchmark are meaningless. Comparing `f` shapes at a common `Sigma` would
have been a study of `||J||^2` in disguise, exactly the trap Part 7 identified in Part 2.

`information_scale` fixes the **median** Fisher information over the states actually visited, leaving
`Sigma = I` for the replication baselines so Part 7's numbers stay comparable. It cannot fix the
range, and the spread is itself a result: `1x` for `[x, x]` and `|x|`, `4.8x` for `[x + cos x, x]`,
`1.9e4x` for `[x^3, x^3]`, `1.2e5x` for `e^x`, **infinite** for `round(x)` and `clip(x)`. "Fixed
observation precision" is not a property a nonlinear `f` has.

## The main result is negative

**With precision held fixed, the shape of `f` costs the correct filter essentially nothing.** Over
122 cells the mean Kolmogorov distance is `0.0474`, spread `0.0390` to `0.0950` -- a factor of
`2.4` between the best and worst cell in the matrix:

| group | n | Gaussian | t joint `nu=5` | t joint `nu=1` | t indep `nu=5` |
|---|---|---|---|---|---|
| A: linear image | 7 | `0.0489 +- 0.0053` | `0.0482 +- 0.0058` | `0.0461 +- 0.0055` | `0.0476 +- 0.0074` |
| B1: curved, bi-Lipschitz | 10 | `0.0464 +- 0.0032` | `0.0466 +- 0.0017` | `0.0445 +- 0.0014` | `0.0456 +- 0.0018` |
| B2: curved, lower bound only | 4 | `0.0541 +- 0.0067` | `0.0538 +- 0.0064` | `0.0500 +- 0.0058` | `0.0536 +- 0.0054` |
| C: (P1)b violated | 10 | `0.0514 +- 0.0161` | `0.0461 +- 0.0058` | `0.0444 +- 0.0070` | `0.0461 +- 0.0058` |

The groups are as structurally different as the paper's conditions allow, and they **agree to within
their own scatter**. `clip(x, +-3)`, which destroys information outright, is the *best* cell in the
matrix. `round(x)`, discontinuous with zero derivative almost everywhere, scores `0.0418`. Scaling
in `N` confirms this is sampling error and not bias: slopes `-0.478` to `-0.509` against the
theoretical `-0.5`, across a discontinuous `f`, a censored one, a curved image and a bimodal
posterior.

Taken with Part 7, the arc that began with Part 2's withdrawn claim closes: **the bootstrap filter
is sensitive to how sharp the likelihood is relative to the proposal, and close to indifferent to
the geometry of the map producing it.**

## Four things that do cost something

1. **Assuming Section 5.3.1 when it fails: `1.2x` to `15.9x`.** The reduction is not available
   approximately. The `affine` variant replaces `f` by its best affine fit -- whose image *is* a
   line, so the projection always has something to discard -- and scores it against the exact
   posterior of the true model. Worst: `e^x` paired with `x` at `15.9x`, `tanh x` at `14.0x`, the
   circle at `13.7x`. The cost is **geometric, not distributional**: reading across the four
   potentials changes it by at most ~20%.
2. **The projection, under a heavy-tailed potential: `1.30x` to `3.22x`**, reproducing Part 7 on new
   functions. Exactly `1.000` under a Gaussian -- to `2.8e-17`, round-off on identical random
   numbers -- and decaying to 1 by `nu = 30` with nothing fitted (`3.22, 2.74, 1.79, 1.21, 1.00` for
   `nu = 1, 2, 5, 10, 30`).
3. **Non-injectivity, but only once the state wanders.** Harmless at `b = 0.9`; up to **`6.5x`** at
   `b = 1`. For an even `f` the posterior has modes at `+-x_t`, so their separation is `2|x_t|` and
   grows with the state: at `|x| = 44` the modes are 88 apart while the proposal moves particles
   `O(1)` per step. This is the only genuine interaction between `f` and the dynamics found here.
4. **Under-observation along a curve.** Observing only `||x||` (`0.270`) or `x_1 x_2` (`0.235`) costs
   twice what observing only `x_1` (`0.121`) or `x_1 + x_2` (`0.118`) costs. Being under-observed is
   cheap; being under-observed along a ring or a hyperbola is not.

## The projection needs equation (9), not merely a non-exponential zeta

Section 5.3.2 works in an orthonormal basis of `V`, so it **rotates** the whitened residual. It is
therefore a reparameterisation only for a potential that is a function of `||u||^2` alone. The
Gaussian and the joint Student-t are (invariance to `< 2e-15`); the **coordinate-wise** Student-t is
not (`1.237`), and for it the projection changes the filter by `3.5e-2` **even on the 24 cells where
it discards nothing at all**. Per-coordinate Student-t noise is the more natural of the two models,
so this is the practically relevant case.

## A limitation of Parts 1-6, now quantified

Parts 1-6 all measured accuracy as a distance between posterior means. For a non-injective `f` that
is not merely noisy but meaningless:

| `f` | mean abs. posterior mean | mean abs. true state | RMSE of the **exact** mean | Kolmogorov | bimodal |
|---|---|---|---|---|---|
| `abs(x)` | `0.000` | `2.026` | `2.576` | `0.0573` | 87% of steps |
| `x^2` | `0.000` | `2.026` | `2.576` | `0.0950` | 69% |
| `cos x` | `0.000` | `2.026` | `2.576` | `0.0481` | 81% |
| `log(1 + x^2)` | `0.000` | `2.026` | `2.576` | `0.0461` | 82% |
| `x + cos x` | `1.893` | `2.026` | `0.841` | `0.0428` | 1% |

The observation cannot tell `x` from `-x`, so the exact posterior is symmetric and its mean is pinned
near zero wherever the state is. A mean-based metric would have reported these four as catastrophic
failures while the filters were performing normally for this matrix. The Kolmogorov distance was
adopted in Part 8 for its scale-freeness; **its more consequential property turns out to be being
defined at all when the posterior is multimodal.**

## What was not settled

`x^2` is the one cell where more particles stop helping: slope `-0.342`, and `0.0370 -> 0.0355` from
`N = 4000` to `N = 16000` where every other function halves. `|x|` is equally bimodal and scales at
`-0.495`, so bimodality alone is not the cause -- `sq_only` is bimodal **and** the most informative
group-C function, and it is that pairing that flattens the curve. It is the same mechanism Part 9
isolates at `b = 1`, reached by raising precision instead of letting the state wander.

`exp_only` and `softsign_only` remain unusable at `b = 1`: their information varies by `1e41` and
`3e3` along a single trajectory, which fixing the median cannot address.

## Next steps

Items 1-3 of the original plan (long horizons, many-dataset error bars, better proposals) are **done
in Part 3**; varying `F` across the stability threshold is **done in Part 5**. What genuinely
remains:

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
   approximated, so how much of the rescue survives is an open question. Part 9 sharpens the target:
   the only failure mode it found twice over is a sharp likelihood that must be satisfied at two
   well-separated modes, which is a **proposal** problem rather than a model one. `sq_only` at
   `b = 0.9` and `|x|` at `b = 1` are the two cells to run.
4. **Locate where the `log T` dependence enters (from Part 8).** Part 8 settles the metric
   question -- the mean distance carries no horizon dependence at all, while the `99.9`th
   percentile grows like `(log T)^0.35-0.71`. Which quantile first picks up a `log T` term would
   sharpen the reading of Section 7.5.2, and costs nothing beyond rerunning Part 8's Section 6 on a
   finer quantile grid. The `d`-dimensional orthant Kolmogorov distance, rather than Part 8's
   marginal version, is the other loose end.
5. **Error bars on the rest of Part 2.** Part 3 re-examined one single-dataset claim and it needed
   correcting; the others deserve the same treatment. Parts 7 and 9 add a reason to: both found the
   effects that survive a fixed-precision control to be small enough that single-dataset scatter
   could hide or invent one.
6. **A posterior-weighted curvature measure (from Part 9).** The relative residual of the affine fit
   predicts the cost of forcing the Gaussian reduction at a correlation of only `0.596`. Weighting
   the misfit by the posterior rather than by the sampling window would plausibly do better, and
   would turn Part 9 Section 6 from a ranking into a prediction.
7. **A state-dependent `Sigma`, or a compact state space (from Part 9).** `e^x` and `x/(1 + |x|)`
   cannot be run at `rho(F) = 1`: their Fisher information varies by `1e41` and `1e3` along one
   trajectory, so equalising its median is not enough. Either control would close the two cells
   Part 9 had to drop.
8. **Map the detectability boundary (from Part 5).** Part 5 Section 9 shows the posterior becomes
   unbounded when an explosive direction is never observed, in a single case. Sweeping `M` from 1 to
   `d` at fixed `b > 1` would measure how much observation is enough to keep it bounded.
9. **A de-trended formulation for explosive long runs (from Part 5).** Filtering `z_t = F^-t x_t`, or
   working in log scale, would make `b = 1.1` at `T = 2000` measurable instead of losing it to
   floating-point cancellation.
10. **Does the optimal proposal help where the posterior is unbounded?** Part 3's rescue fixes the
   large-`M` collapse. Part 5 Section 9 suggests it would *not* help with unbounded posterior
   uncertainty, since that is not a proposal problem - a clean negative result if confirmed.
11. **Close the Part 4 controls.** The random-walk variant is done (Part 4b) and the worst case is
   located empirically at `nu ~ 10` (Part 4 Section 8). What remains is the shape-fixed control --
   a Gaussian filter run at the Student-t's reduced scale -- and whether the `nu ~ 10` location can
   be derived rather than measured.

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install numpy matplotlib jupyter
.venv/bin/jupyter notebook
```

## Reference

Gerber, M. & Chopin, N. (2015). *Sequential Quasi-Monte Carlo.* Journal of the Royal Statistical
Society: Series B, 77(3), 509–579.

Doucet, A., Godsill, S. & Andrieu, C. (2000). *On sequential Monte Carlo sampling methods for
Bayesian filtering.* Statistics and Computing, 10(3), 197–208.

Del Moral, P. & Gerber, M. (2026). *A perturbation theory for Feynman-Kac models and their particle
interpretations.* arXiv:2610.21587. — Section 5.3.3 is the basis of the Part 2 correction above.
