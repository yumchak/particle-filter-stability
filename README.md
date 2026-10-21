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
| `filters.py` | Core filter code for Parts 1–3: Kalman filter and bootstrap particle filter in 1D and in `d` dimensions. NumPy only. |
| `student_t_filters.py` | Part 4 additions: a particle filter with Gaussian or Student-t state noise, and a **grid filter** that computes the exact posterior for either model. Builds on `filters.py` without changing it. |

The notebooks are numbered in reading order, and all nine are committed **with their outputs**, so
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
   approximated, so how much of the rescue survives is an open question.
4. **Error bars on the rest of Part 2.** Part 3 re-examined one single-dataset claim and it needed
   correcting; the others deserve the same treatment.
5. **Map the detectability boundary (from Part 5).** Part 5 Section 9 shows the posterior becomes
   unbounded when an explosive direction is never observed, in a single case. Sweeping `M` from 1 to
   `d` at fixed `b > 1` would measure how much observation is enough to keep it bounded.
6. **A de-trended formulation for explosive long runs (from Part 5).** Filtering `z_t = F^-t x_t`, or
   working in log scale, would make `b = 1.1` at `T = 2000` measurable instead of losing it to
   floating-point cancellation.
7. **Does the optimal proposal help where the posterior is unbounded?** Part 3's rescue fixes the
   large-`M` collapse. Part 5 Section 9 suggests it would *not* help with unbounded posterior
   uncertainty, since that is not a proposal problem - a clean negative result if confirmed.
8. **Close the Part 4 controls.** The random-walk variant is done (Part 4b) and the worst case is
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
