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
| `01b_kf_pf_1d_rho1.ipynb`<br>*baseline:* `01_kf_pf_1d_comparison.ipynb` | **Part 1.** The 1D Kalman-vs-particle-filter study: simulation, both filters, the PF-KF comparison, stability over time and `N`, ESS and resampling, and a nonlinear example. **Current version is `01b`, at the threshold `b = 1`**; it shows the 1D fully-observed case is unaffected, and that Part 1's "the model forgets" explanation of its flat error curves was not the real mechanism. The `01` baseline is the same study at `b = 0.9` and is still referenced by Parts 6, 8 and 11. |
| `02b_md_dimension_rho1.ipynb`<br>*baseline:* `02_md_dimension_study.ipynb` | **Part 2.** The multidimensional study: the three `M` vs `d` regimes, side-by-side comparison, and dimension scaling. **Current version is `02b`, at `rho(F) = 1`**, everything else identical, answering whether the stable/unstable regime matters for the `M` vs `d` results. The `02` baseline is at `rho(F) = 0.9`; both carry correction boxes -- four of their claims have been withdrawn or corrected by Parts 3, 5, 7, 11 and 12. |
| `03b_robustness_rho1.ipynb`<br>*baseline:* `03_robustness_study.ipynb` | **Part 3.** Stress-tests Part 2's claims: long horizons (`T = 2000`), error bars over 30 datasets, and whether a better proposal removes the large-`M` collapse. **Current version is `03b`, at `rho(F) = 1`**: the error stays flat over `T = 2000`, the `M < d` penalty is 69% over 30 datasets, and the optimal proposal removes the degeneracy but not the threshold penalty. The `03` baseline is at `rho(F) = 0.9` and is the source of the 30-dataset protocol Parts 11 and 12 reuse. |
| `04b_student_t_rho1.ipynb`<br>*baseline:* `04_student_t_vs_gaussian.ipynb` | **Part 4.** The 1D particle filter run with Student-t instead of Gaussian state noise, in ordinary data and under very large observations. **Current version is `04b`, at `b = 1`** -- the random-walk variant Part 4 said it had not run: observations do become large, and it still makes no difference, the dial being the innovation rather than the observation level. The `04` baseline is at `b = 0.9` and holds the `nu` sweep of its Section 8; its in-model ESS claim was withdrawn by Part 12. |
| `05_f_stability_threshold.ipynb` | **Part 5.** Varies the spectral radius of `F` across the stability threshold, in 1D and in `d` dimensions, with and without coupling: accuracy, cloud health, long-run behaviour, and what happens when an explosive direction is never observed. |
| `06_long_horizons.ipynb` | **Part 6.** How the error grows with `T`, up to `10^6`. Tracks the **maximum** error across decades rather than comparing halves of one run. Finds the maximum grows `7-9x` while the mean grows `1.1x` -- the statistic, not the horizon, was what limited Parts 1-3b. |
| `07_student_t_likelihood.ipynb` | **Part 7.** Student-t observation likelihood with `M > d`, under a control that holds total precision fixed. Extra coordinates cost nothing under any likelihood; what Part 2 saw was precision; and the `d`-dimensional projection stops being free, at up to `4.7x`. |
| `08_kolmogorov_distance.ipynb` | **Part 8.** Replaces the mean distance with the Kolmogorov distance -- bounded, scale-free, calibratable -- and shows Part 6's growing maximum is an extreme-value effect, not degradation: the per-step distribution is stationary over `10^6` steps. |
| `09_nonlinear_observations.ipynb` | **Part 9.** 39 nonlinear observation functions `f` x 4 potentials x 3 filter variants, scored against an exact grid posterior in one and two dimensions. The shape of `f` costs nothing; *assuming* the Gaussian reduction when it fails costs up to `16x`. |
| `10_detectability_boundary.ipynb` | **Part 10.** Sweeps `M` from 1 to `d` above the stability threshold, as Part 5 asked. Finds no boundary in `M` -- `M = d-1` is as bad as `M = 1` -- and that detectability, an exact test on `(F, H)`, decides it in `84/84` cells. Then the first failure of time-uniform stability in the project. |
| `11_part2_error_bars.ipynb` | **Part 11.** The six Part 2 claims Part 3 did not re-test, over 30 datasets instead of one. Two fall: the `O(1/sqrt N)` rate does **not** hold at `M < d`, and the `M < d` cloud is not healthier. Both failures trace to the same cause, and the rate deficit sits entirely in the unobserved directions. |
| `12_screen_2b_4_5.ipynb` | **Part 12.** The same screen on Parts 2b, 4 and 5: ten claims classified and **predicted before being run**, then tested over 30 datasets. Two fall (Part 2b's rate, Part 4's in-model ESS claim), Part 5 comes out clean, and the rule scores `7/10` -- the three misses correcting it. |
| `13_proposals_nonlinear.ipynb` | **Part 13.** How much of Part 3's optimal-proposal rescue survives once the model is not linear-Gaussian and the proposal must be approximated. Four proposals, including an exact one computed numerically as a ceiling. The rescue is real; the approximations give up the `N^-1/2` rate because their tails are lighter than the target's, and `t_5` components restore it for free. |
| `filters.py` | Core filter code for Parts 1–3, 5, 6 and 10: Kalman filter and bootstrap particle filter in 1D and in `d` dimensions, the Kolmogorov distance against a Gaussian posterior (Part 8), and Part 10's detectability tools -- the observability matrix and its subspaces, the Hautus test, and the data-free covariance recursion. NumPy only. |
| `student_t_filters.py` | Part 4 additions: a particle filter with Gaussian or Student-t state noise, and a **grid filter** that computes the exact posterior for either model. Builds on `filters.py` without changing it. |
| `observation_models.py` | Parts 7-9: arbitrary observation functions `f` (a 41-entry registry with **measured** rather than asserted structure), four potential shapes including the paper's own, the Section 5.3.2 projection, the affine surrogate, the information-equalising control, and exact grid benchmarks for `d = 1` and `d = 2`. Part 13 adds four proposals for a general scalar `f` -- bootstrap, an exact locally optimal one by numerical CDF inversion, and Gaussian or Student-t mixtures from linearising at the prior mean or at every likelihood mode. |

The notebooks are numbered in reading order. Parts 1-4 each exist in two versions: the plain number
is the original run below the stability threshold (`rho(F) = 0.9`) and the `b` suffix is the same
study at the threshold (`rho(F) = 1`), which is the current version and the one listed above. The two
are companions rather than replacements -- the sub-threshold runs are the baseline that Parts 6, 8,
11 and 12 compare against, so both are kept. From Part 5 onwards there is one notebook per part.

All notebooks are committed **with their outputs**, so every plot and number renders on GitHub
without running anything. Part 5 needs no additions to
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

> **Correction (added after Part 11).** Every claim and every table entry above came from **one
> dataset**, `DATA_SEED = 42`, averaged over ten particle-filter seeds — which averages away the
> filter's Monte Carlo noise and leaves the dataset's contribution untouched. Part 3 re-tested three
> claims over 30 datasets and corrected one. Part 11 tested the remaining six and **two more do not
> survive**:
>
> - **"The `O(1/sqrt(N))` rate survives in every regime [...] being under-observed costs a constant
>   factor, not a worse convergence rate" — withdrawn.** Over 30 datasets `M = d` and `M > d` sit on
>   `-0.5` (`-0.5027 +- 0.0057`, `-0.5034 +- 0.0090`) but `M < d` gives `-0.4745 +- 0.0108`, which is
>   `4.6` standard errors from `-0.5` and `4.5` from `M = d`. The penalty is correspondingly not
>   constant: it rises `1.93 -> 2.23` over `N = 50` to `5000` (paired `t = +4.4`). The practical
>   reading survives — the rate is worse by 5%, so the penalty reaches `3x` only near `N = 1e7` —
>   but the stated claim does not.
> - **"The `M < d` cloud is in fact *healthier* (min ESS 83)" — withdrawn.** Over 30 datasets the
>   minimum ESS is `24.2 +- 4.5` for `M < d` against `21.1 +- 4.3` for `M = d`: `|t| = 0.98`, and
>   `|t| = 0.06` on the mean ESS. The two are indistinguishable. The `82.7` above lies outside the
>   entire range of the 30 fresh datasets (`8.0`-`54.4`). What survives is the weaker statement that
>   `M < d` does not *degrade* the cloud, which still makes the point the claim was written for.
>
> Two further amendments. The `2.6x` concentration of error in the unmeasured coordinates measures
> `2.86 +- 0.11`, but `1.834x` of it is the posterior being wider there — data-free, so exact — and
> the filter's own share is `1.56 +- 0.06`. And the quoted ESS figures in the `d` sweep ("never below
> ~90", "pinned near 530") measure `56.0` and `506-526`.
>
> **The two failures have one cause.** `min ESS` and `max gap` are **extreme-value** statistics: they
> vary by 37-82% across datasets where the time-averaged quantities vary by 3-11%, and the minimum
> ESS additionally falls by `4.7-5.2x` from `T = 100` to `T = 1600` while the mean ESS moves by under
> `1%`. Both misled — `min ESS` on a headline claim, `max gap` on its ordering, which reverses end to
> end. The table's other five orderings, and the mean-based factors, reproduce. Treat the two
> extreme-value columns above as unsupported.

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

> **Correction (added after Part 12).** Everything above came from `DATA_SEED = 42` with 10 filter
> seeds. Part 3b error-barred the relative gaps and the long-horizon flatness; Part 12 screened the
> rest over 30 datasets.
>
> - **"the fitted `O(1/sqrt(N))` slopes survive" — withdrawn.** `M < d` at `rho = 1` fits
>   `-0.4693 +- 0.0151`, which is `3.98` standard errors from `-0.5` and `3.77` from `M = d`. The
>   single-dataset `-0.483` pointed the right way without being able to settle it. This is Part 11's
>   finding at `rho = 0.9` (`-0.4745 +- 0.0108`) repeated at the threshold — and the two are the
>   **same** to within a fraction of a standard error, so the rate deficit belongs to the direction
>   being unconstrained, not to the dynamics.
> - **The other four claims survive as patterns, not as figures.** All lie inside the datasets'
>   observed range and outside the 95% CI of the mean: `3.45 -> 3.673 +- 0.223`,
>   `0.110 -> 0.1616 +- 0.0284`, `0.198 -> 0.2425 +- 0.0218`, min ESS `82 -> 58.1 +- 12.7`.
> - **At `rho = 1` even the time averages need error bars.** The relative gap at `d = 16`, `M = 1`
>   varies by a factor of `7.1` across datasets, against `1.4`-`1.5` for the same quantities at
>   `rho = 0.9`. With a random-walk state each dataset explores a different region. Part 4b noticed
>   the same thing independently.
> - And the concentration ratio splits as Part 10 predicts: the posterior is `2.075x` wider in the
>   unmeasured coordinates (exact, being data-free) against `1.834x` at `rho = 0.9`, while the
>   filter's own share rises `1.562 -> 1.772 +- 0.107`.

---

# Part 3: Robustness — long horizons, error bars, and better proposals

Part 2's headline claims each rested on **one dataset**, **100 time steps**, and **one kind of
particle filter**. Part 3 attacks all three assumptions. Two claims survive; one needed correcting.

| Part 2 claim | Verdict |
|---|---|
| Error stays flat over time | **Confirmed and strengthened.** Flat to within 3% over `T = 2000`, with drift of a fraction of a percent per 1000 steps, inconsistent in sign across regimes. |
| Regimes perform near-identically relative to posterior width | **Corrected.** Over 30 datasets the relative gaps are `0.055`, `0.059`, `0.078` and every pairwise difference is statistically clear (`\|t\|` = 2.6, 12.6, 9.7). `M < d` is genuinely ~40% worse. The ordering survives; "almost identical" does not. *(Part 11 adds: the ordering holds under the Kolmogorov distance too, but the `40%` is metric-dependent — the same cells give `15%` under the bounded metric, and there `M > d` and `M < d` become indistinguishable.)* |
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

> **Correction (added after Part 12).** The in-model sections above used 10-200 **filter** seeds, all
> on `DATA_SEED = 42` — the design Part 11 showed averages away the wrong noise. Over 12 independent
> datasets the first key finding fails on both halves:
>
> - **"there is no difference" — too strong.** Three of six Student-t filters separate from the
>   Gaussian: `nu = 5` at `1.030x` (`|t| = 2.41`), `nu = 2` at `1.033x` (`|t| = 3.11`), and `nu = 1`
>   at **`1.101x`** with `|t| = 10.56`. All seven still sit between `0.0280` and `0.0308`, so the
>   substantive point holds; the right statement is "differences under 10%, and the Cauchy's is
>   real".
> - **"the Student-t filters look healthier (min ESS 33 vs 7.6)" — an artefact of which ESS was
>   reported.** On the same runs the **minimum** ESS says the Cauchy is healthier (`23.2 +- 4.2`
>   against `7.6 +- 2.3`, `|t| = 6.39`) and the **mean** ESS says it is worse (`395.7 +- 1.5` against
>   `504.6 +- 2.2`, `|t| = 80.07`). Two overwhelmingly significant results of **opposite sign**:
>   heavy tails occasionally throw particles far out, lifting the worst step and lowering the
>   average. The quoted `33` is also outside the observed range `7.1`-`32.1`.
>
> **The `nu ~ 10` claim, by contrast, is confirmed more firmly than one dataset could have shown it.**
> At `y* = 20` the maximum is interior in **12 of 12** datasets with the argmax in `{7, 10}` in
> **12 of 12**, and the peak is `3.23 +- 0.44x` the Gaussian against the quoted `3.2`. One amendment:
> at `y* = 50` the argmax moves to `nu = 15` (7 of 12) and the peak is `2.53 +- 0.10x`, not `3.2` —
> a larger outlier needs a heavier tail to be maximally awkward, which is the direction the mechanism
> predicts.


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

**All three of Part 5's single-dataset claims survive Part 12's screen, and one is strengthened.**
Every half-run ratio is consistent with 1 over 30 datasets (`0.9920 +- 0.0199`, `1.0028 +- 0.0167`,
`1.0113 +- 0.0176`, `|t| <= 1.26`), so the `0.960` at `b = 1.0` above — the value that looked like a
4% improvement and sat oddly against the other two — **was noise**, and the corrected picture has no
residual pattern in `b` at all. The coupling claim survives in the form it was made: it said the two
threshold penalties were *close* at `1.163` and `1.106`, and over 30 datasets they differ by `0.5%`
(`|t| = 0.20`). And **Section 7b, which flagged its own `d = 16` ratio of `1.104` as "within noise of
the others", was right**: `1.0127 +- 0.0263`, `t = +0.95`. All four of its hard cases sit on 1,
including the collapsed cloud at `d = 4`, `M = 16` with minimum ESS `1.01 +- 0.01` and resampling at
2000 of 2000 steps.

### Limitations

Only `Q = R = I` was tested, so the insensitivity above depends on the observations being
informative at every step. Section 9 used a single unobserved direction at `d = 2`; **Part 10 takes
that case apart** and finds the quantity that matters is not how many directions go unobserved but
whether the unstable ones are detectable. Section 7b used only 4 seeds, and its `d = 16` ratio of
`1.104` is within noise of the others. And `b` and `T` cannot both be pushed, so nothing here says
what happens at `b = 1.1` beyond `T ~ 241`.

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

**One further limitation, identified later.** Every run here is *detectable*: the observations
constrain every unstable direction of the state. Part 10 Section 8 repeats this measurement in a
direction they never constrain and finds the error growing without bound -- `0.038` to `0.610` over
`T = 2000` at `b = 1` -- so the flatness established above is a property of the models tested, not
of particle filters. Within those models it stands.

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

> **Correction (added after Part 13).** The paragraph above is withdrawn. Over 12 datasets instead
> of one, `sq_only` fits `-0.5007 +- 0.0207` -- `t = -0.07` from the theoretical `-0.5` and
> `t = -15.00` from `-0.342`, with Part 9's value `7.7` CI widths outside the interval and `0` of
> `12` datasets above `-0.40`. `sq_only` scales like everything else. The explanation offered above
> was an explanation of noise; there was nothing to explain. Part 13 Section 3 has the measurement.

`exp_only` and `softsign_only` remain unusable at `b = 1`: their information varies by `1e41` and
`3e3` along a single trajectory, which fixing the median cannot address.

---

---

# Part 10: The detectability boundary

Part 5 Section 9 found the one place in this project where the exact posterior stops being a
sensible thing to approximate: at `d = 2`, `M = 1`, `rho(F) = 1.1` the posterior variance in the
unobserved direction reached `1.25e13` against `0.64` when it was observed, while the ESS stayed
healthy at 50-65. The next-steps list proposed mapping that boundary by sweeping `M` from 1 to `d`.

> **Map the detectability boundary.** Sweeping `M` from 1 to `d` at fixed `b > 1` would measure how
> much observation is enough to keep the posterior bounded.

The sweep was run, over `84` cells at `d = 2, 4, 8`, `M = 1..d`, `b = 1.0, 1.01, 1.1` and coupling
`0` and `0.2`. **The premise was wrong, and that is worth more than the curve would have been.**

### Sections 1-6 cost nothing to run

The covariance recursion `P_t = (I - K_t H)(F P_{t-1} F' + Q)` never looks at `y`, so boundedness is
settled **without simulating any data** -- which is what makes it safe to ask above the threshold,
where Part 5 Section 8 showed a filter run measures floating-point cancellation rather than the
model. `posterior_cov_recursion` agrees with `kalman_filter_md` to **exactly zero**, bit for bit,
in all ten validation cells.

### The question has no answer in `M`

- **At `coupling = 0` it is a cliff, and the cliff is in the wrong variable.** Every cell with
  `M < d` diverges and every cell with `M = d` is bounded, with nothing in between. The diverging
  cells are not merely similar: at `b = 1.1` they all report `lam_max(P_T) = 6.1934e33`,
  *identically*, at every `d` and every `M` from 1 to `d-1`. **Seven observations of an
  eight-dimensional state are worth exactly as much as one.**
- **At `coupling = 0.2` the cliff is gone.** Every cell is bounded, including `M = 1` at `d = 8`,
  where one scalar per step holds the posterior variance of an explosive 8-dimensional state at
  `1815` instead of `1e33`.
- So the same sweep answers "all `d` of them" and "one is enough" depending on a parameter the
  question never mentioned. **Whatever decides boundedness, it is not the number of observations.**

### What does decide it, and it is exact

Detectability: every eigenvalue of `F` with `|lambda| >= 1` must be observable (the Hautus test,
`rank [[F - lambda I], [H]] = d`). It agreed with the measured recursion in **84 of 84 cells**, no
exceptions, computed from `(F, H)` by two rank calculations in microseconds where the recursion needs
1600 steps. `detectable` in `filters.py` now carries it.

The rate confirms the mechanism, not just the verdict. Every diverging cell grows at **exactly**
`2 log b` -- to `7.4e-5` at `b = 1.01` and to `0.00e0`, exactly in double precision, at `b = 1.1`.
The posterior variance in an unobserved direction grows at precisely the rate of the state's own:
the signature of a direction the data never constrain at all, rather than one they constrain weakly.

**At `b = 1` exactly the divergence is linear, not geometric:** `lam_max(P_t) = 1.0000 t + 0.0000`,
log-log slope `1.0000`, the constant being `Q = 1`. Each step adds one unit of process variance to a
direction that receives no information. A geometric rate fitted there reads `3.4e-3` and rounds to
"bounded" on any sensible tolerance, so the verdict has to be read from two horizons instead -- and
that choice is what makes the `84/84` possible. **The marginally stable case diverges too, just
polynomially, and a geometric test cannot see it.**

### The cliff belongs to `F`, not to `H`

At `coupling = 0`, `F = b I`, so `H F^k = b^k H` and every block of the observability matrix is a
multiple of the first: its rank is `rank H <= M` whatever `H` is. The control confirms it -- a dense
random `H` mixing every coordinate into every measurement gives `lam_max(P_T) = 6.1934e33` at every
`M < d`, **the same value to five figures as the sparse canonical selector**, at `d = 2, 4, 8` alike.
An observation can only see the subspace it points at, because `F` never rotates anything into it.

Two asides. At `M = d` the random `H` is *worse* (`18.8`, `9.47`, `23.5` against `0.639`), being
poorly conditioned. And at `d = 2`, `M = 1`, `coupling = 0.2` a random single row gives `20123`
against the canonical `4.24`, a factor of `4700`: a dense observation is not automatically a more
informative one. Detectability is binary; the bound it leaves behind is not.

### The boundary is sharp in theory and blurred in arithmetic

Coupling is the real knob, and **boundedness is discontinuous in it while the bound is continuous**:
at `d = 4`, `M = 1`, `b = 1.1` the posterior stays bounded for every `c != 0` while `lam_max` runs
`20.7 -> 37.0 -> 139 -> 1260 -> 6.0e5 -> 2.4e14` as `c` falls from `0.3` to `0.001`, then jumps to
`6.2e33`. The rank test is right about the limit and useless near it: it says "detectable, fine" at
`c = 0.001`, where the posterior variance is `2.4e14`.

`sigma_min(O)` was the natural continuous stand-in and it **only half works** -- a clean power law
within a fixed `d` (exponent `-2.08`, `-1.93`, `-1.89` at `d = 2, 4, 8`, close to the `-2` dimensional
analysis suggests) and a collapse across `d` (pooled exponent `-1.24`, residual **a factor of 706**).
The intercepts are what differ, because `sigma_min` of a `dM x d` matrix built from `d` powers of `F`
shrinks with `d` for reasons unrelated to how much information reaches the unstable mode. Reported as
a negative: there is a continuous detectability margin to be found, and this is not it.

**More seriously, the sharp boundary dissolves in double precision before it dissolves in theory.**
At `d = 8`, `sigma_min(O)` reaches `2.2e-16` at `coupling = 0.01` and the *numerical* rank drops to
7, then 6, then 5, while the mathematical rank is 8 throughout. The recursion follows, its rate
climbing `0.031, 0.066, 0.141, 0.180` towards `2 log b = 0.191`. **These models are detectable and
unfilterable.** A theorem resting on a rank gives no warning that the information it relies on is
carried at `1e-23`, and the dimension at which that happens is modest.

### What one more observation buys, when the posterior is bounded

Geometric decay: each extra observed coordinate divides the posterior spread by a roughly constant
factor, between `2.7` and `4.7` at `coupling = 0.2`. The factor is set by the coupling and is nearly
**independent of `d`** -- log-linear slope `-1.49` at both `d = 8` and `d = 12` for `c = 0.1`,
`-1.11` and `-1.10` for `c = 0.2`, `-0.97` and `-0.95` for `c = 0.3`. Stronger coupling buys a
*smaller* factor per observation, for the obvious reason that the first measurement has already done
more of the work. This is the curve the item was reaching for; it is not a boundary, because nothing
happens at any particular `M`.

### What this costs the particle filter: not what one would guess

Scored with Part 8's Kolmogorov distance, computed twice per step along the observable and
unobservable subspaces of `(F, H)` -- a split without which the effect averages away against the
healthy coordinates.

- **The posterior's spread is almost irrelevant.** Across detectable cells `lam_max(P)` spans
  `6.6x`, `57.8x` and `2839x` at `d = 2, 4, 8` while the distance spans `1.17x`, `1.22x` and
  `1.78x`. A posterior nearly three thousand times wider is approximated nearly as well.
- **With the posterior held *identical*, the error still varies two-fold.** The `coupling = 0`,
  `M < d` cells share one posterior to six figures, and across them the unobservable-subspace
  distance runs `0.299 -> 0.638` at `d = 4` and `0.422 -> 0.743` at `d = 8` as the ESS falls from
  `486` to `25`. That is Part 2's degeneracy, not detectability.
- **At matched `M` and matched ESS, undetectability costs up to `5x`** -- `0.299` against `0.112` at
  `d = 4, M = 1` with ESS `486` against `476`, so not a weight effect.
- **But at `d = 8` the sign reverses.** A **bounded** posterior with `lam_max = 1815` is harder
  (`0.551`) than an unbounded one (`0.422`), because where a direction is never observed the
  posterior along it *is* the prior propagation -- exactly what the bootstrap proposal samples from,
  so the proposal is locally optimal there for free. Where it is observed weakly and from far away,
  the data do constrain it, the proposal ignores them, and the particles land in the wrong place.
  This is Part 9 Section 9's failure mode in a linear model.

### The one result that changes a conclusion of this project

Part 3 Section 4, Part 3b, Part 5 Section 7 and all of Part 6 measured the error against `T` and
found it flat -- Part 6 to `T = 10^6`, with a bounded metric, so not an artefact of the statistic.
**Every one of those runs was detectable.** In the unobservable subspace the error grows without
bound:

| `b` | `t = 0` | `t = 199` | `t = 1999` | half-run ratio |
|---|---|---|---|---|
| 0.9 | 0.0379 | 0.0669 | 0.0814 | 1.003 |
| 0.99 | 0.0379 | 0.1685 | 0.1995 | 1.096 |
| 1.0 | 0.0379 | 0.3389 | 0.6099 | **1.405** |
| 1.01 | 0.0379 | 0.4816 | 0.8290 | **1.331** |

while the **observable** subspace of the very same runs stays at `0.040-0.045` at every `t` and every
`b`. The stable cases are their own control: `b = 0.9` is flat to `0.3%` over `T = 2000`.

`b = 1.0` is the cell that settles it. There `b^T = 1` exactly and the state is a random walk of
size `sqrt(T) ~ 45`, so **no floating-point reading is available** -- none of Part 5 Section 8's
cancellation applies -- and the distance still goes from `0.038` to `0.610`.

**The mechanism is measured, not inferred.** The cloud becomes progressively **under-dispersed**
along the unobserved direction: its spread, as a fraction of the exact posterior's, falls
`1.00 -> 0.81 -> 0.57 -> 0.50` at `b = 1.0` and to `0.16` at `b = 1.01`, while staying at
`0.99-1.01` at `b = 0.9`. The error in the posterior mean follows, to `0.62` posterior sd. The cause
is resampling: weights come from the observed coordinates alone, so every resampling event prunes
diversity in a direction that carries no information to justify the pruning, and above the threshold
the prior's pull cannot restore what is removed.

**And the standard diagnostic reports nothing.** The ESS sits at `500-650` of `1000` in all four
runs, no trend, no difference between `b = 0.9` and `b = 1.01`. A practitioner watching ESS would see
a healthy filter at every step of a run whose error had grown twenty-fold. This sharpens Part 5's
observation that the ESS stayed healthy while the posterior diverged: the ESS is blind here not
because the filter is fine but because it only ever measures the directions the likelihood touches.

### Two closing controls

- **More particles still help, at nearly the usual rate.** At fixed `t = 149` the distance falls
  `0.327 -> 0.057` over `N = 250` to `16000`, slope `-0.425` in `log N` against `-0.5`. So this is a
  constant in front of `N^-1/2` that grows with `t`, not a bias: consistent at any fixed horizon,
  uniformly accurate over none.
- **`b = 1.1` run to `T = 2000` shows exactly what Part 5's horizon limit protects against.** The two
  distances **freeze** -- `0.7288` and `0.5000`, identically, from `t = 499` to `t = 1999`, the
  dispersion ratio stuck at `0.6362`. Adding `O(1)` noise to a state of order `1e20` changes nothing,
  so every trajectory has become a deterministic multiple of its own past. An error constant to four
  decimals over fifteen hundred steps is not a result; the round `0.5000` is the signature of the
  arithmetic. Reported rather than deleted, because it would be easy to present as saturation.

### It also answers the proposal question (old next-steps item 10)

A **qualified no**, closer to the conjecture than to a clean negative. Over the 11 undetectable cells
the locally optimal proposal improves the unobservable subspace by `1.52x` (range `1.15-2.21`),
against `2.00x` for the observable subspace of the same cells and `2.51x` (up to `4.24x`) for the
detectable ones -- least help exactly where the posterior is unbounded, as predicted. The reason is
structural: there `p(x_t | x_{t-1}, y_t) = p(x_t | x_{t-1})`, so the two proposals **coincide**, and
what the optimal one buys is indirect, through resampling less often.

Indirect is not nothing. At `b = 1.0`, `T = 2000` it roughly halves the error (`0.610 -> 0.473`) and
holds the dispersion ratio at `0.789` instead of `0.505`. But both are still growing at `T = 2000`.
The proposal delays the failure by a constant factor and cannot remove it, because the object being
approximated is itself unbounded. Part 5's reading was right that this is not a proposal problem --
and Section 8 adds that it is not purely a posterior problem either, since the filter's error grows
faster than the posterior's own uncertainty.

### Limitations

1. **One family of `F`.** Everything rests on `build_model`'s symmetric tridiagonal base, where
   `coupling != 0` happens to make the first coordinate a cyclic vector and so makes `M = 1`
   sufficient. That is generic, not universal: an `F` whose unstable eigenvector is orthogonal to
   every power of `H` would be undetectable at any `M < d` with coupling present. The Hautus test
   handles that correctly, but no such case was **run**, so the `84/84` is agreement over one
   structured family.
2. **`Q = R = I` throughout**, as in Part 5. The geometric-in-`M` decay may well be a statement about
   isotropic noise.
3. **The near-undetectable regime is diagnosed, not measured.** A high-precision reference run would
   separate "the filter cannot use this information" from "this arithmetic cannot represent it".
4. **Section 8 uses one cell** (`d = 4`, `M = 1`, `coupling = 0`, 8 seeds). Whether the growth
   saturates at the Kolmogorov ceiling of 1, and at what `T`, is not established -- and `b = 1.1`
   cannot be used to find out.
5. **No continuous detectability margin was found.** A closed form for the information reaching the
   unstable eigenvector was tried during this work and discarded: it under-predicted `lam_max` by
   factors of 3 to 80, because summing the information available at each past time over-counts what
   is actually recoverable. That remains open.

---

---

# Part 11: Error bars on the rest of Part 2

Part 2 produced nine claims and a six-column summary table, all from **one simulated dataset**
(`DATA_SEED = 42`) averaged over ten particle-filter seeds. That design averages away the filter's
own Monte Carlo noise and leaves the *dataset's* contribution untouched: a quantity that happens to
be unusual on seed 42 stays unusual however many filter seeds are added.

Part 3 re-tested three of those claims over 30 independent datasets, and **one of the three needed
correcting**. This notebook tests the remaining six, at Part 2's own settings (`T = 100`,
`N = 1000`, `b = 0.9`) with Part 3's method (`data_seed = 100 + k`, `pf_seed = k`, `k = 0..29`, 95%
CIs, Welch's `t`). Where a claim concerns the *shape* of a curve — a slope, a ratio, a flattening —
the quantity is fitted **per dataset** and the 30 fits summarised, which is a paired design and
strictly better than one fit through 30-dataset averages.

### Two more claims do not survive

- **The `O(1/sqrt(N))` rate does not hold in every regime.** `M = d` and `M > d` sit on `-0.5`
  (`-0.5027 +- 0.0057` and `-0.5034 +- 0.0090`, `|t| < 1` against theory). `M < d` gives
  **`-0.4745 +- 0.0108`**: `4.63` standard errors from `-0.5`, and `4.53` from `M = d` on 44 degrees
  of freedom. Part 2's single-dataset fit of `-0.507` for that case was closer to the truth for the
  two regimes it was *not* describing than for the one it was.
- **The second half goes with it.** If the rates differ the penalty cannot be a constant factor, and
  it is not: the `M < d` / `M = d` ratio runs `1.928 -> 2.021 -> 2.076 -> 2.227` over `N = 50` to
  `5000`, a trend of `+0.062` per e-fold with paired `t = +4.37`. "Roughly `2x`" is right at
  `N = 1000` and becomes wrong as `N` grows.
- **Whether that matters is a separate question, and the honest answer is: a little.** Part 2's
  framing was "a worse constant is merely expensive, a worse rate would be fatal", which makes this
  the fatal case. The rate is worse by 5%: extrapolated, the penalty reaches `3x` around `N = 1e7`.
  The claim as stated is wrong; the decision it supported — `M < d` is expensive, not hopeless —
  stands.
- **The `M < d` cloud is not healthier.** Part 2 reported minimum ESS `82.7` against `29.3` and
  `15.6`. Over 30 datasets: `24.18 +- 4.50` against `21.08 +- 4.26` and `12.35 +- 3.60`, giving
  `|t| = 0.98` on the minimum and `|t| = 0.06` on the mean ESS. `M < d` and `M = d` are
  **indistinguishable**, not ordered — and `82.7` lies outside the entire observed range of the 30
  fresh datasets (`8.0` to `54.4`). It was not a noisy estimate of `24.18`; it was a value 30
  independent draws never came near.

### One claim needed its attribution split rather than its number corrected

The `M < d` error does concentrate in the unmeasured coordinates, at `2.86 +- 0.11` rather than
`2.6`. But the posterior is `1.834x` wider there — exactly, since the covariance recursion is
data-free — so most of the `2.86` is the model being harder, not the filter being worse. Dividing it
out leaves the filter's own share at `1.562 +- 0.061`, `t = 18.1` against 1. **Genuinely worse by
56%, not by 186%.** Part 2's prose half-knew this ("irreducible uncertainty [...], not particle
degeneracy") without separating the two numerically.

### Three survive, one of them better stated than before

- **The `d`-sweep factor reproduces almost exactly:** `1.921 +- 0.099` against Part 2's `1.94`
  (endpoints `0.0357 +- 0.0013` to `0.0680 +- 0.0025`).
- **The flattening is now established rather than asserted.** Increments of `+0.0271 +- 0.0046` over
  `d = 1-4` and `+0.0051 +- 0.0045` over `d = 4-16`; paired `t = +4.99`, with **28 of 30 datasets**
  showing the second increment smaller. A single dataset could not have supported a claim about the
  second difference of a five-point curve, and this one happened to be right.
- **Five of the table's six regime orderings survive.** The exception is `max gap`, where Part 2 read
  `M > d < M < d < M = d` and the 30-dataset means give `M = d < M > d < M < d` — a reversal of both
  ends.
- What fails *inside* the surviving claims are the quoted ESS figures: "min ESS never below ~90" and
  "mean ESS pinned near 530" measure `56.0 +- 14.7` and `506-526`.

### The failures have one cause, and it is not "single datasets are noisy"

Part 2's quantities fall into two groups an order of magnitude apart:

| quantity | sd / mean across datasets | max / min over 30 |
|---|---|---|
| `mean ESS`, `resampled` | 0.016-0.045 | 1.08-1.25 |
| `mean gap`, `rel gap` | 0.088-0.108 | 1.38-1.54 |
| `max gap` | 0.372-0.497 | **4.26-7.27** |
| `min ESS` | 0.520-0.815 | **6.79-20.90** |

Time-averaged quantities reproduce their orderings and their factors. The two **extreme-value**
quantities vary by 37-82%, and both misled: the minimum ESS on a headline claim, the maximum gap on
its ordering. Of the 15 sample-statistic table entries, **13 fall outside the 95% CI** of the
30-dataset mean, and all nine gap entries are **low** by 8-20%, because dataset 42 is a quiet
dataset for all three regimes at once — which no amount of filter-seed averaging can reveal.

`post_sd` is the one row a single dataset established *exactly*: zero spread across the 30, because
the Kalman covariance recursion never reads the data (Part 10 Section 1 checks this to bit
equality). The difficulty of each regime was known exactly; only the filter's performance on it
needed averaging.

**And the minimum ESS is worse than noisy — it is horizon-dependent.** Over the same datasets it
falls by `4.7-5.2x` from `T = 100` to `T = 1600` while the mean ESS moves by `0.1-0.8%`. It is a
minimum over a stationary series, so it falls with the length of the series by definition. This is
Part 6's growing maximum and Part 8's extreme-value resolution of it, now recognised in the
cloud-health diagnostic as well as in the error: **the minimum ESS over a run is not a property of a
filter, and no number of datasets fixes that — only quoting it at a stated horizon, or quoting the
mean instead, does.**

### The one new mechanism

Fitting the `N`-slope separately by coordinate at `d = 4`, `M = 2` locates the broken rate claim
exactly, and Part 10 supplies the language:

| coupling | observability rank | measured coords | unmeasured coords | paired difference |
|---|---|---|---|---|
| 0.3 | 4 / 4 | `-0.5035 +- 0.0051` | `-0.4670 +- 0.0143` | `+0.0365`, `t = +4.63` |
| 0.0 | **2 / 4** | `-0.5111 +- 0.0088` | **`-0.4413 +- 0.0171`** | `+0.0698`, `t = +7.89` |

The measured coordinates converge at exactly `-0.5` in both models. The **entire** deficit sits in
the coordinates the data do not pin down, and it **doubles** when those go from under-observed
(rank 4, the coupling eventually reveals them) to strictly unobservable (rank 2, where `F = 0.9 I`
never rotates them into view). Both models are detectable in Part 10's sense, since `|lambda| = 0.9`
keeps the posterior bounded either way; only the directness of the information changes.

So Part 2's broken claim is the same phenomenon Part 10 Section 8 found at the threshold, seen at
`b = 0.9` where it costs 5% of the convergence rate instead of all of it. **Parts 2, 10 and 11
describe one thing: the particle filter's error behaves differently in directions the observations
do not constrain, and how differently depends on how completely they fail to constrain them.**

### Does any verdict depend on Part 2's metric?

No. Re-scored with Part 8's Kolmogorov distance on the same 30 datasets, the ordering is
`M = d < M > d < M < d` under both metrics, with `|t| = 8.67` and `9.38` separating `M = d` from the
others. Two notes anyway: the **magnitude** is metric-dependent — Part 3's "`M < d` is ~40% worse" is
`1.402` under the relative gap and `1.152` under the Kolmogorov distance, where `M > d` and `M < d`
become indistinguishable (`|t| = 0.60`). And the smoke test of this very notebook, run with 4
datasets instead of 30, printed `THE TWO METRICS DISAGREE` on that same `|t| = 0.60` difference — an
apt accident in a notebook arguing for error bars, and caught only by running the design it argues
for.

### Limitations

1. **Part 2's settings only** (`T = 100`, `N = 1000`, `b = 0.9`), deliberately, so the comparison is
   clean. Nothing here says the surviving claims hold elsewhere.
2. **The rate result rests on four values of `N`** (`50` to `5000`), the grid Part 2 used. A
   shallower slope over two decades could be a slow transient rather than an asymptotic rate;
   distinguishing them needs `N` up to `1e5`-`1e6`, which was not run. The extrapolation to `N = 1e7`
   inherits that caveat and is an order of magnitude, not a prediction.
3. **The mechanism in Section 2b is one `(d, M)` pair.** The sweep over `M` that would turn it into a
   curve was not run.
4. **Per-dataset slopes are treated as 30 independent observations**, ignoring their own estimation
   error. For the slopes this is conservative — that error inflates the variance and widens the
   intervals — so the significant results stay significant.
5. **Part 2b has not been given error bars.** Its numbers rest on the same single-dataset design.

---

---

# Part 12: The same screen on Parts 2b, 4 and 5

Part 11 found *why* two of Part 2's claims failed, not just that they did: Part 2's quantities split
into two groups an order of magnitude apart in dataset-to-dataset variation, and both failures rested
on the high-variation group — **extreme-value statistics**, a minimum or maximum over a run.

Three parts were built on the same design and unscreened: **Part 2b** (`DATA_SEED = 42`, 10 filter
seeds), **Part 4** (10-200 **filter** seeds, all on one dataset), and **Part 5** (8 filter seeds).
Many filter seeds on one dataset removes the filter's Monte Carlo error and leaves the dataset's
untouched, which is precisely the wrong noise to average away.

### The screen was a prediction, not a fishing trip

Ten claims were classified — `time-avg`, `extreme`, `shape` — and each outcome **predicted from Part
11's rule before being run**, mechanically, with no case-by-case judgement. Section 7 scores it. That
distinction matters: a screen that tests ten claims and reports which failed is only as informative
as its prior, while one that commits to a rule first is a test of the rule.

### Two claims fall

- **Part 2b's "the `O(1/sqrt(N))` rate survives" — withdrawn.** `M < d` at `rho = 1` fits
  `-0.4693 +- 0.0151`: `3.98` standard errors from `-0.5`, `3.77` from `M = d`. Part 2b's `-0.483`
  pointed the right way without settling it.
- **My own prediction about the threshold was wrong, and the data say so cleanly.** Part 10's
  unbounded error growth suggested the deficit should be *larger* at `rho = 1`. It is `-0.4693` here
  against `-0.4745` at `rho = 0.9` — the same to a fraction of a standard error. The deficit belongs
  to the direction being unconstrained, not to what the dynamics do to it. Part 10's result is about
  the error at fixed `N` as `T` grows; this is about fixed `T` as `N` grows, and the threshold enters
  the first and not the second.
- **Part 4's in-model claim fails on both halves.** "No difference between the filters" is too
  strong — three of six Student-t filters separate from the Gaussian, the Cauchy by `1.101x` with
  `|t| = 10.56`. And "the Student-t filters look healthier" is an artefact of which ESS was reported:

  | diagnostic | Gaussian | `t nu = 1` | verdict | `\|t\|` |
  |---|---|---|---|---|
  | **minimum** ESS | `7.6 +- 2.3` | `23.2 +- 4.2` | Cauchy healthier | 6.39 |
  | **mean** ESS | `504.6 +- 2.2` | `395.7 +- 1.5` | Cauchy **worse** | 80.07 |

  Two overwhelmingly significant results of **opposite sign on the same runs**. Heavy tails
  occasionally throw particles far out, which lifts the worst single step and lowers the average.
  Part 4 quoted the one that flattered its subject.

### Part 5 comes out clean, and tidier than it was

All three claims survive; see the box in Part 5 above. The `0.960` half-run ratio at `b = 1.0` was
noise, the coupling penalties differ by `0.5%` rather than `5%`, and Section 7b's self-flagged
`1.104` was indeed within noise. Part 5 is the one part of this repo whose single-dataset claims were
all sound, including its sharpest — that a cloud running on one effective particle is inaccurate but
not diverging.

### Part 2b survives as patterns, not as figures

Four claims, every one inside the datasets' observed range and outside the 95% CI of the mean, three
low by 6-32%: `3.45 -> 3.673 +- 0.223`, `0.110 -> 0.1616 +- 0.0284`, `0.198 -> 0.2425 +- 0.0218`,
min ESS `82 -> 58.1 +- 12.7`.

### The rule scored 7 of 10, and the misses are the result

All three misses were one-sided — predicted to fail, actually survived. A rule scoring 10 of 10 would
only have shown it was consistent with the data it was built on.

| kind | n | predicted to fail | actually failed |
|---|---|---|---|
| `time-avg` | 5 | 0 | **0** |
| `extreme` | 3 | 3 | **1** |
| `shape` | 2 | 2 | **1** |

The time-average half is exact: 5 for 5 here, **0 failures in 8 pooled** across Parts 3, 3b, 11 and
12. The failure half over-predicts, for two reasons the misses identify.

- **An extreme-value statistic pinned against a boundary cannot mislead.** The ESS cannot fall below
  1. At `d = 4`, `M = 16` it measures `1.01 +- 0.01` with resampling at every one of 2000 steps, and
  at `M = 16` in Part 2b's sweep `1.48 +- 0.19` against a quoted `1.5`. Away from the floor the same
  statistic ranges over `8.0`-`147.0` on the same 30 datasets, a factor of `18`. What makes minima
  dangerous is variation in *where* the worst step falls, and a floor removes it.
- **The rule has no notion of effect size.** Part 4's `nu ~ 10` maximum is `3.23 +- 0.44x` the
  Gaussian — seven standard errors — and one dataset located it exactly: interior in **12 of 12**
  datasets, argmax in `{7, 10}` in **12 of 12**. Part 2's rate deficit is `0.025` on a slope of
  `0.5` and needed 30 paired fits. Both are shape claims from one dataset; only one was at risk.

Corrected by its own misses:

> A single-dataset claim is at risk when it rests on **where** something extreme happened — the worst
> step, the smallest ESS, the location of a maximum — **and** that quantity is free to move. It is
> safe when the quantity is a time average, when it is pinned against a boundary, or when the effect
> is large compared with the spread between datasets.

**Across the repo, 5 of 22 single-dataset claims have now failed a multi-dataset test** — Part 3 one,
Part 11 two, Part 12 two. A 23% failure rate, concentrated entirely in those categories, and the
number to keep in mind when reading any un-screened figure in Parts 1, 1b, 4b or 6.

### One qualification the screen produced about itself

At `rho(F) = 1` the relative gap at `d = 16`, `M = 1` varies by a factor of **7.1** across the 30
datasets, against `1.4`-`1.5` for Part 11's `rho = 0.9` quantities. The tidy split into reliable and
unreliable statistics is a `rho < 1` phenomenon: with a random-walk state each dataset explores a
different region, so at the threshold even time averages need error bars. Part 4b had noticed the
same thing and the rule does not capture it.

### Limitations

1. **Ten claims, chosen by hand.** They are the ones each part quoted as findings and that no earlier
   screen covered, but the selection was not mechanical.
2. **12 datasets, not 30, for the Student-t sections**, each needing a grid filter per
   (`nu`, dataset) pair. The `nu = 10` versus `nu = 7` split at `y* = 20` (9 against 3) is not
   resolved at that sample size.
3. **Parts 1, 1b, 4b and 6 remain unscreened.** Part 6 rests on 2 seeds at `T = 1e6` and 4 at `1e5`
   by its own account; Part 4b's intervals are explicitly three times wider than Part 4's.
4. **The corrected rule has not been tested on anything.** It was derived from the three misses here,
   so quoting it as a `7/10` rule would be circular. Its honest status is a hypothesis with a stated
   mechanism, and Parts 1, 1b, 4b and 6 are where it could be tested.

# Part 13: Optimal proposals beyond the linear-Gaussian case

Part 3 rescued the `M < d` particle filter with the **locally optimal proposal** of Doucet et al.
(2000), which samples from `p(x_t | x_{t-1}, y_t)` instead of from the prior transition. That rescue
leans on a closed form that exists only because the model is linear-Gaussian. Part 13 asks what is
left of it once `f` is nonlinear, so that the conditional has no closed form and the incremental
weight no longer collapses to `p(y_t | x_{t-1})`.

Four proposals, in `d = 1` where `grid_filter_obs` gives an exact benchmark: the **bootstrap** filter
of Parts 1-9; an **exact** locally optimal proposal computed by inverting a CDF on a local grid --
unaffordable in `d > 1`, and included as a *ceiling* rather than a method; a **linear** proposal from
linearising `f` at each particle's prior mean, which is the textbook EKF proposal; and a **mixture**
from linearising at every local maximum of the likelihood. Twenty datasets, paired within dataset,
scored by the Kolmogorov distance of Part 8.

### The cell it was sent to investigate did not need investigating

Part 9 named `sq_only` at `b = 0.9` because its error scaled at `-0.342` rather than `-0.5`. Over 12
datasets the paired slope is **`-0.5007 +- 0.0207`**: `t = -0.07` from `-0.5`, `t = -15.00` from
`-0.342`, and not one of the 12 datasets above `-0.40`. The claim is withdrawn above.

That is the **sixth** single-dataset claim in this repository to fail a multi-dataset test, and the
third of the six to be a *shape* claim -- a slope, a rate, a location. Part 12's corrected rule flags
exactly that category, and this is the first time it has predicted something it was not derived
from, which is what next-steps item 5 asked for.

### The rescue is real, and smaller than in the linear-Gaussian case

At `N = 1000`, as `KS(bootstrap) / KS(proposal)` paired within each of 20 datasets:

| cell | `exact` | `linear` | `mixture` |
|---|---|---|---|
| `sq_only`, `b = 0.9` | **`1.706 +- 0.120`** | `1.415 +- 0.058` | `1.321 +- 0.087` |
| `abs_only`, `b = 1` | `1.524 +- 0.167` | `1.446 +- 0.359` | `1.595 +- 0.334` |
| `abs_only`, `b = 0.9` | `1.249 +- 0.062` | **`0.837 +- 0.060`** | `1.247 +- 0.058` |
| `lin_xx`, `b = 0.9` | `1.403 +- 0.034` | `1.399 +- 0.027` | `1.413 +- 0.034` |

The ceiling is `1.25x` to `1.71x`, well short of Part 3's linear-Gaussian rescue, and the reason is
in the ESS: the bootstrap filter here already runs at `ESS/N = 0.44-0.56`, because Part 9 holds each
`f`'s observation precision fixed rather than letting the sharp ones dominate. A proposal that reads
`y_t` before moving the particles helps in proportion to how sharp the likelihood is, and these are
deliberately not sharp.

One linearisation recovers **`68% +- 7%`** of the ceiling in log KS -- except where the posterior is
bimodal, where it is **worse than doing nothing**: `0.837 +- 0.060` on `abs_only` at `b = 0.9`, a 16%
loss, because linearising `|x|` at the prior mean commits the proposal to whichever branch the mean
sits nearer. Linearising at *both* modes gives `1.247` against the ceiling's `1.249`. Where the mode
structure is the problem, covering the modes is the whole fix.

On a linear `f` all three agree at `1.40 +- 0.03`, as they must: linearisation is exact there, so the
three constructions are one filter. That control is what makes the spread above approximation error
rather than implementation noise.

### The approximations buy a constant and sell the rate

| `sq_only`, `b = 0.9` | paired slope | `t` vs `-0.5` | max weight slope | max weight at `N = 16000` |
|---|---|---|---|---|
| `bootstrap` | `-0.5007 +- 0.0207` | `-0.07` | `-0.946 +- 0.019` | `0.00042` |
| `exact` | `-0.5049 +- 0.0137` | `-0.71` | `-0.885 +- 0.006` | `0.00040` |
| `linear` | **`-0.4414 +- 0.0165`** | **`+6.94`** | **`-0.490 +- 0.024`** | **`0.00360`** |
| `mixture` | **`-0.4419 +- 0.0114`** | **`+10.01`** | **`-0.525 +- 0.028`** | `0.00325` |
| `linear t_5` | `-0.5015 +- 0.0102` | `-0.30` | `-0.809 +- 0.004` | `0.00084` |
| `mixture t_5` | `-0.4954 +- 0.0105` | `+0.87` | `-0.801 +- 0.007` | `0.00100` |

The two **approximate** proposals lose about `0.06` of slope at `t = 6.9` and `t = 10.0`; the
**exact** one does not move. That the exact proposal holds the rate is what rules out the
measurement as the cause. At `b = 1` the same split is wider: `linear` fits `-0.3550 +- 0.0470`
there, and is already *behind* the bootstrap filter at `N = 16000` (`0.0309` against `0.0278`) having
been well ahead at `N = 250`.

**The mechanism was written down before it was measured.** The Gaussian component has variance
`v = (1/Q + tau)^-1 < Q` whenever `f` carries local information, so the importance ratio
`p(x | x_{t-1}) G_t(x) / q(x)` grows like `exp((1/v - 1/Q) x^2 / 2)` in the tails. The proposal is
**thinner than the target it is correcting**, which is the one thing an importance distribution must
never be, and an unbounded ratio may have infinite variance. The independent symptom confirms it:
`1/N` at `N = 16000` is `6.25e-05`, and the heaviest normalised weight is **fifty-eight times** that
and falling at `N^-0.49` instead of `N^-0.95`.

**Giving each component `t_5` tails at the same variance repairs it, for nothing.** The rate returns
(`-0.5015 +- 0.0102`), the max-weight slope moves to `-0.81`, and the filter is *more accurate at
every `N` measured* -- `0.01030` against `0.01320` at `N = 16000`, a 22% reduction. Nothing was
fitted: `nu = 5` is the smallest round value with a variance to rescale, and only the tails change.

### Cost inverts the ranking

| proposal | cost per step | accuracy gain | what that budget buys as particles | wins? |
|---|---|---|---|---|
| `exact` | `31.0x` | `1.706x` | `5.578x` | no |
| `mixture` | `7.4x` | `1.321x` | `2.728x` | no |
| `mixture t_5` | `7.8x` | `1.394x` | `2.805x` | no |
| `linear` | `1.5x` | `1.415x` | `1.239x` | **yes** |
| `linear t_5` | `1.8x` | `1.521x` | `1.336x` | **yes** |

The exponent is measured, not assumed: the bootstrap filter's own slope is `-0.5007`, so `x` times
the cost buys `x^0.501`. **The cheapest approximation is the only one that pays for itself**, and
read with the section above, `linear t_5` is the single variant that wins on cost *and* keeps the
rate. That is the one line worth taking away.

### What generalises: it helps where the filter is broken, and hurts where it is not

Mean paired ratio over the whole of Part 9's group C plus a linear control:

| | `exact` | `linear` | `mixture` | `linear t_5` | `mixture t_5` |
|---|---|---|---|---|---|
| `b = 0.9` (11 functions) | **`1.39`** | `0.92` | `0.84` | `1.00` | `0.90` |
| `b = 1` (9 functions) | **`1.96`** | `1.07` | `1.51` | `1.05` | `1.37` |

The exact proposal helps in **every one of twenty cells**, by `1.07x` to `6.91x`. The approximations
average a coin flip at `b = 0.9`, with four outright disasters: `cos_only` under `mixture` at
**`0.05 +- 0.00`**, twenty times worse than doing nothing, then `xpluscos_only` at `0.37`,
`tanh_only` at `0.39` and `exp_only` under `linear` at `0.40`. Two causes, both intelligible --
a periodic or flat likelihood defeats a fixed budget of three modes (the mixture places almost no
density where the posterior actually is), and linearising `e^x` gives a near-zero proposal variance
at a point the linearisation got wrong.

Where they earn their place is where the bootstrap filter is failing badly. The worst cell in the
project is `log1px2_only` at `b = 1`, with a bootstrap distance of `0.30481` against the `0.047`
floor the rest of Part 9's matrix sits at. There the crude `mixture` scores **`7.96 +- 1.68`** and
**beats the exact proposal's** `6.91 +- 1.41`. So: **the gain from approximating the optimal proposal
scales with how far the bootstrap filter is from its own sampling limit.** At the limit there is
nothing to win and a badly shaped proposal can only lose.

### One thing a proposal cannot fix

Part 9's bimodality penalty grows with the mode separation `2|x_t|`. Splitting the per-step error by
`|x_t|` over 19 datasets, the ratio between the `|x| > 8` and `|x| < 2` bins is `2.94 +- 0.68`
(bootstrap), `2.80 +- 0.48` (exact), `2.07 +- 0.40` (linear) and `2.30 +- 0.83` (mixture) --
**statistically indistinguishable**. Every proposal shifts the curve down; none flattens it. A
locally optimal proposal is local *in time* by construction, and what resolves the sign of `x_t` when
the modes are fifty apart is the path. That is smoothing, not filtering, and it is next-steps item 2.

### The screen scored 3 of 7

Seven predictions were registered before the runs and four missed, each informatively: nothing was
supposed to help at `b = 1` (it does); `mixture` was supposed to beat `linear` there (`+0.149 +- 0.298`,
consistent with zero -- the effect is cleanly visible at `b = 0.9` instead, where the error bars are
three to six times narrower); the approximations were supposed to keep the `-0.5` rate (the most
useful miss, since it produced the mechanism and the fix); and nothing was supposed to beat the
matched-cost bootstrap filter (the cheap proposals do).

### Limitations

- **`d = 1` only.** Both the exact proposal and the exact benchmark need the grid. The tail-weight
  argument is algebra about `v < Q` and holds in any dimension, but it has been *measured* in one.
- **The mixture's mode budget is fixed at three**, and all four disasters above are cases where that
  is the wrong number. Keeping every mode within the prior's reach is the obvious repair, untested.
- **`nu = 5` was not tuned.** Whether the optimum is `3` or `30`, and whether it depends on `f`, is
  open. The only claim is that moving from infinite to `5` degrees of freedom restores the rate.
- **Two decades of `N`**, the same limitation item 12 raises about Part 11. The infinite-variance
  reading rests on the independent max-weight measurement rather than on the slope alone.
- **One potential.** Everything here is Gaussian, and a Student-t potential has heavier tails than
  any Gaussian proposal can match, so Section 6's balance would plausibly shift.
- **The `b = 1` cells remain noisy**: ratio CIs of `+-0.17` to `+-0.36` against `+-0.06` at
  `b = 0.9`, on 20 datasets. Several questions are answered at `b = 0.9` instead, and said so.

---

## Next steps

Items 1-3 of the original plan (long horizons, many-dataset error bars, better proposals) are **done
in Part 3**; varying `F` across the stability threshold is **done in Part 5**. Mapping the
detectability boundary, and asking whether a better proposal helps where the posterior is unbounded,
are **done in Part 10** -- the first by showing the boundary is not in `M` at all. Error bars on the
rest of Part 2 are **done in Part 11**, where two more of its claims fell, and on Parts 2b, 4 and 5
in **Part 12**, where two more fell and Part 5 came out clean. Optimal proposals beyond the
linear-Gaussian case are **done in Part 13**, which withdrew a sixth single-dataset claim on the way
and found that the textbook approximation gives up the `N^-1/2` rate. What genuinely remains:

1. **Sequential Quasi-Monte Carlo (SQMC).** The `O(1/sqrt(N))` rate confirmed in every regime is the
   fundamental limit of plain Monte Carlo. SQMC replaces random draws with low-discrepancy points
   and can converge faster; in more than one dimension this needs a **Hilbert space-filling curve**
   to order the multidimensional particles. The slopes measured here are the baseline any SQMC
   result should beat.
2. **Smoothing rather than filtering.** Estimating `p(x_t | y_0:T)` instead of `p(x_t | y_0:t)`
   should help most precisely where filtering struggled — the unmeasured coordinates of the `M < d`
   case, where later observations carry information about earlier unmeasured states. The Kalman
   smoother is again available as an exact benchmark.
3. **An adaptive mode budget for the mixture proposal (from Part 13).** Part 13's four disasters
   -- `cos_only` at `0.05`, `tanh_only` at `0.39`, `clip_only` at `0.13` -- are all cases where a
   fixed budget of three likelihood modes is the wrong number: a periodic `f` has many and a flat one
   has none. Keeping every mode inside the prior's reach, and falling back to the prior mean when
   there are none, is a small change with a clear prediction attached: it should remove every cell
   where the mixture scores below `1`, and change nothing elsewhere.
4. **Locate where the `log T` dependence enters (from Part 8).** Part 8 settles the metric
   question -- the mean distance carries no horizon dependence at all, while the `99.9`th
   percentile grows like `(log T)^0.35-0.71`. Which quantile first picks up a `log T` term would
   sharpen the reading of Section 7.5.2, and costs nothing beyond rerunning Part 8's Section 6 on a
   finer quantile grid. The `d`-dimensional orthant Kolmogorov distance, rather than Part 8's
   marginal version, is the other loose end.
5. **Test Part 12's corrected rule on Parts 1, 1b, 4b and 6.** Part 13 gave it its first
   out-of-sample test and it passed, flagging Part 9's `-0.342` as a shape claim from one dataset,
   which it was. That is one prediction. Part 6 is the natural next target: its conclusions rest on
   2 seeds at `T = 1e6` and 4 at `1e5` by its own account, and it is built almost entirely on
   extreme-value statistics -- which the rule now says are dangerous only where they are free to
   move. A screen there is both a test of the rule and the last unscreened part of the repo.
6. **A posterior-weighted curvature measure (from Part 9).** The relative residual of the affine fit
   predicts the cost of forcing the Gaussian reduction at a correlation of only `0.596`. Weighting
   the misfit by the posterior rather than by the sampling window would plausibly do better, and
   would turn Part 9 Section 6 from a ranking into a prediction.
7. **A state-dependent `Sigma`, or a compact state space (from Part 9).** `e^x` and `x/(1 + |x|)`
   cannot be run at `rho(F) = 1`: their Fisher information varies by `1e41` and `1e3` along one
   trajectory, so equalising its median is not enough. Either control would close the two cells
   Part 9 had to drop.
8. **A de-trended formulation for explosive long runs (from Part 5).** Filtering `z_t = F^-t x_t`,
   or working in log scale, would make `b = 1.1` at `T = 2000` measurable instead of losing it to
   floating-point cancellation. Part 10 Section 8b raises the stakes: run naively, that cell returns
   an error frozen at `0.7288` for fifteen hundred steps, which looks like saturation of a real
   effect and is not.
9. **A continuous detectability margin (from Part 10).** `sigma_min(O)` is a clean power law within a
   fixed `d` and fails across `d` by a factor of `706`. A closed form for the information reaching
   the unstable eigenvector was tried and discarded -- it under-predicts by `3-80x`, because summing
   the information available at each past time over-counts what is recoverable. Getting this right
   would turn Part 10's binary test into a quantity, and would predict `lam_max(P_inf)` rather than
   merely ordering it.
10. **An `F` that is undetectable *with* coupling (from Part 10).** Part 10's `84/84` agreement is
   over one structured family, where `coupling != 0` always makes `M = 1` sufficient. An `F` whose
   unstable eigenvector is orthogonal to every power of `H` would test the Hautus test where it is
   not nearly tautological, and is cheap to construct by hand.
11. **High precision in the near-undetectable regime (from Part 10).** At `d = 8`, `coupling <= 0.01`
   the numerical rank of `O` drops below `d` while the mathematical rank is full, and the posterior
   diverges at a rate climbing towards `2 log b`. A reference run in extended precision would
   separate "the filter cannot use information carried at `1e-23`" from "double precision cannot
   represent it" -- the first is a real limit on filtering, the second an artefact.
12. **Push the `N` grid to settle the `M < d` rate (from Part 11).** The shallower slope
   (`-0.4745 +- 0.0108` against `-0.5`) is fitted over `N = 50` to `5000`. Two decades cannot
   separate an asymptotic rate from a slow transient, and the distinction decides whether `M < d`
   has a genuinely worse rate or merely a penalty that is still settling. `N` up to `1e6` at
   `T = 100` is affordable.
13. **Sweep `M` for the per-coordinate rate (from Part 11).** Part 11 Section 2b shows the entire
   rate deficit living in the unobserved coordinates, and doubling when they become strictly
   unobservable, at one `(d, M)` pair. Sweeping `M` from 1 to `d` would turn that into a curve and
   connect it directly to Part 10's geometric-in-`M` result.
14. **Tune the proposal's tail weight, and test it in `d > 1` (from Part 13).** `nu = 5` restored
   the `N^-1/2` rate that the Gaussian approximation gave up, at no cost and without being tuned.
   The optimum over `nu`, whether it depends on `f`, and whether the same repair works on the
   `M < d` linear-Gaussian cells of Parts 2 and 3 -- where the proposal *is* exact, so the tails
   should not matter and the prediction is "no change" -- are all cheap. The `d > 1` version needs
   no exact benchmark, because the max-weight diagnostic of Part 13 Section 6 is computable without
   one.
15. **Close the Part 4 controls.** The random-walk variant is done (Part 4b) and the worst case is
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
