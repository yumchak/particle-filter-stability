# Particle Filter Stability

**Supervised research project — Dr Mathieu Gerber, Associate Professor in Statistical Science, University of Bristol**

## Overview

This project studies how well a particle filter approximates the exact Kalman filter in the
linear-Gaussian case, and how that approximation behaves as the number of particles `N` and the
time horizon `T` grow. In a linear-Gaussian state-space model the Kalman filter returns the exact
posterior `p(x_t | y_0:t)`, so it can be used as ground truth: any gap between the particle filter
and the Kalman filter is Monte Carlo error and nothing else. That makes this setting the ideal test
bed for measuring particle filter accuracy directly, checking that the error decays at the expected
rate in `N`, and — the question that actually drives the project — checking whether the error stays
bounded as `t` grows rather than accumulating step after step.

## The model

```
State (hidden):      x_t = b * x_{t-1} + v_t,   v_t ~ N(0, Q)
Observation (seen):  y_t = a * x_t     + e_t,   e_t ~ N(0, R)
```

Parameters used throughout: `b = 0.9`, `a = 1.0`, `Q = 1.0`, `R = 1.0`, `T = 100`, seed `= 42`.

## What's in this repo

| File | Contents |
|---|---|
| `kf_pf_lab.ipynb` | **The main study.** Simulates the model, runs both filters, compares the particle filter against the exact Kalman posterior, and runs the stability, ESS, and nonlinear experiments. All plots and outputs are saved in the notebook. |
| `filters.py` | Reusable filter functions imported by the notebook: `simulate_data()`, `kalman_filter()`, `particle_filter()`, `systematic_resample()`, `effective_sample_size()`. |

`filters.py` has no dependencies beyond NumPy, so the functions can be reused directly in later
experiments without the notebook.

## Key findings so far

- **The particle filter matches the exact Kalman filter.** At `N = 1000` over `T = 100` steps the
  mean absolute gap between the two estimates is `0.027` and the largest single-step gap is `0.18`,
  around 1.5% of the scale of the signal itself.
- **The error follows the O(1/sqrt(N)) Monte Carlo rate.** Fitting error against `N` on log–log axes
  gives a slope of `-0.493`, against the theoretical `-0.5`. Accuracy is bought with particles at a
  known exchange rate: four times the computation for twice the accuracy.
- **ESS-triggered resampling controls particle degeneracy.** Resampling whenever the effective sample
  size falls below `N/2` keeps the ESS oscillating between roughly `N/2` and `N` (mean `531`,
  minimum `29` at `N = 1000`) instead of collapsing toward a single dominant particle.
- **The error stays flat over time.** At every `N` tested, the second half of the run carries within
  about 10% of the first half's error, with no systematic upward drift. This is preliminary evidence
  of time-uniform stability: the error sits at a level set by `N` rather than accumulating with `t`.
- **The particle filter works where the Kalman filter cannot.** On the nonlinear observation model
  `y_t = x_t^2 / 20 + e_t` the particle filter runs unchanged apart from the likelihood evaluation,
  while no exact Kalman solution exists. Because squaring hides the sign of the state, the posterior
  there is genuinely **bimodal** — a shape no Gaussian filter can represent. The posterior mean is
  consequently uninformative (it averages the two modes to roughly zero), while the estimate of
  `|x_t|`, the quantity the data can identify, tracks the truth and cuts the error 56% below a
  constant baseline.

## Next steps

1. **Extend to higher dimensions.** Everything so far is one-dimensional. Particle filters are known
   to degrade sharply as the state dimension grows, with the required `N` potentially scaling
   exponentially with dimension; the next step is to reproduce that breakdown and locate where it
   begins to bite.
2. **Study the three cases relating hidden-state dimension `d` and observation dimension `M`:**
   `M < d` (partially observed — some state directions are never measured directly), `M = d` (fully
   observed), and `M > d` (over-observed — redundant measurements sharpen the likelihood, which is
   informative but makes the weights more uneven and degeneracy worse). How stability and the
   required `N` change across these regimes is the core of the next phase.
3. **Explore Sequential Quasi-Monte Carlo (SQMC).** The `O(1/sqrt(N))` rate confirmed here is the
   fundamental speed limit of plain Monte Carlo. SQMC replaces the random draws with low-discrepancy
   quasi-random points and can converge strictly faster, making it the frontier for beating that
   rate. The `-0.493` slope measured in this notebook is the baseline any SQMC result should be
   compared against.

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install numpy matplotlib jupyter
.venv/bin/jupyter notebook kf_pf_lab.ipynb
```

## Reference

Gerber, M. & Chopin, N. (2015). *Sequential Quasi-Monte Carlo.* Journal of the Royal Statistical
Society: Series B, 77(3), 509–579.
