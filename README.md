# particle-filter-stability
# Particle Filter Stability in Sequential Monte Carlo

Supervised summer research project investigating the stability of particle 
filters in sequential Bayesian inference.

**Supervisor:** Dr Mathieu Gerber, Associate Professor in Statistical Science, 
University of Bristol
**Status:** Ongoing (from July 2026)

## Overview

Sequential Monte Carlo (SMC), or particle filtering, approximates a sequence of 
distributions p(x_t | y_0:t) as new observations arrive over time. This project 
studies the **stability** of that approximation: whether the gap between the 
particle estimate and the true distribution stays bounded as t grows, rather 
than accumulating error over time.

## Scope

- Bayesian filtering recursion (predict / update steps) in linear-Gaussian 
  state-space models
- Kalman filter as the exact closed-form benchmark
- Particle filter implementation, validated against the Kalman filter in the 
  linear-Gaussian case
- Investigation of resampling, weight degeneracy, and error accumulation over time

## Structure

(to be added as implementation progresses)

## References

- Gerber, M. & Chopin, N. (2015). *Sequential Quasi-Monte Carlo*. 
  Journal of the Royal Statistical Society: Series B.
