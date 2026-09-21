"""
student_t_filters.py
====================

Part 4 helpers: the 1D particle filter run with a Student-t state noise.

Question (from Dr Gerber): in the 1D study, run the particle filter AS IF the
state noise v_t were Student-t, and compare against the particle filter run
with Gaussian v_t. The observations are generated from the ordinary Gaussian
model with a = 1, so they are unbounded. The conjecture is that the Student-t
particle filter should then be less stable than the Gaussian one.

    Data model (Gaussian, as in Part 1):
        x_t = b x_{t-1} + v_t,   v_t ~ N(0, Q)
        y_t = a x_t     + e_t,   e_t ~ N(0, R)

    Filter model used by the Student-t particle filter:
        x_t = b x_{t-1} + v_t,   v_t = s * T_nu,  T_nu ~ Student-t(nu)
        y_t = a x_t     + e_t,   e_t ~ N(0, R)

Because the Student-t model is not linear-Gaussian, the Kalman filter is no
longer its exact filter. So this module also provides a GRID FILTER: a
deterministic numerical filter that computes the exact filtering distribution
of either model on a fine grid. That lets the particle filter's approximation
error be measured against the exact answer for the model it is actually
running, in both cases.

This module does not modify filters.py. It imports the shared pieces from it.

Functions
---------
noise_scale             : Student-t scale for a given variance convention
transition_logpdf       : log density of v_t under either noise model
particle_filter_noise   : bootstrap PF with Gaussian or Student-t v_t
grid_filter             : exact numerical filter for either noise model

Supervised research project, Dr Mathieu Gerber, University of Bristol.
"""

import math

import numpy as np

from filters import systematic_resample, effective_sample_size

__all__ = [
    "noise_scale",
    "transition_logpdf",
    "particle_filter_noise",
    "grid_filter",
]


def noise_scale(Q=1.0, nu=None, match="scale"):
    """Scale s of the state noise v_t = s * T_nu.

    Two conventions, both useful:

    match="scale"     s = sqrt(Q). The Student-t has the same scale parameter
                      as the Gaussian N(0, Q). This is the direct reading of
                      "replace the Gaussian by a Student-t", and as nu -> inf
                      it recovers N(0, Q) exactly. For small nu its variance is
                      LARGER than Q (infinite for nu <= 2).

    match="variance"  s = sqrt(Q (nu - 2) / nu), so Var(v_t) = Q exactly, the
                      same as the Gaussian. Only defined for nu > 2. This is a
                      control: it removes "the Student-t is simply wider" as an
                      explanation, leaving only the difference in tail SHAPE.

    For the Gaussian (nu=None) both conventions give sqrt(Q).
    """
    if nu is None:
        return math.sqrt(Q)
    if match == "scale":
        return math.sqrt(Q)
    if match == "variance":
        if nu <= 2:
            raise ValueError("variance matching needs nu > 2 (finite variance)")
        return math.sqrt(Q * (nu - 2.0) / nu)
    raise ValueError("match must be 'scale' or 'variance'")


def transition_logpdf(v, noise="gaussian", nu=None, scale=1.0):
    """Log density of the state noise v under the chosen model.

    Gaussian:   log N(v; 0, scale^2)
    Student-t:  log of  Gamma((nu+1)/2) / (Gamma(nu/2) sqrt(nu pi) scale)
                        * (1 + (v/scale)^2 / nu)^(-(nu+1)/2)

    The key difference is in the tails. The Gaussian log density falls like
    -v^2 (so large jumps are essentially impossible); the Student-t log density
    falls only like -(nu+1) log|v| (so large jumps stay plausible).
    """
    v = np.asarray(v, dtype=float)
    z = v / scale
    if noise == "gaussian":
        return -0.5 * z ** 2 - math.log(scale) - 0.5 * math.log(2 * math.pi)
    if noise == "student_t":
        const = (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
                 - 0.5 * math.log(nu * math.pi) - math.log(scale))
        return const - 0.5 * (nu + 1) * np.log1p(z ** 2 / nu)
    raise ValueError("noise must be 'gaussian' or 'student_t'")


def particle_filter_noise(y, N=1000, b=0.9, a=1.0, Q=1.0, R=1.0, x0=0.0,
                          seed=0, ess_threshold=0.5, noise="gaussian",
                          nu=None, match="scale", return_particles=False):
    """Bootstrap particle filter whose transition noise is Gaussian or Student-t.

    Identical to Part 1's `filters.particle_filter` except for ONE line: the
    propagation step draws v_t from the chosen noise model. With
    noise="gaussian" it makes exactly the same random draws in exactly the same
    order as Part 1, so it reproduces Part 1's numbers bit for bit (checked in
    the notebook).

    It also returns the particle filter's estimate of the log-likelihood,

        log p_hat(y_0:t) = sum_s log( sum_i W_{s-1}^i g(y_s | x_s^i) ),

    where W_{s-1} are the normalised weights carried into step s. For a stable
    particle filter the variance of this estimate grows roughly LINEARLY in t;
    faster growth or sudden jumps are a standard signature of instability.

    Parameters
    ----------
    y : ndarray, shape (T,)
        Observations.
    N, b, a, Q, R, x0, seed, ess_threshold
        As in `filters.particle_filter`.
    noise : {"gaussian", "student_t"}
        Noise model the FILTER assumes for v_t.
    nu : float or None
        Degrees of freedom for the Student-t. Smaller = heavier tails.
    match : {"scale", "variance"}
        Scale convention; see `noise_scale`.
    return_particles : bool
        Also return the weighted particle cloud at every step.

    Returns
    -------
    est : ndarray (T,)        weighted-mean estimate of x_t
    ess : ndarray (T,)        ESS after weighting, before resampling
    resampled : ndarray (T,)  whether resampling fired
    loglik : ndarray (T,)     running log-likelihood estimate log p_hat(y_0:t)
    particles_hist, weights_hist : ndarray (T, N)   only if return_particles
    """
    y = np.asarray(y, dtype=float)
    T = y.shape[0]
    rng = np.random.default_rng(seed)
    s = noise_scale(Q, nu if noise == "student_t" else None, match)

    particles = np.full(N, float(x0))
    weights = np.full(N, 1.0 / N)

    est = np.zeros(T)
    ess = np.zeros(T)
    resampled = np.zeros(T, dtype=bool)
    loglik = np.zeros(T)
    particles_hist = np.zeros((T, N)) if return_particles else None
    weights_hist = np.zeros((T, N)) if return_particles else None

    # Constant of the Gaussian observation density. Part 1 dropped it because it
    # cancels in the weights; it is needed here for the likelihood estimate.
    log_g_const = -0.5 * math.log(2 * math.pi * R)
    running = 0.0

    for t in range(T):
        # --- 1. Propagate: the ONLY line that differs between the two filters ---
        if noise == "gaussian":
            particles = b * particles + s * rng.standard_normal(N)
        elif noise == "student_t":
            # Student-t draws: most are small, but occasionally a particle is
            # thrown a long way. That is the heavy tail.
            particles = b * particles + s * rng.standard_t(nu, size=N)
        else:
            raise ValueError("noise must be 'gaussian' or 'student_t'")

        # --- 2. Weight by the Gaussian observation likelihood (same as Part 1) ---
        log_w = -0.5 * (y[t] - a * particles) ** 2 / R
        log_w = log_w + np.log(weights)

        # Log-likelihood increment, via log-sum-exp for numerical stability:
        # log sum_i W_i g_i = max + log sum_i exp(log_w_i - max) + const.
        top = np.max(log_w)
        shifted = log_w - top
        w = np.exp(shifted)
        running += top + math.log(np.sum(w)) + log_g_const
        loglik[t] = running

        # --- 3. Normalise (the shift by the max is the Part 1 stabilisation) ---
        weights = w / np.sum(w)

        est[t] = np.sum(weights * particles)
        ess[t] = effective_sample_size(weights)

        if return_particles:
            particles_hist[t] = particles
            weights_hist[t] = weights

        # --- 4. ESS-triggered resampling, as in Part 1 ---
        if ess[t] < ess_threshold * N:
            idx = systematic_resample(weights, rng)
            particles = particles[idx]
            weights = np.full(N, 1.0 / N)
            resampled[t] = True

    if return_particles:
        return est, ess, resampled, loglik, particles_hist, weights_hist
    return est, ess, resampled, loglik


def grid_filter(y, b=0.9, a=1.0, Q=1.0, R=1.0, x0=0.0, noise="gaussian",
                nu=None, match="scale", half_width=None, h=0.02):
    """Exact filter for either noise model, computed on a fine grid.

    In one dimension the filtering recursion can be carried out numerically to
    very high accuracy, with no Monte Carlo noise at all:

        predict:  p(x_t | y_0:t-1) = sum_k f(x_t - b x_k) p(x_k | y_0:t-1) h
        update:   p(x_t | y_0:t)   is proportional to  g(y_t | x_t) p(x_t | y_0:t-1)

    where f is the density of v_t and g the Gaussian observation density. For
    the Gaussian model this reproduces the Kalman filter (checked in the
    notebook), and for the Student-t model it gives the exact answer that the
    Kalman filter cannot. It is the benchmark for the Student-t particle filter.

    The prediction is deliberately NOT renormalised. Probability mass that the
    Student-t kernel throws beyond the grid is simply lost, which is correct:
    the observation likelihood is essentially zero out there, so that mass
    would contribute nothing to the filter or to the likelihood anyway.

    Parameters
    ----------
    y : ndarray (T,)
        Observations.
    b, a, Q, R, x0 : float
        Model parameters; x0 is the known state at t = -1, as in the particle
        filter.
    noise, nu, match
        Noise model, as in `particle_filter_noise`.
    half_width : float or None
        Grid covers [-half_width, half_width]. By default it spans the largest
        |y|/|a| plus a margin of 20, far beyond any posterior mass.
    h : float
        Grid spacing.

    Returns
    -------
    mean : ndarray (T,)     exact filtering mean E[x_t | y_0:t]
    var : ndarray (T,)      exact filtering variance
    loglik : ndarray (T,)   exact running log-likelihood log p(y_0:t)
    """
    y = np.asarray(y, dtype=float)
    T = y.shape[0]
    s = noise_scale(Q, nu if noise == "student_t" else None, match)

    if half_width is None:
        half_width = np.max(np.abs(y)) / abs(a) + 20.0
    G = int(round(2 * half_width / h)) + 1
    grid = np.linspace(-half_width, half_width, G)
    step = grid[1] - grid[0]

    # Transition matrix K[j, k] = f(x_j - b x_k) * step, built once.
    K = np.exp(transition_logpdf(grid[:, None] - b * grid[None, :],
                                 noise, nu, s)) * step

    mean = np.zeros(T)
    var = np.zeros(T)
    loglik = np.zeros(T)
    log_g_const = -0.5 * math.log(2 * math.pi * R)

    running = 0.0
    post = None
    for t in range(T):
        if t == 0:
            # The state at t = -1 is the known point x0.
            pred = np.exp(transition_logpdf(grid - b * x0, noise, nu, s)) * step
        else:
            pred = K @ post

        g = np.exp(-0.5 * (y[t] - a * grid) ** 2 / R + log_g_const)
        joint = pred * g
        Z = joint.sum()                 # p(y_t | y_0:t-1)
        running += math.log(Z)
        loglik[t] = running

        post = joint / Z
        mean[t] = post @ grid
        var[t] = post @ grid ** 2 - mean[t] ** 2

    return mean, var, loglik
