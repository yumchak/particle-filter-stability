"""
filters.py
==========

Reusable filtering routines for the 1D linear-Gaussian state-space model studied
in this project (and a nonlinear variant used in the stretch section).

Model
-----
    State:        x_t = b * x_{t-1} + v_t,   v_t ~ N(0, Q)
    Observation:  y_t = a * x_t     + e_t,   e_t ~ N(0, R)

In this linear-Gaussian setting the Kalman filter returns the EXACT posterior
p(x_t | y_0:t). The particle filter only approximates it with N samples, so the
Kalman filter serves as the ground-truth benchmark against which the particle
filter is validated.

Functions
---------
simulate_data            : generate true states and noisy observations
kalman_filter            : exact recursive posterior (mean and variance)
systematic_resample      : low-variance resampling of particle indices
effective_sample_size    : "health meter" for the particle weights
particle_filter          : bootstrap particle filter with ESS-triggered resampling

Supervised research project, Dr Mathieu Gerber, University of Bristol.
"""

import numpy as np

__all__ = [
    "simulate_data",
    "kalman_filter",
    "systematic_resample",
    "effective_sample_size",
    "particle_filter",
]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def simulate_data(T=100, b=0.9, a=1.0, Q=1.0, R=1.0, x0=0.0, seed=42,
                  obs_fn=None):
    """Simulate the hidden states and the observations of the state-space model.

    Parameters
    ----------
    T : int
        Number of time steps to simulate (t = 0, ..., T-1).
    b : float
        State transition coefficient.
    a : float
        Observation coefficient (ignored when `obs_fn` is supplied).
    Q : float
        State (process) noise variance.
    R : float
        Observation noise variance.
    x0 : float
        Value of the state at time t = -1, i.e. the state the first transition
        starts from.
    seed : int
        Seed for the random number generator, so runs are reproducible.
    obs_fn : callable or None
        Optional nonlinear observation mean, a function h(x) used in place of
        a * x. Used for the nonlinear stretch example, y_t = x_t**2 / 20 + e_t.

    Returns
    -------
    x : ndarray, shape (T,)
        The true hidden states. In a real problem these are never observed;
        here we keep them so we can measure how well each filter tracks them.
    y : ndarray, shape (T,)
        The noisy observations, the only thing the filters are allowed to see.
    """
    rng = np.random.default_rng(seed)

    x = np.zeros(T)
    y = np.zeros(T)

    x_prev = x0
    for t in range(T):
        # Propagate the state one step forward and add process noise.
        x[t] = b * x_prev + np.sqrt(Q) * rng.standard_normal()
        # Observe the state through the (possibly nonlinear) observation map.
        mean_obs = a * x[t] if obs_fn is None else obs_fn(x[t])
        y[t] = mean_obs + np.sqrt(R) * rng.standard_normal()
        x_prev = x[t]

    return x, y


# ---------------------------------------------------------------------------
# Kalman filter: the exact benchmark
# ---------------------------------------------------------------------------
def kalman_filter(y, b=0.9, a=1.0, Q=1.0, R=1.0, m0=0.0, P0=1.0):
    """Exact posterior mean and variance of p(x_t | y_0:t) for a 1D LG model.

    The recursion alternates two steps at every time t:

    Predict  -- push the previous posterior through the state equation:
                m_pred = b * m_{t-1},   P_pred = b^2 * P_{t-1} + Q
                (the mean is scaled by b, the variance grows by the process
                noise Q: without data, uncertainty increases.)

    Update   -- fold in the new observation y_t via Bayes' rule:
                S = a^2 * P_pred + R        (variance of the predicted y_t)
                K = a * P_pred / S          (Kalman gain, in [0, 1] here)
                m_t = m_pred + K * (y_t - a * m_pred)
                P_t = (1 - K * a) * P_pred
                (the gain K decides how much to trust the new observation
                relative to the prediction; the variance always shrinks.)

    Parameters
    ----------
    y : ndarray, shape (T,)
        Observations.
    b, a, Q, R : float
        Model parameters, as in the module docstring.
    m0, P0 : float
        Mean and variance of the prior on the state before any data.

    Returns
    -------
    m : ndarray, shape (T,)
        Filtering means E[x_t | y_0:t] -- the exact posterior estimate.
    P : ndarray, shape (T,)
        Filtering variances Var[x_t | y_0:t].
    """
    y = np.asarray(y, dtype=float)
    T = y.shape[0]

    m = np.zeros(T)
    P = np.zeros(T)

    m_prev, P_prev = m0, P0
    for t in range(T):
        # --- Predict step: where do we think x_t is, before seeing y_t? ---
        m_pred = b * m_prev
        P_pred = b * b * P_prev + Q

        # --- Update step: correct the prediction using y_t. ---
        S = a * a * P_pred + R           # predicted variance of the observation
        K = a * P_pred / S               # Kalman gain
        innovation = y[t] - a * m_pred   # surprise: observed minus predicted
        m[t] = m_pred + K * innovation
        P[t] = (1.0 - K * a) * P_pred

        m_prev, P_prev = m[t], P[t]

    return m, P


# ---------------------------------------------------------------------------
# Resampling and diagnostics
# ---------------------------------------------------------------------------
def systematic_resample(weights, rng):
    """Systematic (low-variance) resampling: return the indices to keep.

    Idea: lay the normalised weights end to end on the interval [0, 1], so each
    particle occupies a segment as wide as its weight. Then place N equally
    spaced pointers on that interval, offset by a single uniform random draw,
    and record which segment each pointer lands in. Heavy particles cover more
    of the interval and so get selected several times; negligible ones are
    dropped.

    Compared with drawing N independent uniforms (multinomial resampling), the
    single shared random offset makes the selection far less noisy, which is why
    systematic resampling is the standard choice in practice.

    Parameters
    ----------
    weights : ndarray, shape (N,)
        Normalised weights (must sum to 1).
    rng : numpy.random.Generator
        Random number generator, passed in so results stay reproducible.

    Returns
    -------
    indices : ndarray of int, shape (N,)
        Indices of the resampled particles.
    """
    N = weights.shape[0]
    # N equally spaced pointers, shifted together by one uniform draw in [0,1/N).
    positions = (rng.random() + np.arange(N)) / N
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0  # guard against floating point leaving a tiny gap
    # searchsorted finds, for each pointer, which weight segment it falls into.
    return np.searchsorted(cumulative, positions)


def effective_sample_size(weights):
    """Effective Sample Size (ESS) of a set of normalised weights.

        ESS = 1 / sum_i w_i^2

    Interpretation: how many equally weighted particles the current weighted
    sample is "worth". If all weights are equal (w_i = 1/N) then ESS = N, the
    best case. If one particle carries all the weight then ESS = 1, meaning the
    approximation has collapsed onto a single sample -- weight degeneracy.
    ESS is therefore the health meter of the particle cloud.
    """
    return 1.0 / np.sum(weights ** 2)


# ---------------------------------------------------------------------------
# Bootstrap particle filter
# ---------------------------------------------------------------------------
def particle_filter(y, N=1000, b=0.9, a=1.0, Q=1.0, R=1.0, x0=0.0, seed=0,
                    ess_threshold=0.5, obs_fn=None, return_particles=False):
    """Bootstrap particle filter for the 1D state-space model.

    At each time step the filter performs the cycle:

    1. Propagate : move every particle forward with the state equation,
                   x_t^i = b * x_{t-1}^i + noise. This is sampling from the
                   prior transition, which is what makes it a *bootstrap* filter.
    2. Weight    : score each particle by how well it explains the new
                   observation, w^i proportional to p(y_t | x_t^i). Computed in
                   log space for numerical stability.
    3. Normalise : rescale the weights so they sum to 1, giving a discrete
                   approximation of the posterior.
    4. Resample  : if the ESS drops below `ess_threshold * N`, duplicate the
                   good particles and discard the poor ones, then reset all
                   weights to 1/N. This prevents degeneracy.

    Parameters
    ----------
    y : ndarray, shape (T,)
        Observations.
    N : int
        Number of particles.
    b, a, Q, R : float
        Model parameters.
    x0 : float
        Initial state value all particles start from at t = -1.
    seed : int
        Seed for the particle filter's own random number generator.
    ess_threshold : float
        Resample whenever ESS < ess_threshold * N. The usual choice is 0.5,
        i.e. the N/2 rule.
    obs_fn : callable or None
        Optional nonlinear observation mean h(x) used in place of a * x. The
        particle filter needs no change beyond this: it only ever evaluates the
        likelihood, never inverts the observation map. That is precisely why it
        works where the Kalman filter cannot.
    return_particles : bool
        If True, also return the full weighted particle cloud at every time
        step. Useful for inspecting the shape of the posterior (e.g. whether it
        is bimodal) rather than only its mean.

    Returns
    -------
    est : ndarray, shape (T,)
        Weighted-mean estimate of x_t at each time step.
    ess : ndarray, shape (T,)
        ESS after weighting at each time step (before any resampling).
    resampled : ndarray of bool, shape (T,)
        Whether resampling was triggered at each time step.
    particles_hist, weights_hist : ndarray, shape (T, N)
        Only returned when `return_particles=True`: the particles and their
        normalised weights at each time step, recorded after weighting and
        before any resampling.
    """
    y = np.asarray(y, dtype=float)
    T = y.shape[0]
    rng = np.random.default_rng(seed)

    # All particles start at x0; equal weights.
    particles = np.full(N, float(x0))
    weights = np.full(N, 1.0 / N)

    est = np.zeros(T)
    ess = np.zeros(T)
    resampled = np.zeros(T, dtype=bool)
    # Optional record of the full posterior approximation at every step.
    particles_hist = np.zeros((T, N)) if return_particles else None
    weights_hist = np.zeros((T, N)) if return_particles else None

    for t in range(T):
        # --- 1. Propagate: sample x_t^i ~ N(b * x_{t-1}^i, Q) ---
        particles = b * particles + np.sqrt(Q) * rng.standard_normal(N)

        # --- 2. Weight: log-likelihood of y_t under each particle ---
        mean_obs = a * particles if obs_fn is None else obs_fn(particles)
        # Gaussian log density, dropping constants that cancel on normalisation.
        log_w = -0.5 * (y[t] - mean_obs) ** 2 / R
        # Carry over the previous weights (they are all equal right after a
        # resampling step, but not otherwise).
        log_w = log_w + np.log(weights)

        # --- 3. Normalise, using the log-sum-exp trick ---
        # Subtracting the maximum before exponentiating is essential: raw
        # log-weights can be large and negative, and exp() of them underflows
        # to exactly 0, which would make every weight 0 and the normalisation
        # a 0/0. Shifting by the max makes the largest term exp(0) = 1, and
        # the shift cancels out when we divide by the sum.
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        weights = w / np.sum(w)

        # --- Estimate: posterior mean is the weighted average of particles ---
        est[t] = np.sum(weights * particles)
        ess[t] = effective_sample_size(weights)

        if return_particles:
            # Save the weighted cloud BEFORE resampling, so the recorded
            # weights still describe the posterior we just computed.
            particles_hist[t] = particles
            weights_hist[t] = weights

        # --- 4. Resample only when the cloud is unhealthy (ESS < N/2) ---
        # Resampling fights degeneracy but adds Monte Carlo noise, so we do it
        # only when needed rather than at every step.
        if ess[t] < ess_threshold * N:
            idx = systematic_resample(weights, rng)
            particles = particles[idx]
            weights = np.full(N, 1.0 / N)  # reset to uniform after resampling
            resampled[t] = True

    if return_particles:
        return est, ess, resampled, particles_hist, weights_hist
    return est, ess, resampled
