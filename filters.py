"""
filters.py
==========

Reusable filtering routines for the linear-Gaussian state-space models studied
in this project.

Part 1 (scalar)          Part 2 (multidimensional)
-----------------------  ----------------------------------------
x_t = b x_{t-1} + v_t    x_t = F x_{t-1} + v_t,  x_t in R^d
y_t = a x_t     + e_t    y_t = H x_t     + e_t,  y_t in R^M

In this linear-Gaussian setting the Kalman filter returns the EXACT posterior
p(x_t | y_0:t). The particle filter only approximates it with N samples, so the
Kalman filter serves as the ground-truth benchmark against which the particle
filter is validated.

Part 1 -- scalar (unchanged, so Part 1 keeps reproducing exactly)
-----------------------------------------------------------------
simulate_data            : generate true states and noisy observations
kalman_filter            : exact recursive posterior (mean and variance)
particle_filter          : bootstrap particle filter with ESS-triggered resampling

Shared by both parts
--------------------
systematic_resample      : low-variance resampling of particle indices
effective_sample_size    : "health meter" for the particle weights

Part 2 -- multidimensional
--------------------------
build_model              : construct F, Q, H, R for a given (d, M) case
stationary_covariance    : long-run spread of the state, as a scale yardstick
simulate_data_md         : simulate the vector-valued model
kalman_filter_md         : exact posterior mean and covariance (the benchmark)
particle_filter_md       : bootstrap particle filter in d dimensions
exact_posterior_bruteforce : non-recursive check that the benchmark is correct
pf_kf_gap                : dimension-normalised PF-vs-KF discrepancy
summarise_case           : the standard metrics reported for every case
run_case                 : build + simulate + filter + summarise, in one call

Supervised research project, Dr Mathieu Gerber, University of Bristol.
"""

import numpy as np

__all__ = [
    # Part 1 -- scalar
    "simulate_data",
    "kalman_filter",
    "particle_filter",
    # shared
    "systematic_resample",
    "effective_sample_size",
    # Part 2 -- multidimensional
    "build_model",
    "stationary_covariance",
    "simulate_data_md",
    "kalman_filter_md",
    "particle_filter_md",
    "exact_posterior_bruteforce",
    "pf_kf_gap",
    "summarise_case",
    "run_case",
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


# ===========================================================================
# PART 2 -- MULTIDIMENSIONAL EXTENSION
# ===========================================================================
#
# Part 1 above is scalar. Everything below generalises it to a d-dimensional
# hidden state observed through an M-dimensional measurement:
#
#     State:        x_t = F x_{t-1} + v_t,   v_t ~ N(0, Q),   x_t in R^d
#     Observation:  y_t = H x_t     + e_t,   e_t ~ N(0, R),   y_t in R^M
#
# The scalar functions are left untouched so Part 1 keeps reproducing exactly.
# The new functions carry an `_md` suffix ("multidimensional"). Running the
# general code at d = M = 1 and checking it against the scalar code is then a
# genuine test of the generalisation, not a tautology.
#
# Shape conventions used throughout:
#     x, states        (T, d)
#     y, observations  (T, M)
#     particles        (N, d)
#     F, Q             (d, d)
#     H                (M, d)
#     R                (M, M)
# ---------------------------------------------------------------------------


def build_model(d, M, b=0.9, q=1.0, r=1.0, coupling=0.3,
                h_structure="canonical", h_seed=0):
    """Build the model matrices (F, Q, H, R) for a given (d, M) pair.

    The three regimes studied in Part 2 differ only in the shape of H, so a
    single builder covers all of them.

    F -- state transition.
        Start from a symmetric tridiagonal matrix with 1 on the diagonal and
        `coupling` on the two neighbouring diagonals, so each coordinate is
        pushed by its neighbours rather than evolving in isolation. That base
        matrix is then rescaled to have spectral radius exactly `b`.

        The rescaling matters: a state-space model is stable (has a stationary
        distribution) only when the spectral radius of F is below 1. Simply
        adding coupling to 0.9 * I would push the spectral radius above 1 for
        larger d and the state would blow up. Rescaling guarantees stability at
        every d, which is essential for the dimension sweep in Section 6.

        At d = 1 the base matrix is [[1]], so F = [[b]] and we recover the
        scalar Part 1 model exactly.

    Q -- process noise covariance, q * I_d (isotropic).

    H -- observation matrix, M x d. With the default "canonical" structure,
        row k observes state coordinate (k mod d):

            M = d : H = I_d, every coordinate measured exactly once.
            M < d : H = [I_M | 0], only the first M coordinates are measured
                    directly; the remaining d - M are never observed and must
                    be inferred through the coupling in F.
            M > d : the identity rows repeat, so coordinates are measured more
                    than once. Two independent measurements of the same
                    coordinate, each with noise variance r, carry the same
                    information as one measurement with variance r/2 -- this is
                    the concrete meaning of "projecting the observation onto the
                    informative subspace".

        NOTE. This is one canonical choice of projection, picked because it is
        easy to reason about. `h_structure="random"` gives a random Gaussian H
        instead, for checking that conclusions are not an artefact of the
        canonical structure. Which construction is the right one for the wider
        study is being confirmed separately; nothing below depends on it.

    R -- observation noise covariance, r * I_M (isotropic).

    Parameters
    ----------
    d, M : int
        Hidden-state dimension and observation dimension.
    b : float
        Target spectral radius of F. Must be < 1 for a stable model.
    q, r : float
        Process and observation noise variances (isotropic).
    coupling : float
        Strength of the neighbour coupling in the base F. 0 gives independent
        coordinates.
    h_structure : {"canonical", "random"}
        Structure of H, as described above.
    h_seed : int
        Seed used only when h_structure="random".

    Returns
    -------
    F, Q, H, R : ndarrays of shapes (d,d), (d,d), (M,d), (M,M)
    """
    if d < 1 or M < 1:
        raise ValueError("d and M must both be at least 1")

    # --- F: coupled, then rescaled to spectral radius exactly b ---
    base = np.eye(d)
    if d > 1 and coupling != 0.0:
        base = base + coupling * (np.eye(d, k=1) + np.eye(d, k=-1))
    spectral_radius = np.max(np.abs(np.linalg.eigvals(base)))
    F = b * base / spectral_radius

    Q = q * np.eye(d)
    R = r * np.eye(M)

    # --- H: which state coordinates each measurement looks at ---
    if h_structure == "canonical":
        H = np.zeros((M, d))
        # Row k reads coordinate (k mod d). For M <= d this is [I_M | 0];
        # for M > d the identity rows wrap around and repeat.
        H[np.arange(M), np.arange(M) % d] = 1.0
    elif h_structure == "random":
        rng = np.random.default_rng(h_seed)
        # Scaled by 1/sqrt(d) so each measurement has O(1) magnitude
        # regardless of how many coordinates it mixes.
        H = rng.standard_normal((M, d)) / np.sqrt(d)
    else:
        raise ValueError("h_structure must be 'canonical' or 'random'")

    return F, Q, H, R


def stationary_covariance(F, Q):
    """Stationary covariance of the state, i.e. the P solving P = F P F' + Q.

    This is the long-run spread of the hidden state itself, with no data
    involved. It is useful as a yardstick: it tells us the natural scale of the
    problem, so filter errors can be judged relative to how much the state moves
    around in the first place.

    Solved by vectorising the equation into a linear system:
        vec(P) = (I - F (x) F)^{-1} vec(Q),
    where (x) is the Kronecker product. Fine for the small d used here.
    """
    d = F.shape[0]
    A = np.eye(d * d) - np.kron(F, F)
    return np.linalg.solve(A, Q.reshape(-1)).reshape(d, d)


def simulate_data_md(F, Q, H, R, T=100, x0=None, seed=42):
    """Simulate states and observations from the multidimensional model.

    Correlated Gaussian noise is generated with the Cholesky factor: if
    L L' = Q and z ~ N(0, I), then L z ~ N(0, Q).

    The random draws are made in the same order as the scalar `simulate_data`
    (one state noise, then one observation noise, per time step), so at
    d = M = 1 this reproduces the Part 1 data exactly.

    Parameters
    ----------
    F, Q, H, R : ndarray
        Model matrices, e.g. from `build_model`.
    T : int
        Number of time steps.
    x0 : ndarray, shape (d,) or None
        The state at time t = -1 that the first transition starts from.
        Defaults to the zero vector.
    seed : int
        Seed for reproducibility.

    Returns
    -------
    x : ndarray, shape (T, d)   -- true hidden states (never seen by a filter)
    y : ndarray, shape (T, M)   -- observations (the only filter input)
    """
    d = F.shape[0]
    M = H.shape[0]
    rng = np.random.default_rng(seed)

    LQ = np.linalg.cholesky(Q)
    LR = np.linalg.cholesky(R)

    x = np.zeros((T, d))
    y = np.zeros((T, M))

    x_prev = np.zeros(d) if x0 is None else np.asarray(x0, dtype=float)
    for t in range(T):
        x[t] = F @ x_prev + LQ @ rng.standard_normal(d)
        y[t] = H @ x[t] + LR @ rng.standard_normal(M)
        x_prev = x[t]

    return x, y


def kalman_filter_md(y, F, Q, H, R, m0=None, P0=None):
    """Exact filtering distribution p(x_t | y_0:t) for the multidimensional LG model.

    The same two steps as the scalar version, now with matrices:

    Predict:  m_pred = F m_{t-1}
              P_pred = F P_{t-1} F' + Q

    Update:   S = H P_pred H' + R          (covariance of the predicted y_t)
              K = P_pred H' S^{-1}         (Kalman gain, now a d x M matrix)
              m_t = m_pred + K (y_t - H m_pred)
              P_t = (I - K H) P_pred

    Because the model is linear with Gaussian noise, this is exact -- it is the
    true posterior, not an approximation. That is what makes it a valid
    benchmark for the particle filter in every regime below.

    Parameters
    ----------
    y : ndarray, shape (T, M)
        Observations.
    F, Q, H, R : ndarray
        Model matrices.
    m0 : ndarray, shape (d,) or None
        Prior mean for the state at t = -1. Defaults to zeros.
    P0 : ndarray, shape (d, d) or None
        Prior covariance at t = -1. Defaults to the ZERO matrix, meaning the
        starting state is known exactly. That matches how `simulate_data_md`
        and `particle_filter_md` start (both from a fixed x0), so the Kalman
        filter and the particle filter are solving the identical inference
        problem and any difference between them is purely Monte Carlo error.

    Returns
    -------
    m : ndarray, shape (T, d)      -- filtering means E[x_t | y_0:t]
    P : ndarray, shape (T, d, d)   -- filtering covariances
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    T = y.shape[0]
    d = F.shape[0]
    I = np.eye(d)

    m_prev = np.zeros(d) if m0 is None else np.asarray(m0, dtype=float)
    P_prev = np.zeros((d, d)) if P0 is None else np.asarray(P0, dtype=float)

    m = np.zeros((T, d))
    P = np.zeros((T, d, d))

    for t in range(T):
        # --- Predict ---
        m_pred = F @ m_prev
        P_pred = F @ P_prev @ F.T + Q

        # --- Update ---
        S = H @ P_pred @ H.T + R
        PHt = P_pred @ H.T
        # K = PHt @ inv(S), obtained by solving S' K' = PHt' rather than
        # inverting S explicitly, which is both faster and better conditioned.
        K = np.linalg.solve(S.T, PHt.T).T
        m[t] = m_pred + K @ (y[t] - H @ m_pred)
        P_new = (I - K @ H) @ P_pred
        # Round-off can make P slightly non-symmetric; force symmetry back so
        # the covariance stays a valid covariance over long runs.
        P[t] = 0.5 * (P_new + P_new.T)

        m_prev, P_prev = m[t], P[t]

    return m, P


def particle_filter_md(y, F, Q, H, R, N=1000, x0=None, seed=0,
                       ess_threshold=0.5, return_particles=False,
                       proposal="bootstrap"):
    """Bootstrap particle filter for the multidimensional model.

    Identical cycle to the scalar version -- propagate, weight, normalise,
    resample -- with two things now genuinely multidimensional:

    Propagation. Each particle is a point in R^d and moves as
        x_t^i = F x_{t-1}^i + L_Q z^i,     z^i ~ N(0, I_d),
    where L_Q is the Cholesky factor of Q.

    Weighting. The likelihood is now an M-dimensional Gaussian density:
        log p(y_t | x^i) = -0.5 (y_t - H x^i)' R^{-1} (y_t - H x^i) + const.
    The quadratic form is evaluated by solving with the Cholesky factor of R
    rather than forming R^{-1}, which is the numerically stable way to do it.
    The dropped constant (-M/2 log 2*pi - 0.5 log|R|) is the same for every
    particle, so it cancels when the weights are normalised.

    This is where the dimension regimes bite. M controls how many independent
    constraints each observation places on the particle cloud, and d controls
    how large a space the N particles have to cover. The whole study is about
    what happens as those two numbers move apart.

    Parameters
    ----------
    y : ndarray, shape (T, M)
        Observations.
    F, Q, H, R : ndarray
        Model matrices.
    N : int
        Number of particles.
    x0 : ndarray, shape (d,) or None
        Starting state for every particle at t = -1. Defaults to zeros, which
        matches the default prior of `kalman_filter_md` (a point mass there).
    seed : int
        Seed for the filter's own randomness.
    ess_threshold : float
        Resample when ESS < ess_threshold * N. The usual N/2 rule.
    return_particles : bool
        Also return the full weighted cloud at every step.
    proposal : {"bootstrap", "optimal"}
        How particles are moved forward.

        "bootstrap" (default) samples from the prior transition
        p(x_t | x_{t-1}), ignoring y_t until the weighting step. Simple and
        general, but it means particles are placed without any regard for
        where the observation says they should be -- so when the likelihood is
        sharp, few of them land anywhere useful and the weights become uneven.

        "optimal" samples from p(x_t | x_{t-1}, y_t) instead, which looks at
        the observation BEFORE moving the particles. For a linear-Gaussian
        model this distribution is available in closed form:

            Sigma* = (Q^-1 + H' R^-1 H)^-1
            m*_i   = Sigma* (Q^-1 F x_{t-1}^i + H' R^-1 y_t)

        and the incremental weight collapses to the predictive likelihood
        p(y_t | x_{t-1}^i) = N(y_t; H F x_{t-1}^i, H Q H' + R), which no longer
        depends on the newly sampled state at all. This is the locally optimal
        proposal of Doucet et al. (2000); it minimises the variance of the
        incremental weights given the past, and is the standard remedy when a
        sharp likelihood is degrading the bootstrap filter.

        Both choices target the SAME posterior, so both are still compared
        against the same exact Kalman benchmark.

    Returns
    -------
    est : ndarray, shape (T, d)     -- weighted-mean estimate of x_t
    ess : ndarray, shape (T,)       -- ESS after weighting, before resampling
    resampled : ndarray, shape (T,) -- whether resampling fired at each step
    particles_hist : ndarray, shape (T, N, d)  (only if return_particles)
    weights_hist   : ndarray, shape (T, N)     (only if return_particles)
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    T = y.shape[0]
    d = F.shape[0]
    rng = np.random.default_rng(seed)

    LQ = np.linalg.cholesky(Q)   # for sampling the process noise
    LR = np.linalg.cholesky(R)   # for evaluating the observation likelihood

    if proposal not in ("bootstrap", "optimal"):
        raise ValueError("proposal must be 'bootstrap' or 'optimal'")
    if proposal == "optimal":
        # All of these are constant in t, so they are formed once here rather
        # than rebuilt every step.
        Qinv = np.linalg.inv(Q)
        Rinv = np.linalg.inv(R)
        Sigma_star = np.linalg.inv(Qinv + H.T @ Rinv @ H)   # proposal covariance
        L_star = np.linalg.cholesky(Sigma_star)             # for sampling from it
        y_term = H.T @ Rinv                                 # applied to y_t below
        # Covariance of the predictive likelihood p(y_t | x_{t-1}), which is
        # what the incremental weight uses under this proposal.
        S_pred = H @ Q @ H.T + R
        L_pred = np.linalg.cholesky(S_pred)

    start = np.zeros(d) if x0 is None else np.asarray(x0, dtype=float)
    particles = np.tile(start, (N, 1))          # (N, d)
    weights = np.full(N, 1.0 / N)

    est = np.zeros((T, d))
    ess = np.zeros(T)
    resampled = np.zeros(T, dtype=bool)
    particles_hist = np.zeros((T, N, d)) if return_particles else None
    weights_hist = np.zeros((T, N)) if return_particles else None

    for t in range(T):
        if proposal == "bootstrap":
            # --- 1. Propagate every particle through the state equation ---
            # (N,d) @ (d,d) applies F to each particle; the noise term turns
            # standard normals into draws with covariance Q.
            z = rng.standard_normal((N, d))
            particles = particles @ F.T + z @ LQ.T

            # --- 2. Weight by the M-dimensional Gaussian likelihood ---
            innov = y[t] - particles @ H.T      # (N, M) residual per particle
            # Solve L_R u = innov' so that u'u = innov' R^{-1} innov, without
            # ever forming R^{-1}.
            u = np.linalg.solve(LR, innov.T)    # (M, N)
            quad = np.sum(u ** 2, axis=0)       # (N,) Mahalanobis distances
            log_w = -0.5 * quad + np.log(weights)
        else:
            # --- Optimal proposal: look at y_t BEFORE moving the particles ---
            prior_mean = particles @ F.T        # (N,d) where the prior alone points

            # Weight first: under this proposal the incremental weight is the
            # predictive likelihood p(y_t | x_{t-1}), which depends only on the
            # PREVIOUS particle, not on the new sample. That is exactly why this
            # proposal has lower weight variance than the bootstrap one.
            innov = y[t] - prior_mean @ H.T     # (N,M)
            u = np.linalg.solve(L_pred, innov.T)
            quad = np.sum(u ** 2, axis=0)
            log_w = -0.5 * quad + np.log(weights)

            # Then move the particles, pulled towards the observation.
            z = rng.standard_normal((N, d))
            mean_star = (prior_mean @ Qinv.T + y[t] @ y_term.T) @ Sigma_star.T
            particles = mean_star + z @ L_star.T

        # --- 3. Normalise with the log-sum-exp shift ---
        # Subtracting the max before exponentiating avoids underflow: without
        # it, log-weights far below zero all exp() to exactly 0 and the
        # normalisation becomes 0/0. The shift cancels in the division.
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        weights = w / np.sum(w)

        # --- Estimate: the posterior mean is the weighted average particle ---
        est[t] = weights @ particles            # (N,) @ (N,d) -> (d,)
        ess[t] = effective_sample_size(weights)

        if return_particles:
            particles_hist[t] = particles
            weights_hist[t] = weights

        # --- 4. Resample only when the cloud has become unhealthy ---
        if ess[t] < ess_threshold * N:
            idx = systematic_resample(weights, rng)
            particles = particles[idx]
            weights = np.full(N, 1.0 / N)
            resampled[t] = True

    if return_particles:
        return est, ess, resampled, particles_hist, weights_hist
    return est, ess, resampled


def exact_posterior_bruteforce(y, F, Q, H, R, m0=None, P0=None):
    """Filtering mean/covariance at the FINAL step, computed without recursion.

    This exists purely to verify the Kalman filter. Everything in this study
    treats the Kalman filter as ground truth, so it is worth checking that the
    ground truth is itself correct, using a completely different method.

    Method: under the linear-Gaussian model the whole trajectory
    (x_0, ..., x_{T-1}) and all observations (y_0, ..., y_{T-1}) are jointly
    Gaussian. We build that joint distribution explicitly from the model
    matrices and then condition on the observed y using the standard Gaussian
    conditioning formula

        E[X | Y = y] = mu_X + S_XY S_YY^{-1} (y - mu_Y)
        Cov[X | Y]   = S_XX - S_XY S_YY^{-1} S_YX

    and read off the block for the last time step. No filtering recursion is
    used anywhere, so agreement with `kalman_filter_md` is a real check.

    Only the LAST time step is comparable: conditioning on all of y_0:T-1 gives
    the *smoothing* distribution for earlier times, but for the final time step
    smoothing and filtering coincide. Call it at several values of T to check
    more than one point.

    Cost is O((T*d)^3), so use it on short runs only.

    Returns
    -------
    mean : ndarray, shape (d,)     -- E[x_{T-1} | y_0:T-1]
    cov  : ndarray, shape (d, d)   -- Cov[x_{T-1} | y_0:T-1]
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    T, M = y.shape
    d = F.shape[0]

    mean0 = np.zeros(d) if m0 is None else np.asarray(m0, dtype=float)
    cov0 = np.zeros((d, d)) if P0 is None else np.asarray(P0, dtype=float)

    # Marginal mean and covariance of each x_t, by rolling the model forward.
    mus = np.zeros((T, d))
    covs = np.zeros((T, d, d))
    mu_prev, cov_prev = mean0, cov0
    for t in range(T):
        mus[t] = F @ mu_prev
        covs[t] = F @ cov_prev @ F.T + Q
        mu_prev, cov_prev = mus[t], covs[t]

    # Joint covariance of the stacked state trajectory. For t >= s,
    # x_t = F^(t-s) x_s + (noise independent of x_s), so
    # Cov(x_s, x_t) = Cov(x_s) (F^(t-s))'.
    Sxx = np.zeros((T * d, T * d))
    Fpow = [np.linalg.matrix_power(F, k) for k in range(T)]
    for s in range(T):
        for t in range(s, T):
            block = covs[s] @ Fpow[t - s].T
            Sxx[s*d:(s+1)*d, t*d:(t+1)*d] = block
            Sxx[t*d:(t+1)*d, s*d:(s+1)*d] = block.T

    # Stack the observation model: Y = Hbig X + E, with block-diagonal Hbig.
    Hbig = np.zeros((T * M, T * d))
    Rbig = np.zeros((T * M, T * M))
    for t in range(T):
        Hbig[t*M:(t+1)*M, t*d:(t+1)*d] = H
        Rbig[t*M:(t+1)*M, t*M:(t+1)*M] = R

    mu_x = mus.reshape(-1)
    mu_y = Hbig @ mu_x
    Sxy = Sxx @ Hbig.T
    Syy = Hbig @ Sxx @ Hbig.T + Rbig

    # Condition the joint Gaussian on the observed y.
    resid = y.reshape(-1) - mu_y
    gain = np.linalg.solve(Syy, np.column_stack([resid, Sxy.T])).T
    post_mean = mu_x + Sxy @ np.linalg.solve(Syy, resid)
    post_cov = Sxx - Sxy @ np.linalg.solve(Syy, Sxy.T)

    return post_mean[-d:], post_cov[-d:, -d:]


# ---------------------------------------------------------------------------
# Metrics and a convenience runner
# ---------------------------------------------------------------------------
def pf_kf_gap(pf_est, kf_mean):
    """Per-time-step gap between the particle filter and the exact posterior.

        gap_t = || pf_est_t - kf_mean_t ||_2 / sqrt(d)

    Dividing by sqrt(d) turns the Euclidean norm into a ROOT-MEAN-SQUARE ERROR
    PER COORDINATE. This matters: a plain Euclidean norm grows with d simply
    because there are more coordinates to add up, which would make a
    high-dimensional case look worse even if every individual coordinate were
    estimated just as well. Normalising keeps the comparison across different d
    honest.

    Returns
    -------
    gap : ndarray, shape (T,)
    """
    pf_est = np.atleast_2d(pf_est)
    kf_mean = np.atleast_2d(kf_mean)
    d = pf_est.shape[1]
    return np.linalg.norm(pf_est - kf_mean, axis=1) / np.sqrt(d)


def summarise_case(pf_est, kf_mean, kf_cov, ess):
    """Standard set of numbers reported for every case in the study.

    Returns a dict with:
        mean_gap, max_gap : time-average and worst-case PF-KF gap
        min_ess, mean_ess : particle-cloud health
        post_sd           : mean posterior spread, sqrt(trace(P_t)/d) averaged
                            over t -- the genuine uncertainty in the problem
        rel_gap           : mean_gap / post_sd, the Monte Carlo error expressed
                            as a FRACTION of the real posterior uncertainty.
                            This is the most transferable number: it is
                            dimensionless and comparable across every regime,
                            and it answers "is the approximation error small
                            compared with the uncertainty we already have?"
        gap               : the full per-time-step gap series
    """
    d = np.atleast_2d(kf_mean).shape[1]
    gap = pf_kf_gap(pf_est, kf_mean)
    # sqrt(trace(P)/d) is the RMS posterior standard deviation per coordinate,
    # the same normalisation used for the gap so the ratio is meaningful.
    post_sd = np.mean(np.sqrt(np.trace(kf_cov, axis1=1, axis2=2) / d))
    return {
        "mean_gap": float(gap.mean()),
        "max_gap": float(gap.max()),
        "min_ess": float(ess.min()),
        "mean_ess": float(ess.mean()),
        "post_sd": float(post_sd),
        "rel_gap": float(gap.mean() / post_sd),
        "gap": gap,
    }


def run_case(d, M, N=1000, T=100, data_seed=42, pf_seed=0,
             proposal="bootstrap", **model_kwargs):
    """Build a model, simulate data, run both filters, and summarise.

    A thin wrapper over the four steps the notebook performs explicitly in its
    first sections: build_model -> simulate_data_md -> kalman_filter_md ->
    particle_filter_md. Having it here keeps the sweep sections short and
    guarantees every case is run identically.

    `data_seed` controls the simulated data, `pf_seed` the particle filter's own
    randomness. Keeping them separate makes it possible to hold the dataset
    fixed while varying only the Monte Carlo noise.

    Returns
    -------
    dict with keys: d, M, N, F, Q, H, R, x, y, kf_mean, kf_cov, pf_est, ess,
    resampled, and everything from `summarise_case`.
    """
    F, Q, H, R = build_model(d, M, **model_kwargs)
    x, y = simulate_data_md(F, Q, H, R, T=T, seed=data_seed)
    kf_mean, kf_cov = kalman_filter_md(y, F, Q, H, R)
    pf_est, ess, resampled = particle_filter_md(y, F, Q, H, R, N=N, seed=pf_seed,
                                               proposal=proposal)

    out = {
        "d": d, "M": M, "N": N, "T": T,
        "F": F, "Q": Q, "H": H, "R": R,
        "x": x, "y": y,
        "kf_mean": kf_mean, "kf_cov": kf_cov,
        "pf_est": pf_est, "ess": ess, "resampled": resampled,
    }
    out.update(summarise_case(pf_est, kf_mean, kf_cov, ess))
    return out
