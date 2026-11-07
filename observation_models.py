"""
observation_models.py
=====================

Parts 7 and 9: observation models beyond the linear-Gaussian case.

Everything in Parts 1-6 uses y_t = H x_t + e_t with Gaussian e_t. That model is
a special case of the Feynman-Kac representation in Del Moral & Gerber (2026),
whose potential function is their equation (9),

    G_t(x) = zeta_Gt( (y_t - H_t(x))' Sigma_Gt^{-1} (y_t - H_t(x)) ),

with zeta_Gt(z) = exp(-z/2) giving the Gaussian likelihood. This module lets
BOTH pieces vary: the function H_t (here written f, and allowed to be
nonlinear) and the shape zeta_Gt (Gaussian, or Student-t with nu degrees of
freedom).

Why it matters, and what the paper actually requires
----------------------------------------------------
Two separate conditions are at work, and they are easy to conflate.

1. (P1)b: the functions H_t must be L-bi-Lipschitz. The paper spells out the
   consequences: "under (P1)b, the dimension r of the observations y_t is
   larger or equal to the dimension d of the state space E, and the functions
   h_t are all unbounded", because "a function h : R^d -> R^r can only be
   bi-Lipschitz if r >= d and if ||h||_inf = infinity". So a BOUNDED f (tanh), a
   NON-INJECTIVE f (|x|, x^2, cos), or one whose slope vanishes or explodes
   (x + cos x on its own, e^x) all fall outside the theory.

   The paper also gives the repair used on the whiteboard: "if r > d and
   g : R^d -> R^{r-d} is a Lipschitz function, then ... the function
   h(x) := (f(x), g(x)) is bi-Lipschitz" for f bi-Lipschitz. That is exactly
   why [x + cos(x), x] is admissible while [x + cos(x)] alone is not.

2. Section 5.3.1 assumes H_t(R^d) is a d-dimensional LINEAR SUBSPACE of R^r --
   "for instance H_t(x) = A g(x) with A in R^{r x d} a matrix of rank d and
   g : R^d -> R^d a surjective function". Note what this does and does not say.
   The condition is on the IMAGE, not on f being linear: g may be nonlinear, so
   f(x) = [x^3, x^3] satisfies it while f(x) = [x + cos x, x] does not.

   When it holds, Section 5.3.2 decomposes the observation. With
   V_t = { Sigma^{-1/2} f(x) }, P_t the orthogonal projection onto V_t,

       zhat_t = P_t Sigma^{-1/2} y_t,   w_t = (I - P_t) Sigma^{-1/2} y_t,
       fhat_t = Sigma^{-1/2} f,

   Pythagoras gives their equation (39),

       G_t(x) = zeta_Gt( ||zhat_t - fhat_t(x)||^2 + ||w_t||^2 ).

   Section 5.3.3 then observes that for the EXPONENTIAL (Gaussian) choice this
   factorises, G_t(x) = exp(-||zhat - fhat(x)||^2) exp(-||w_t||^2), so the
   second factor is a constant in x and cancels when the weights are
   normalised. The r-dimensional observation is then equivalent to the
   d-dimensional zhat_t: the extra coordinates leave no trace. For any other
   zeta -- Student-t included -- ||w_t||^2 stays inside the function and does
   not cancel.

Putting the two together gives three regimes, which the notebooks test:

    f image linear + Gaussian   ->  reduces to d dimensions, w_t cancels
    f image linear + Student-t  ->  no reduction, w_t stays in the weights
    f image curved  + anything  ->  no reduction, and the "extra" coordinates
                                     genuinely carry information

No Kalman filter, and no posterior mean
---------------------------------------
Once f is nonlinear the model is not linear-Gaussian, so the Kalman filter is
not its exact filter and cannot be the benchmark. For d = 1 this module
provides `grid_filter_obs`, a deterministic numerical filter that computes the
exact filtering distribution to grid accuracy, exactly as `student_t_filters`
does for Part 4.

Worse, for a non-injective f such as |x| the exact posterior is BIMODAL and
symmetric, so its mean sits near zero and "distance between posterior means"
measures nothing at all. `ks_vs_grid` therefore compares the whole particle
distribution against the whole exact distribution, using the bounded
Kolmogorov distance introduced in Part 8. That metric is meaningful whatever
shape the posterior has.

Reference
---------
Del Moral, P. & Gerber, M. (2026). A perturbation theory for Feynman-Kac models
and their particle interpretations. arXiv:2610.21587. Equation (9) (potential),
Section 5.1.1 (P1)b (bi-Lipschitz), Sections 5.3.1-5.3.3 (linear observation
spaces, the projection, and why the Gaussian case collapses).
"""

import math
from dataclasses import dataclass, field

import numpy as np

from filters import systematic_resample, effective_sample_size


# ---------------------------------------------------------------------------
# 1. The potential shape zeta_G
# ---------------------------------------------------------------------------
#
# Each class below supplies two things that must agree with each other: the log
# potential used to WEIGHT particles, and a sampler for the observation noise
# used to GENERATE data. Keeping them in one object is what stops the two
# drifting apart, which would silently turn every experiment into a study of
# model misspecification instead.
#
# All of them are written in terms of the whitened residual
#     u = L^{-1} (y - f(x)),     Sigma = L L',
# so that u'u is the Mahalanobis form appearing in equation (9).


class GaussianPotential:
    """zeta_G(z) = exp(-z/2): the ordinary Gaussian likelihood.

    This is the one case where Section 5.3.3 applies, so it is the control
    against which every heavy-tailed variant is read.
    """

    name = "gaussian"
    nu = np.inf

    def log_weight(self, u):
        return -0.5 * np.sum(u ** 2, axis=-1)

    def sample_noise(self, n, r, rng):
        return rng.standard_normal((n, r))

    def __repr__(self):
        return "Gaussian"


class StudentPotentialJoint:
    """Multivariate Student-t: one common scale shared by all r coordinates.

        zeta_G(z) proportional to (1 + z/nu)^(-(nu + r)/2)

    This is the form written on the whiteboard, and it is a genuine density on
    R^r, so data can be generated from exactly the model being filtered. The
    single shared scale is what couples the coordinates: the orthogonal
    residual ||w_t||^2 of equation (39) sits inside the same bracket as
    ||zhat - fhat(x)||^2 and cannot be factored out.
    """

    def __init__(self, nu):
        if nu <= 0:
            raise ValueError("nu must be positive")
        self.nu = float(nu)
        self.name = f"student_t_joint(nu={nu:g})"

    def log_weight(self, u):
        r = u.shape[-1]
        return -0.5 * (self.nu + r) * np.log1p(np.sum(u ** 2, axis=-1) / self.nu)

    def sample_noise(self, n, r, rng):
        # t_r(0, I, nu) = Z / sqrt(W/nu) with Z ~ N(0, I_r) and W ~ chi^2_nu,
        # ONE W shared across the r coordinates.
        z = rng.standard_normal((n, r))
        w = rng.chisquare(self.nu, size=(n, 1))
        return z / np.sqrt(w / self.nu)

    def __repr__(self):
        return f"Student-t joint (nu={self.nu:g})"


class StudentPotentialIndep:
    """Independent Student-t in each coordinate.

        zeta_G(u) proportional to prod_j (1 + u_j^2/nu)^(-(nu + 1)/2)

    Not of the form (9) -- it is not a function of the single quadratic form --
    but it is the other natural reading of "Student-t observation noise", and
    the contrast is informative: here a single wild coordinate can be
    discounted on its own, whereas under the joint model it drags the whole
    observation vector with it.
    """

    def __init__(self, nu):
        if nu <= 0:
            raise ValueError("nu must be positive")
        self.nu = float(nu)
        self.name = f"student_t_indep(nu={nu:g})"

    def log_weight(self, u):
        return -0.5 * (self.nu + 1.0) * np.sum(np.log1p(u ** 2 / self.nu), axis=-1)

    def sample_noise(self, n, r, rng):
        z = rng.standard_normal((n, r))
        w = rng.chisquare(self.nu, size=(n, r))     # one W PER coordinate
        return z / np.sqrt(w / self.nu)

    def __repr__(self):
        return f"Student-t independent (nu={self.nu:g})"


class StudentPotentialPaper:
    """The paper's own choice: zeta_Gt(z) = phi(z^(1/2); nu).

    Section 5.1.1 suggests taking zeta_Gt(z) := phi(z^{1/2}; kappa_G) where phi
    is the density of the UNIVARIATE t_kappa(0, 1), giving

        zeta_G(z) proportional to (1 + z/nu)^(-(nu + 1)/2),

    i.e. exponent (nu + 1)/2 rather than the (nu + r)/2 of the multivariate
    density. The condition (P1)d is then met with nu_G = nu + 1 - d, and the
    paper notes the constraint nu_G > 0 required by (P2) holds when
    kappa_G is in (d - 1, infinity].

    Remark 5.1 of the paper is explicit that such a potential need not be
    integrable over R^r, so for r > 1 this is a valid POTENTIAL but not a
    likelihood, and there is no matching way to simulate data. `sample_noise`
    therefore refuses for r > 1 rather than quietly generating from a different
    model; use `StudentPotentialJoint` when data must be generated.
    """

    def __init__(self, nu):
        if nu <= 0:
            raise ValueError("nu must be positive")
        self.nu = float(nu)
        self.name = f"student_t_paper(nu={nu:g})"

    def log_weight(self, u):
        return -0.5 * (self.nu + 1.0) * np.log1p(np.sum(u ** 2, axis=-1) / self.nu)

    def sample_noise(self, n, r, rng):
        if r > 1:
            raise ValueError(
                "zeta(z) = phi(sqrt(z); nu) is not a density on R^r for r > 1 "
                "(Remark 5.1); use StudentPotentialJoint to generate data")
        z = rng.standard_normal((n, 1))
        w = rng.chisquare(self.nu, size=(n, 1))
        return z / np.sqrt(w / self.nu)

    def __repr__(self):
        return f"Student-t paper form (nu={self.nu:g})"


def potential(kind, nu=None):
    """Convenience constructor: potential('gaussian'), potential('t', 5), ..."""
    if kind in ("gaussian", "normal", None):
        return GaussianPotential()
    if kind in ("t", "student", "student_t", "joint"):
        return StudentPotentialJoint(nu)
    if kind in ("t_indep", "indep"):
        return StudentPotentialIndep(nu)
    if kind in ("t_paper", "paper"):
        return StudentPotentialPaper(nu)
    raise ValueError(f"unknown potential kind: {kind!r}")


# ---------------------------------------------------------------------------
# 2. Observation functions f
# ---------------------------------------------------------------------------
#
# Every entry records what it is EXPECTED to violate, so that Part 9 can print
# the prediction next to the measurement instead of rationalising afterwards.
# The two structural flags are the ones that matter:
#
#   linear_image -- is f(R^d) contained in a d-dimensional affine subspace?
#                   If yes, Section 5.3.3 applies under a Gaussian potential
#                   and the r-dimensional observation collapses to d dimensions.
#                   This is a condition on the IMAGE, so a nonlinear f can
#                   still satisfy it ([x^3, x^3] does).
#   bi_lipschitz -- "yes", "no", or "upper" when f is bounded below but not
#                   Lipschitz above (x^2, e^x paired with x), which is the
#                   half-way case the paper does not separately name.


@dataclass
class ObsFunction:
    name: str
    d: int
    r: int
    f: object                       # callable (n, d) -> (n, r)
    latex: str = ""
    linear_image: bool = False
    bi_lipschitz: str = "no"
    injective: bool = True
    bounded: bool = False
    smooth: bool = True
    group: str = ""
    note: str = ""
    offset: object = None           # constant added to the observation

    def __call__(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if x.shape[-1] != self.d:
            x = x.reshape(-1, self.d)
        out = np.asarray(self.f(x), dtype=float).reshape(x.shape[0], self.r)
        if self.offset is not None:
            out = out + np.asarray(self.offset, dtype=float)
        return out


def _col(x, j=0):
    return x[:, j]


def _stack(*cols):
    return np.column_stack(cols)


# --- Group A: image is a line, so the Gaussian reduction HOLDS ---------------
# These are the controls. [x^3, x^3] is the important one: nonlinear, yet the
# image is still a line, so the reduction must still work. It separates
# "nonlinear" from "breaks the reduction", which are routinely conflated.

_A = dict(group="A: linear image", linear_image=True, bi_lipschitz="yes")

REGISTRY = {}


def _add(fn):
    REGISTRY[fn.name] = fn
    return fn


_add(ObsFunction("lin_xx", 1, 2, lambda x: _stack(_col(x), _col(x)),
                 r"[x,\,x]", note="canonical M > d baseline", **_A))
_add(ObsFunction("lin_xx_offset", 1, 2, lambda x: _stack(_col(x), _col(x)),
                 r"[x,\,x] + [1,\,2]", offset=np.array([1.0, 2.0]),
                 note="affine, absorbed into y - b: should change nothing", **_A))
_add(ObsFunction("lin_x3x", 1, 2, lambda x: _stack(_col(x), 3 * _col(x)),
                 r"[x,\,3x]", note="unequal information per coordinate", **_A))
_add(ObsFunction("lin_x_minusx", 1, 2, lambda x: _stack(_col(x), -_col(x)),
                 r"[x,\,-x]", note="sign flip", **_A))
_add(ObsFunction("lin_xxx", 1, 3, lambda x: _stack(_col(x), _col(x), _col(x)),
                 r"[x,\,x,\,x]", note="r = 3, more replication", **_A))
_add(ObsFunction("cube_both", 1, 2,
                 lambda x: _stack(_col(x) ** 3, _col(x) ** 3),
                 r"[x^3,\,x^3]", bi_lipschitz="no", smooth=True,
                 group="A: linear image", linear_image=True,
                 note="NONLINEAR but image is still a line: reduction must hold"))
_add(ObsFunction("contract_both", 1, 2,
                 lambda x: _stack(_col(x) + 0.5 * np.sin(_col(x)),
                                  _col(x) + 0.5 * np.sin(_col(x))),
                 r"[g,\,g],\ g = x + \tfrac12\sin x",
                 note="g is a contraction perturbation, so bi-Lipschitz", **_A))


# --- Group B: curved image. Gaussian reduction FAILS -------------------------
# Pairing with x restores injectivity and the lower Lipschitz bound, which is
# the repair the paper gives for r > d. B1 keeps g Lipschitz (so f is fully
# bi-Lipschitz); B2 does not (so only the lower bound survives).

def _pair(g):
    return lambda x: _stack(g(_col(x)), _col(x))


_B1 = dict(group="B1: curved, bi-Lipschitz", linear_image=False,
           bi_lipschitz="yes")
_B2 = dict(group="B2: curved, lower bound only", linear_image=False,
           bi_lipschitz="upper")

_add(ObsFunction("cos_x", 1, 2, _pair(np.cos), r"[\cos x,\,x]", bounded=False,
                 note="bounded first coordinate", **_B1))
_add(ObsFunction("xpluscos_x", 1, 2, _pair(lambda v: v + np.cos(v)),
                 r"[x+\cos x,\,x]",
                 note="THE whiteboard model; x+cos x alone is not admissible",
                 **_B1))
_add(ObsFunction("sin_x", 1, 2, _pair(np.sin), r"[\sin x,\,x]", **_B1))
_add(ObsFunction("wiggle_x", 1, 2, _pair(lambda v: 0.5 * np.sin(5 * v)),
                 r"[\tfrac12\sin 5x,\,x]", note="high curvature", **_B1))
_add(ObsFunction("tanh_x", 1, 2, _pair(np.tanh), r"[\tanh x,\,x]",
                 note="saturating first coordinate", **_B1))
_add(ObsFunction("arctan_x", 1, 2, _pair(np.arctan), r"[\arctan x,\,x]", **_B1))
_add(ObsFunction("abs_x", 1, 2, _pair(np.abs), r"[|x|,\,x]", smooth=False,
                 note="Lipschitz but not differentiable at 0", **_B1))
_add(ObsFunction("log1px2_x", 1, 2, _pair(lambda v: np.log1p(v ** 2)),
                 r"[\log(1+x^2),\,x]",
                 note="whiteboard; even, so pairing restores injectivity", **_B1))
_add(ObsFunction("bump_x", 1, 2, _pair(lambda v: 1.0 / (1.0 + v ** 2)),
                 r"[1/(1+x^2),\,x]", **_B1))

_add(ObsFunction("sq_x", 1, 2, _pair(lambda v: v ** 2), r"[x^2,\,x]", **_B2))
_add(ObsFunction("cube_x", 1, 2, _pair(lambda v: v ** 3), r"[x^3,\,x]", **_B2))
_add(ObsFunction("exp_x", 1, 2, _pair(np.exp), r"[e^x,\,x]",
                 note="slope unbounded above", **_B2))
_add(ObsFunction("sqrtabs_x", 1, 2,
                 _pair(lambda v: np.sign(v) * np.sqrt(np.abs(v))),
                 r"[\mathrm{sign}(x)\sqrt{|x|},\,x]", smooth=False,
                 note="Holder, slope blows up at 0", **_B2))
_add(ObsFunction("circle", 1, 2,
                 lambda x: _stack(np.cos(_col(x)), np.sin(_col(x))),
                 r"[\cos x,\,\sin x]", group="B1: curved, bi-Lipschitz",
                 linear_image=False, bi_lipschitz="no", injective=False,
                 bounded=True, note="image is a circle: bounded AND periodic"))


# --- Group C: (P1)b violated outright, r = d = 1 -----------------------------
# No pairing, so nothing repairs them. These are the deliberate failures.

_C = dict(group="C: (P1)b violated", linear_image=True, bi_lipschitz="no")

_add(ObsFunction("abs_only", 1, 1, lambda x: np.abs(x), r"|x|",
                 injective=False, smooth=False,
                 note="BIMODAL posterior: the posterior mean is meaningless", **_C))
_add(ObsFunction("sq_only", 1, 1, lambda x: x ** 2, r"x^2",
                 injective=False, note="bimodal", **_C))
_add(ObsFunction("log1px2_only", 1, 1, lambda x: np.log1p(x ** 2),
                 r"\log(1+x^2)", injective=False, bounded=False,
                 note="whiteboard; even, so bimodal", **_C))
_add(ObsFunction("cos_only", 1, 1, lambda x: np.cos(x), r"\cos x",
                 injective=False, bounded=True,
                 note="periodic: infinitely many modes", **_C))
_add(ObsFunction("exp_only", 1, 1, lambda x: np.exp(x), r"e^x",
                 note="injective, but neither it nor its inverse is Lipschitz",
                 **_C))
_add(ObsFunction("tanh_only", 1, 1, lambda x: np.tanh(x), r"\tanh x",
                 bounded=True,
                 note="bounded, so cannot be bi-Lipschitz (||h||_inf < inf)", **_C))
_add(ObsFunction("softsign_only", 1, 1, lambda x: x / (1.0 + np.abs(x)),
                 r"x/(1+|x|)", bounded=True, smooth=False,
                 note="bounded bijection onto (-1, 1)", **_C))
_add(ObsFunction("xpluscos_only", 1, 1, lambda x: x + np.cos(x), r"x+\cos x",
                 note="injective, but slope 1 - sin x vanishes", **_C))
_add(ObsFunction("round_only", 1, 1, lambda x: np.round(x), r"\mathrm{round}(x)",
                 injective=False, smooth=False,
                 note="quantised: discontinuous, flat between jumps", **_C))
_add(ObsFunction("clip_only", 1, 1, lambda x: np.clip(x, -3.0, 3.0),
                 r"\mathrm{clip}(x,\pm 3)", injective=False, bounded=True,
                 smooth=False,
                 note="censored: the commonest failure in practice", **_C))


# --- Group D: under-observed, r < d (d = 2) ----------------------------------

_D = dict(group="D: r < d", linear_image=True, bi_lipschitz="no")

_add(ObsFunction("d2_first", 2, 1, lambda x: x[:, :1], r"x_1",
                 note="one coordinate never observed", **_D))
_add(ObsFunction("d2_sum", 2, 1, lambda x: (x[:, 0] + x[:, 1]).reshape(-1, 1),
                 r"x_1+x_2", **_D))
_add(ObsFunction("d2_norm", 2, 1,
                 lambda x: np.linalg.norm(x, axis=1).reshape(-1, 1),
                 r"\|x\|", linear_image=False, injective=False,
                 group="D: r < d", bi_lipschitz="no",
                 note="radial: the whole circle is indistinguishable"))
_add(ObsFunction("d2_prod", 2, 1, lambda x: (x[:, 0] * x[:, 1]).reshape(-1, 1),
                 r"x_1x_2", linear_image=False, injective=False,
                 group="D: r < d", bi_lipschitz="no"))


# --- Group E: genuinely multivariate (d = 2) ---------------------------------

_E = dict(group="E: d = 2 nonlinear", linear_image=False, bi_lipschitz="no")

_add(ObsFunction("d2_polar", 2, 2,
                 lambda x: _stack(np.linalg.norm(x, axis=1),
                                  np.arctan2(x[:, 1], x[:, 0])),
                 r"[\|x\|,\,\mathrm{atan2}(x_2,x_1)]",
                 note="bi-Lipschitz away from the origin only", **_E))
_add(ObsFunction("d2_coupled", 2, 2,
                 lambda x: _stack(x[:, 0] + np.cos(x[:, 1]), x[:, 1]),
                 r"[x_1+\cos x_2,\,x_2]", bi_lipschitz="yes",
                 group="E: d = 2 nonlinear", linear_image=False,
                 note="nonlinearity couples the coordinates"))
_add(ObsFunction("d2_xxprod", 2, 3,
                 lambda x: _stack(x[:, 0], x[:, 1], x[:, 0] * x[:, 1]),
                 r"[x_1,\,x_2,\,x_1x_2]", bi_lipschitz="upper",
                 group="E: d = 2 nonlinear", linear_image=False,
                 note="product term lifts the image off any plane"))
_add(ObsFunction("d2_bearing", 2, 1,
                 lambda x: np.arctan2(x[:, 1], x[:, 0]).reshape(-1, 1),
                 r"\mathrm{atan}(x_2/x_1)", injective=False, bounded=True,
                 group="E: d = 2 nonlinear", linear_image=False,
                 bi_lipschitz="no", note="bearings-only tracking"))
_add(ObsFunction("d2_range", 2, 1,
                 lambda x: np.linalg.norm(x, axis=1).reshape(-1, 1),
                 r"\sqrt{x_1^2+x_2^2}", injective=False,
                 group="E: d = 2 nonlinear", linear_image=False,
                 bi_lipschitz="no", note="range-only tracking"))
_add(ObsFunction("d2_range_bearing", 2, 2,
                 lambda x: _stack(np.linalg.norm(x, axis=1),
                                  np.arctan2(x[:, 1], x[:, 0])),
                 r"[\mathrm{range},\,\mathrm{bearing}]",
                 group="E: d = 2 nonlinear", linear_image=False,
                 bi_lipschitz="no",
                 note="the tracking model of the paper's Section 5.5.2"))


def by_group(group_prefix=None, d=None):
    """Registry entries filtered by group letter and/or state dimension."""
    out = []
    for fn in REGISTRY.values():
        if group_prefix is not None and not fn.group.startswith(group_prefix):
            continue
        if d is not None and fn.d != d:
            continue
        out.append(fn)
    return out


# ---------------------------------------------------------------------------
# 3. Structural diagnostics
# ---------------------------------------------------------------------------
#
# The registry records what each f is SUPPOSED to be. These functions measure
# it instead, from the function alone, so Part 9 can put the claim and the
# measurement side by side. Both of the paper's conditions are checkable:
# the dimension of the image is a rank question, and bi-Lipschitz is a pair of
# bounds on ||f(x) - f(x')|| / ||x - x'||.


def image_dimension(fn, xs=None, tol=1e-9, span=6.0, n=4000, seed=0):
    """Dimension of the smallest affine subspace containing f(R^d), numerically.

    Section 5.3.1 needs f(R^d) to be a d-dimensional linear subspace. Sampling
    the image and taking an SVD of the CENTRED values answers it directly: the
    number of singular values materially above zero is the dimension of the
    affine hull. Centring is what makes the [x, x] + [1, 2] case come out at 1
    rather than 2 -- a constant offset is absorbed into y - b and changes
    nothing, which is exactly the point of including that entry.

    Returns (affine_dim, linear_dim, normalised singular values).
    """
    if xs is None:
        rng = np.random.default_rng(seed)
        xs = rng.uniform(-span, span, size=(n, fn.d))
    vals = fn(xs)
    s_lin = np.linalg.svd(vals, compute_uv=False)
    s_aff = np.linalg.svd(vals - vals.mean(axis=0), compute_uv=False)
    norm_aff = s_aff / max(s_aff[0], 1e-300)
    return (int((norm_aff > tol ** 0.5).sum()),
            int((s_lin / max(s_lin[0], 1e-300) > tol ** 0.5).sum()),
            norm_aff)


def lipschitz_bounds(fn, span=6.0, n_pairs=200_000, seed=0, eps=1e-6):
    """Empirical upper and lower Lipschitz constants of f on [-span, span]^d.

    bi-Lipschitz means there are constants 0 < c <= C with
        c ||x - x'|| <= ||f(x) - f(x')|| <= C ||x - x'||.
    The ratio is sampled over random pairs, both far apart and very close
    together (close pairs are what expose a vanishing or exploding slope, and
    far pairs are what expose non-injectivity).

    A lower constant near zero means the lower bound fails -- either f is not
    injective, or its slope vanishes somewhere. A huge upper constant means f
    is not Lipschitz. Note this is a finite-window estimate: a function that
    only misbehaves far outside [-span, span] will look fine here.
    """
    rng = np.random.default_rng(seed)
    half = n_pairs // 2
    a = rng.uniform(-span, span, size=(n_pairs, fn.d))
    b = np.empty_like(a)
    b[:half] = rng.uniform(-span, span, size=(half, fn.d))          # far pairs
    b[half:] = a[half:] + eps * rng.standard_normal((n_pairs - half, fn.d))
    dx = np.linalg.norm(a - b, axis=1)
    keep = dx > 0
    ratio = np.linalg.norm(fn(a) - fn(b), axis=1)[keep] / dx[keep]
    return float(ratio.min()), float(ratio.max())


def injectivity_gap(fn, span=6.0, n=3000, seed=0):
    """Smallest ||f(x) - f(x')|| over well-separated pairs x, x'.

    Near zero means two states far apart produce the same observation, so the
    likelihood cannot tell them apart and the posterior is multimodal. This is
    what makes the posterior MEAN useless for |x|, x^2 and cos x, and why
    Part 9 scores those cases with a distributional metric instead.
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-span, span, size=(n, fn.d))
    vals = fn(xs)
    best = np.inf
    for i in range(0, n, 256):                 # blocked, to bound memory
        blk = xs[i:i + 256]
        dx = np.linalg.norm(blk[:, None, :] - xs[None, :, :], axis=2)
        dv = np.linalg.norm(vals[i:i + 256, None, :] - vals[None, :, :], axis=2)
        mask = dx > 1.0                        # "well separated"
        if mask.any():
            best = min(best, float(dv[mask].min()))
    return best


def describe(fn, span=6.0):
    """One row of structural facts about f, measured rather than asserted."""
    aff, lin, _ = image_dimension(fn, span=span)
    lo, hi = lipschitz_bounds(fn, span=span)
    gap = injectivity_gap(fn, span=span)
    # A reduction is only AVAILABLE when the image sits inside a subspace of
    # dimension at most d AND that leaves something over to discard (aff < r).
    # When r == d there is nothing to throw away, whatever shape f has.
    return {"name": fn.name, "d": fn.d, "r": fn.r,
            "affine_dim": aff, "linear_dim": lin,
            "reduces": bool(aff <= fn.d and aff < fn.r),
            "lip_lo": lo, "lip_hi": hi, "inj_gap": gap,
            "claim_linear_image": fn.linear_image,
            "claim_bi_lipschitz": fn.bi_lipschitz,
            "group": fn.group}


# ---------------------------------------------------------------------------
# 4. The projection of Section 5.3.2
# ---------------------------------------------------------------------------


def projection_split(fn, Sigma, span=6.0, n=4000, seed=0, tol=1e-7):
    """The (P_t, zhat, w) decomposition of Section 5.3.2.

    V = { Sigma^{-1/2} f(x) } is spanned numerically, P is the orthogonal
    projection onto it, and the returned `split` maps an observation y to

        zhat = P Sigma^{-1/2} (y - b),   w = (I - P) Sigma^{-1/2} (y - b)

    with b the constant offset, if any. By equation (39),

        G_t(x) = zeta( ||zhat - fhat(x)||^2 + ||w||^2 ),

    and under a Gaussian zeta the ||w||^2 term is a constant that cancels. When
    the image spans all of R^r -- every curved f -- P is the identity, w is
    zero, and nothing can be discarded. That is the whole content of the
    distinction, and it comes out of this function automatically.
    """
    L = np.linalg.cholesky(np.atleast_2d(Sigma))
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-span, span, size=(n, fn.d))
    vals = fn(xs)
    offset = np.zeros(fn.r) if fn.offset is None else np.asarray(fn.offset, float)
    centred = np.linalg.solve(L, (vals - offset).T).T        # Sigma^{-1/2} f(x)
    mu = centred.mean(axis=0)
    U, s, _ = np.linalg.svd((centred - mu).T, full_matrices=True)
    k = int((s / max(s[0], 1e-300) > tol).sum())
    basis = U[:, :k]                                          # r x k
    P = basis @ basis.T

    def split(y):
        y = np.atleast_2d(np.asarray(y, dtype=float))
        u = np.linalg.solve(L, (y - offset).T).T
        u = u - mu
        zhat = u @ P.T
        w = u - zhat
        return zhat, w

    return {"dim": k, "P": P, "basis": basis, "mean": mu, "L": L, "split": split}


# ---------------------------------------------------------------------------
# 5. Simulation
# ---------------------------------------------------------------------------


def simulate_obs_model(fn, pot, Sigma, T=500, b=1.0, Q=1.0, x0=0.0, seed=42):
    """Simulate the state-space model behind the Feynman-Kac representation.

        x_{t+1} = b x_t + v_t,      v_t ~ N(0, Q I_d)
        y_t     = f(x_t) + L e_t,   e_t ~ the law matching `pot`, Sigma = L L'

    b = 1 with d = 1 is the random walk written on the whiteboard, which is the
    spectral radius 1 case: there is no stationary distribution, and the state
    wanders without bound.

    The noise is drawn from the law that MATCHES the potential, so the filter
    below is running the correct model rather than a misspecified one.
    """
    rng = np.random.default_rng(seed)
    d, r = fn.d, fn.r
    L = np.linalg.cholesky(np.atleast_2d(Sigma))
    x = np.zeros((T, d))
    cur = np.full(d, float(x0)) if np.isscalar(x0) else np.asarray(x0, float)
    for t in range(T):
        cur = b * cur + math.sqrt(Q) * rng.standard_normal(d)
        x[t] = cur
    y = fn(x) + pot.sample_noise(T, r, rng) @ L.T
    return x, y


# ---------------------------------------------------------------------------
# 6. Particle filter for a general (f, zeta_G)
# ---------------------------------------------------------------------------


def particle_filter_obs(y, fn, pot, Sigma, b=1.0, Q=1.0, N=1000, x0=0.0,
                        seed=0, ess_threshold=0.5, projected=False,
                        proj=None, stat_fn=None, stat_every=1):
    """Bootstrap particle filter with an arbitrary f and potential shape.

    The cycle is the one used throughout the study -- propagate from the prior,
    weight by the potential, normalise, resample when the ESS falls below
    `ess_threshold * N` -- with the weighting step now reading

        log G_t(x) = log zeta_G( || L^{-1}(y_t - f(x)) ||^2 ),   Sigma = L L'.

    `projected=True` implements Dr Gerber's control: the r-dimensional
    observation is replaced by its d-dimensional projection zhat_t of
    Section 5.3.2, discarding the orthogonal part w_t. By Section 5.3.3 this
    must leave a GAUSSIAN filter completely unchanged -- the discarded term is
    a constant that cancels in the normalisation -- while changing any
    heavy-tailed one. Because the random number stream is untouched, the
    Gaussian case should come back bit-identical, not merely close, which makes
    it a sharp test rather than a judgement call.
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    T, d, r = y.shape[0], fn.d, fn.r
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(np.atleast_2d(Sigma))
    offset = np.zeros(r) if fn.offset is None else np.asarray(fn.offset, float)

    if projected:
        if proj is None:
            proj = projection_split(fn, Sigma)
        basis, mu = proj["basis"], proj["mean"]

    start = np.full(d, float(x0)) if np.isscalar(x0) else np.asarray(x0, float)
    particles = np.tile(start, (N, 1))
    weights = np.full(N, 1.0 / N)

    est = np.zeros((T, d))
    ess = np.zeros(T)
    resampled = np.zeros(T, dtype=bool)
    stats = [] if stat_fn is not None else None

    sq = math.sqrt(Q)
    for t in range(T):
        particles = b * particles + sq * rng.standard_normal((N, d))

        fx = fn(particles)                                   # (N, r)
        if projected:
            # Work inside V: coordinates of the whitened residual in the basis.
            uy = np.linalg.solve(L, (y[t] - offset)) - mu
            ufx = np.linalg.solve(L, (fx - offset).T).T - mu
            resid = (uy @ basis) - (ufx @ basis)             # (N, k)
        else:
            resid = np.linalg.solve(L, (y[t] - fx).T).T      # (N, r)

        with np.errstate(divide="ignore"):
            # log(0) = -inf is the correct answer for a particle that already
            # carries no weight; it exponentiates back to 0 below.
            log_w = pot.log_weight(resid) + np.log(weights)
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        total = w.sum()
        if not np.isfinite(total) or total <= 0.0:
            # Every particle is impossible under this observation. Fall back to
            # a uniform cloud rather than producing NaNs, and let the ESS
            # record that the step was a total loss.
            weights = np.full(N, 1.0 / N)
        else:
            weights = w / total

        est[t] = weights @ particles
        ess[t] = effective_sample_size(weights)

        if stats is not None and t % stat_every == 0:
            stats.append(stat_fn(t, particles, weights))

        if ess[t] < ess_threshold * N:
            idx = systematic_resample(weights, rng)
            particles = particles[idx]
            weights = np.full(N, 1.0 / N)
            resampled[t] = True

    out = [est, ess, resampled]
    if stats is not None:
        out.append(np.asarray(stats))
    return tuple(out)


# ---------------------------------------------------------------------------
# 7. Grid filter: the exact benchmark when there is no Kalman filter
# ---------------------------------------------------------------------------


def _gaussian_kernel(h, Q, cut=8.0):
    m = int(math.ceil(cut * math.sqrt(Q) / h))
    off = np.arange(-m, m + 1) * h
    k = np.exp(-0.5 * off ** 2 / Q) / math.sqrt(2.0 * math.pi * Q) * h
    return m, k


def grid_filter_obs(y, fn, pot, Sigma, b=1.0, Q=1.0, x0=0.0, half_width=None,
                    h=0.02, x_ref=None, return_post=False):
    """Exact filtering distribution for d = 1, computed on a fine grid.

    Once f is nonlinear the model is no longer linear-Gaussian, so the Kalman
    filter is not its exact filter and cannot serve as the benchmark. In one
    dimension the recursion can instead be carried out numerically:

        predict:  p(x_t | y_{1:t-1}) = (K * p(x_{t-1} | y_{1:t-1}))(x_t)
        update:   p(x_t | y_{1:t})  proportional to  G_t(x_t) p(x_t | y_{1:t-1})

    The prediction is a convolution with the N(0, Q) transition density, done
    by FFT so that a fine grid stays affordable over long runs. When b != 1 the
    previous posterior is first pushed through x -> b x by interpolation, which
    is what makes the step shift-invariant and the convolution valid.

    This returns the whole distribution, not just its mean, which matters: for
    a non-injective f such as |x| the exact posterior is bimodal and symmetric,
    so its mean sits near zero and carries no information about the state. Use
    `ks_vs_grid` to score a particle cloud against it.

    Accuracy. Validated against `kalman_filter_md` on the linear-Gaussian cases
    where both apply. At b = 1 the prediction is a pure convolution and the two
    agree to about 1e-13 in the mean. At b != 1 the push-forward x -> b x is
    done by linear interpolation, which costs accuracy: the same comparison
    gives about 7e-6 at h = 0.01. That is still three orders of magnitude below
    any error being measured, but it is an interpolation error rather than
    round-off, so halve h if a case ever needs more.
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    T = y.shape[0]
    if fn.d != 1:
        raise ValueError("grid_filter_obs is for d = 1 only")

    if half_width is None:
        if x_ref is not None:
            half_width = float(np.max(np.abs(x_ref))) + 10.0 * math.sqrt(Q)
        elif abs(b) < 1.0:
            half_width = 8.0 * math.sqrt(Q / (1.0 - b * b)) + 10.0
        else:
            half_width = 6.0 * math.sqrt(Q * max(T, 1)) + 10.0

    G = int(round(2 * half_width / h)) + 1
    grid = np.linspace(-half_width, half_width, G)
    step = grid[1] - grid[0]
    gcol = grid.reshape(-1, 1)

    m, kern = _gaussian_kernel(step, Q)
    nfft = 1 << int(math.ceil(math.log2(G + 2 * m + 1)))
    kern_f = np.fft.rfft(kern, nfft)

    L = np.linalg.cholesky(np.atleast_2d(Sigma))
    fgrid = fn(gcol)                                    # (G, r), f at each node

    mean = np.zeros(T)
    var = np.zeros(T)
    loglik = np.zeros(T)
    post_hist = np.zeros((T, G)) if return_post else None

    running = 0.0
    post = None
    for t in range(T):
        if t == 0:
            pred = np.exp(-0.5 * (grid - b * float(x0)) ** 2 / Q) \
                / math.sqrt(2.0 * math.pi * Q)
        else:
            src = post if b == 1.0 else \
                np.interp(grid / b, grid, post, left=0.0, right=0.0) / abs(b)
            conv = np.fft.irfft(np.fft.rfft(src, nfft) * kern_f, nfft)
            pred = conv[m:m + G] / step        # kern carried a factor `step`

        resid = np.linalg.solve(L, (y[t] - fgrid).T).T
        lg = pot.log_weight(resid)
        lg -= lg.max()
        joint = pred * np.exp(lg)
        mass = joint.sum() * step
        if mass <= 0 or not np.isfinite(mass):
            raise FloatingPointError(
                f"grid filter lost all mass at t={t}; widen half_width "
                f"(currently {half_width:g}) or shrink h")
        post = joint / mass
        running += math.log(mass)

        mean[t] = np.sum(grid * post) * step
        var[t] = np.sum((grid - mean[t]) ** 2 * post) * step
        loglik[t] = running
        if return_post:
            post_hist[t] = post

    if return_post:
        return mean, var, loglik, grid, post_hist
    return mean, var, loglik, grid


# ---------------------------------------------------------------------------
# 8. Scoring a particle cloud against a non-Gaussian exact posterior
# ---------------------------------------------------------------------------


def ks_vs_grid(particles, weights, grid, post):
    """Kolmogorov distance between a weighted cloud and a gridded density.

    The same bounded metric as Part 8, but with the exact posterior supplied as
    a density on a grid rather than as a Gaussian. It is still a difference of
    two probabilities, so it still lies in [0, 1], and -- unlike any comparison
    of posterior means -- it stays meaningful when the exact posterior is
    bimodal, which is precisely the case for the non-injective f in group C.
    """
    x = np.asarray(particles, dtype=float).ravel()
    w = np.asarray(weights, dtype=float)
    step = grid[1] - grid[0]
    cdf = np.cumsum(post) * step
    cdf /= cdf[-1]

    order = np.argsort(x)
    xs = x[order]
    cum = np.cumsum(w[order])
    exact = np.interp(xs, grid, cdf, left=0.0, right=1.0)
    prev = np.empty_like(cum)
    prev[0] = 0.0
    prev[1:] = cum[:-1]
    return float(np.maximum(np.abs(cum - exact), np.abs(prev - exact)).max())
