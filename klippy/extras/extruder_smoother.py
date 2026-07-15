# Extruder smoothers to synchronize extruder pressure advance with input shaping
#
# Copyright (C) 2023-2024  Dmitry Butyugin <dmbutyugin@google.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.

import collections
import importlib
import math

from . import shaper_calibrate, shaper_defs

ExtruderSmootherCfg = collections.namedtuple(
    "ExtruderSmootherCfg", ("order", "freq_opt_range")
)

EXTRUDER_SMOOTHERS = {
    "default": ExtruderSmootherCfg(-1, (1.0, 1.0, 1)),
    "zv": ExtruderSmootherCfg(5, (0.98, 1.02, 5)),
    "mzv": ExtruderSmootherCfg(7, (0.95, 1.05, 11)),
    "zvd": ExtruderSmootherCfg(7, (0.93, 1.06, 14)),
    "ei": ExtruderSmootherCfg(7, (0.83, 0.89, 7)),
    "2hump_ei": ExtruderSmootherCfg(9, (0.65, 0.75, 11)),
    "3hump_ei": ExtruderSmootherCfg(10, (0.54, 0.66, 13)),
    "smooth_zv": ExtruderSmootherCfg(7, (0.98, 1.0, 3)),
    "smooth_mzv": ExtruderSmootherCfg(9, (0.95, 1.07, 20)),
    "smooth_ei": ExtruderSmootherCfg(9, (0.97, 1.07, 15)),
    "smooth_zvd_ei": ExtruderSmootherCfg(11, (0.90, 1.10, 30)),
    "smooth_2hump_ei": ExtruderSmootherCfg(11, (0.95, 1.07, 20)),
    "smooth_si": ExtruderSmootherCfg(11, (0.95, 1.07, 20)),
}


def _estimate_shaper(np, shaper, test_damping_ratio, test_freqs):
    A, T = np.asarray(shaper[0]), np.asarray(shaper[1])
    inv_D = 1.0 / A.sum()
    n = len(T)
    t_s = T[-1] - T[0]
    hst = t_s * 0.5

    test_freqs = np.asarray(test_freqs)
    n_t = 1000
    time = np.linspace(-hst, hst, n_t)

    omega = 2.0 * math.pi * test_freqs[test_freqs > 0.0]

    velocity = np.zeros(shape=(omega.shape[0], time.shape[-1]))
    for i in range(n):
        velocity += A[i] * shaper_calibrate.step_response_velocity(
            np, time - T[i] + hst, omega, test_damping_ratio
        )
    velocity *= inv_D
    return time, velocity


def _estimate_smoother(np, smoother, test_damping_ratio, test_freqs):
    C, t_sm = smoother[0], smoother[1]
    hst = t_sm * 0.5

    test_freqs = np.asarray(test_freqs)
    n_t = 1000
    time = np.linspace(-t_sm, t_sm, n_t)
    dt = time[1] - time[0]

    w_ind = (time >= -hst) & (time < hst)
    tau = time[w_ind]
    w = np.zeros(shape=tau.shape)
    for c in C[::-1]:
        w = w * tau + c
    w_dt = w * dt / (w * dt).sum()
    wl = tau.shape[0]

    omega = 2.0 * math.pi * test_freqs[test_freqs > 0.0]

    def get_windows(m, wl):
        nrows = m.shape[-1] - wl + 1
        n = m.strides[-1]
        return np.lib.stride_tricks.as_strided(
            m, shape=(m.shape[0], nrows, wl), strides=(m.strides[0], n, n)
        )

    s_v = shaper_calibrate.step_response_velocity(
        np, time, omega, test_damping_ratio
    )
    velocity = np.einsum("ijk,k->ij", get_windows(s_v, wl), w_dt[::-1])
    nrows = velocity.shape[-1]
    # Window starting at time[j] convolves the transient around T = time[j]+hst
    return time[:nrows] + hst, velocity


def _calc_extruder_smoother(np, shaper_name, t, velocities, n, t_sm):
    zero_derivatives = shaper_name.startswith("smooth_")
    if n <= 3:
        return [1.5, 0, -6.0]
    if n <= 5 and zero_derivatives:
        return [15.0 / 8.0, 0.0, -15.0, 0.0, 30.0]

    # Fit h(tau) = w(t) * t_sm with tau = t / t_sm to the normalized velocity
    # profiles. A Legendre basis in x = 2 * tau keeps the system
    # well-conditioned, unlike monomials in tau.
    x = 2.0 * t / t_sm
    weight = np.maximum(1.0 - x * x, 0.0)
    target = (
        t_sm
        * velocities
        / (velocities.sum(axis=-1)[:, np.newaxis] * (t[1] - t[0]))
    ).mean(axis=0)

    legendre = np.polynomial.legendre
    P = legendre.legvander(x, n - 1)
    PtW = P.T * weight
    G = np.matmul(PtW, P)
    g = np.matmul(PtW, target)

    # Equality constraints on the Legendre coefficients a_k:
    # *) integral(h(tau) dtau, tau=[-1/2...1/2]) = 1  <=>  a_0 = 1
    # *) h(+-1/2) = 0, using P_k(+-1) = (+-1)^k
    # *) optionally h'(+-1/2) = 0, using P_k'(+-1) = (+-1)^(k+1) * k*(k+1)/2
    k = np.arange(n)
    sign = np.power(-1.0, k)
    d_p1 = 0.5 * k * (k + 1)
    constraints = [np.eye(n)[0], sign, np.ones(n)]
    rhs = [1.0, 0.0, 0.0]
    if zero_derivatives:
        constraints += [-sign * d_p1, d_p1]
        rhs += [0.0, 0.0]
    elif shaper_name == "3hump_ei":
        constraints.append(d_p1)
        rhs.append(0.0)
    E = np.array(constraints)
    n_c = E.shape[0]

    # Solve the equality-constrained least squares via its KKT system,
    # penalizing negative values of h(tau) toward zero. Strict h(tau) >= 0
    # is not attainable at these polynomial orders, and pushing the penalty
    # harder degrades both the fit and the conditioning.
    K = np.zeros(shape=(n + n_c, n + n_c))
    K[:n, n:] = E.T
    K[n:, :n] = E
    f = np.concatenate([g, rhs])
    penalty = np.zeros(shape=P.shape[0])
    for _ in range(20):
        K[:n, :n] = G + np.matmul(P.T * penalty, P)
        a = np.linalg.solve(K, f)[:n]
        h = np.matmul(P, a)
        if h.min() > -2e-3:
            break
        penalty[h < 0.0] += 10.0

    # Convert Legendre coefficients in x = 2 * tau to monomials in tau
    C = legendre.leg2poly(a) * np.power(2.0, k)
    return C


def get_extruder_smoother(
    shaper_name,
    smooth_time,
    damping_ratio,
    normalize_coeffs=True,
    return_velocities=False,
):
    try:
        np = importlib.import_module("numpy")
    except ImportError:
        raise Exception(
            "Failed to import `numpy` module, make sure it was "
            "installed via `~/klippy-env/bin/pip install` (refer to "
            "docs/Measuring_Resonances.md for more details)."
        )
    shaper_name = shaper_name.lower()
    smoother_cfg = EXTRUDER_SMOOTHERS.get(
        shaper_name, EXTRUDER_SMOOTHERS["default"]
    )
    test_freqs = np.linspace(*smoother_cfg.freq_opt_range)
    n = smoother_cfg.order
    for s in shaper_defs.INPUT_SHAPERS:
        if s.name == shaper_name:
            A, T = s.init_func(1.0, damping_ratio)
            if n < 0:
                n = 2 * len(A) + 1
            t_sm = T[-1] - T[0]
            shaper = A, T
            t, velocities = _estimate_shaper(
                np, shaper, damping_ratio, test_freqs
            )
            break
    for s in shaper_defs.INPUT_SMOOTHERS:
        if s.name == shaper_name:
            C, t_sm = s.init_func(1.0)
            if n < 0:
                n = len(C)
            smoother = C, t_sm
            t, velocities = _estimate_smoother(
                np, smoother, damping_ratio, test_freqs
            )
            break
    C_e = _calc_extruder_smoother(np, shaper_name, t, velocities, n, t_sm)
    smoother = shaper_defs.init_smoother(
        C_e[::-1], smooth_time, normalize_coeffs
    )
    if not return_velocities:
        return smoother
    return smoother, (t, velocities)


def get_multi_mode_extruder_smoother(
    base_names,
    freqs,
    damping_ratios,
    smooth_time,
    normalize_coeffs=True,
):
    # Generalization of get_two_mode_extruder_smoother to N >= 2 peaks:
    # every peak contributes its own row of target velocities (evaluated
    # with its own base shaper's frequency-optimization range and its own
    # damping ratio), stacked together into one fit -- the same idea as
    # the two-mode case, just looped over N instead of hardcoded to 2.
    try:
        np = importlib.import_module("numpy")
    except ImportError:
        raise Exception(
            "Failed to import `numpy` module, make sure it was "
            "installed via `~/klippy-env/bin/pip install` (refer to "
            "docs/Measuring_Resonances.md for more details)."
        )
    base_names = [b.lower() for b in base_names]
    cfgs = [
        EXTRUDER_SMOOTHERS.get(b, EXTRUDER_SMOOTHERS["default"])
        for b in base_names
    ]
    # A single shaper_freq can be normalized away (a shaper's relative
    # impulse structure is scale-invariant), but a multi-mode shaper's
    # structure depends on the frequency *ratios*, not just an overall
    # scale. Fit in a ratio-preserving unit system (freqs[0] -> 1.0, the
    # rest scaled by freqs[i] / freqs[0]) and let init_smoother rescale
    # the result to the real smooth_time below, exactly as the
    # single-mode path does.
    ratios = [f / freqs[0] for f in freqs]
    A, T = shaper_defs.get_multi_mode_shaper(
        base_names, ratios, damping_ratios
    )
    # The order must accommodate whichever base needs the most terms; with
    # exactly 2 matching bases this is exactly the previous lookup.
    n = max(cfg.order for cfg in cfgs)
    if n < 0:
        n = 2 * len(A) + 1
    shaper = A, T
    # Evaluate the shaped velocity near each peak with that peak's own
    # damping ratio; every call shares the same shaper (and hence the same
    # time grid), so the resulting rows can be stacked directly.
    t = None
    velocity_rows = []
    for cfg, ratio, damping_ratio in zip(cfgs, ratios, damping_ratios):
        lo, hi, cnt = cfg.freq_opt_range
        test_freqs = ratio * np.linspace(lo, hi, cnt)
        t, velocities = _estimate_shaper(np, shaper, damping_ratio, test_freqs)
        velocity_rows.append(velocities)
    velocities = np.concatenate(velocity_rows, axis=0)
    t_sm_unit = T[-1] - T[0]
    # "2mode" matches neither a smooth_* name nor "3hump_ei", so
    # _calc_extruder_smoother uses its generic constraint set: those
    # extra constraints were tuned for single-peak targets and don't
    # carry over to a multi-peak fit.
    C_e = _calc_extruder_smoother(np, "2mode", t, velocities, n, t_sm_unit)
    return shaper_defs.init_smoother(C_e[::-1], smooth_time, normalize_coeffs)


def get_two_mode_extruder_smoother(
    base_name,
    freq1,
    freq2,
    damping_ratio1,
    damping_ratio2,
    smooth_time,
    normalize_coeffs=True,
    base_name2=None,
):
    return get_multi_mode_extruder_smoother(
        [base_name, base_name2 or base_name],
        [freq1, freq2],
        [damping_ratio1, damping_ratio2],
        smooth_time,
        normalize_coeffs=normalize_coeffs,
    )
