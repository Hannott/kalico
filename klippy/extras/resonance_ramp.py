# Resonance-weighted jerk-limited ramp construction.
#
# A jerk-limited acceleration ramp that rises from 0 to a peak and back down
# at constant jerk is, in continuous time, a triangle (or, with a plateau, a
# trapezoid) in a(t). That shape's Fourier magnitude has an EXACT zero at
# 1/T_rise (and, for the trapezoid, a second exact zero at 1/T_plateau_edge).
# Kalico's manual jerk/notch config picks ONE (or two) such frequencies by
# hand. This module instead picks them automatically: given a small
# reconstructed vibration-response curve from an actual resonance test (see
# resonance_model.py for how that curve is measured and persisted), it
# searches the SAME small family of known-safe single/dual-zero ramp shapes
# for whichever (f_lo, f_hi) pair leaves the least weighted vibration energy
# -- never an open-ended continuous optimization over arbitrary
# accelerations, which is what made an earlier N-exact-zero attempt in this
# project numerically unpredictable (some frequency combinations solved
# perfectly, others left several percent residual, with no way to tell which
# in advance). Every candidate this module considers is a shape whose
# acceleration is bounded by construction (peak accel, jerk, and duration
# all follow directly from the same well-known ramp physics), so the choice
# only ever affects HOW WELL a print is shaped, never whether the motion
# stays within configured limits.
#
# Pure (only `math`) and decoupled from klippy/toolhead so it can be
# unit-tested in isolation.
#
# Copyright (C) 2026
# This file may be distributed under the terms of the GNU GPLv3 license.
import math


def _peak_response(f, f0, damping, weight):
    # Normalized magnitude of a lightly damped 2nd-order resonance at f0
    # (damping ratio `damping`), scaled by `weight` -- the peak's measured
    # PSD magnitude relative to the tallest peak on that axis. Standard
    # damped-harmonic-oscillator response:
    #   |H(f)| = 1 / sqrt((1 - (f/f0)^2)^2 + (2*damping*f/f0)^2)
    if f0 <= 0.0:
        return 0.0
    r = f / f0
    denom = (1.0 - r * r) ** 2 + (2.0 * damping * r) ** 2
    denom = max(denom, 1e-12)
    return weight / math.sqrt(denom)


class PeakModel:
    """A handful of (freq, damping, weight) peaks reconstructing an
    approximate vibration-response curve without keeping any raw measured
    PSD samples -- see resonance_model.py for how these are extracted (via
    shaper_calibrate's own peak detection, which already discards anything
    below a prominence threshold) and persisted like a bed_mesh profile.
    """

    def __init__(self, peaks):
        # peaks: iterable of (freq_hz, damping_ratio, weight). weight is
        # relative (e.g. normalized to the tallest peak = 1.0), so no
        # absolute PSD units ever reach the planner.
        self.peaks = [
            (float(f), max(float(d), 1e-3), float(w))
            for f, d, w in peaks
            if f and f > 0.0 and w and w > 0.0
        ]

    def is_empty(self):
        return not self.peaks

    def weight(self, f):
        """Approximate relative vibration magnitude at frequency f (Hz)."""
        if not self.peaks:
            return 0.0
        return math.fsum(_peak_response(f, f0, d, w) for f0, d, w in self.peaks)

    def dominant_freq(self):
        if not self.peaks:
            return 0.0
        return max(self.peaks, key=lambda p: p[2])[0]

    def freqs(self, min_freq=0.0):
        return sorted({f for f, _d, _w in self.peaks if f >= min_freq})


def _sinc(x):
    # sin(pi*x)/(pi*x); the normalized sinc, zeros on the nonzero integers.
    if abs(x) < 1e-12:
        return 1.0
    y = math.pi * x
    return math.sin(y) / y


def ramp_spectrum(f, f_lo, f_hi):
    """Normalized magnitude spectrum of the ideal (continuous-time)
    single/dual-zero jerk-limited ramp shape (f_lo == f_hi is the
    single-zero triangle). Scale-invariant in dv -- only the SHAPE matters
    for scoring candidate (f_lo, f_hi) pairs against a vibration curve; the
    move's actual dv only scales the whole spectrum uniformly and so drops
    out of any comparison between candidates.
    """
    if f_lo <= 0.0 or f_hi <= 0.0:
        return 0.0
    return abs(_sinc(f / f_hi) * _sinc(f / f_lo))


def score_pair(f_lo, f_hi, peak_model, freq_grid):
    """Weighted spectral energy this ramp shape would leave at the
    frequencies in `freq_grid`, weighted by `peak_model`. Lower is better.
    """
    return math.fsum(
        peak_model.weight(f) * ramp_spectrum(f, f_lo, f_hi) ** 2
        for f in freq_grid
    )


def candidate_freq_grid(peak_model, n=6):
    """Sample points bracketing each peak, dense enough that a candidate
    shape which nulls exactly ON a peak but leaves its immediate skirt
    excited still scores accordingly. Cheap: evaluated once per search, not
    once per candidate.
    """
    grid = []
    for f0, damping, _w in peak_model.peaks:
        span = max(damping * f0 * 3.0, 1.0)
        for i in range(-n, n + 1):
            f = f0 + span * i / n
            if f > 0.5:
                grid.append(f)
    return grid or [55.0]


def best_notch_pair(peak_model, min_freq=5.0):
    """Search the small, SAFE family of single/dual-zero ramp shapes built
    from this model's own peak frequencies for the (f_lo, f_hi) pair that
    leaves the least weighted vibration energy.

    The candidate set is every peak alone (single-zero) and every pair of
    peaks (dual-zero) -- never a continuous search over arbitrary
    frequencies, and never more than two zeros in one ramp (there is no
    closed-form jerk-limited shape for a third; see the module docstring).
    This only automates the CHOICE of frequency using the measured curve
    instead of a human picking one peak by hand or trusting "the dominant
    one is close enough".

    Returns (f_lo, f_hi), or None if the model has no peak at or above
    min_freq (a ramp notched below roughly 5 Hz stalls the toolhead rather
    than shaping it -- same floor as the manually configured notch law).
    """
    peaks = peak_model.freqs(min_freq)
    if not peaks:
        return None
    grid = candidate_freq_grid(peak_model)
    candidates = [(f, f) for f in peaks]
    for i, fa in enumerate(peaks):
        for fb in peaks[i + 1 :]:
            candidates.append((min(fa, fb), max(fa, fb)))
    best = None
    best_score = None
    for f_lo, f_hi in candidates:
        s = score_pair(f_lo, f_hi, peak_model, grid)
        if best_score is None or s < best_score:
            best, best_score = (f_lo, f_hi), s
    return best


def ramp_jerk(dv, f_lo, f_hi, max_accel=None):
    """Per-ramp jerk (mm/s^3) for the ideal single/dual-zero shape.

    Unsaturated (a_peak = dv*f_lo <= max_accel): J = dv*f_lo*f_hi, giving
    rise time 1/f_hi and peak accel dv*f_lo -- f_lo == f_hi collapses this
    to the single-zero triangle (J = dv*f_n^2, a_peak = dv*f_n).

    Past max_accel, the plateau is forced wider than 1/f_lo and only the
    rise edge is free; J = max_accel*f_hi keeps that edge's zero parked at
    f_hi while the plateau supplies the rest of the speed change at
    max_accel.
    """
    dv = abs(dv)
    if dv <= 1e-12 or f_lo <= 0.0 or f_hi <= 0.0:
        return None
    if max_accel and max_accel > 0.0 and dv > max_accel / f_lo:
        j = max_accel * f_hi
    else:
        j = dv * f_lo * f_hi
    return j if j > 0.0 else None


def notch_duration(dv, f_lo, f_hi, max_accel=None):
    """Total ramp duration (s) for a |dv| speed change under the ideal
    shape -- 1/f_lo + 1/f_hi unsaturated, or dv/max_accel + 1/f_hi past
    saturation (the widened plateau plus the one edge still free)."""
    dv = abs(dv)
    if dv <= 1e-12:
        return 0.0
    if max_accel and max_accel > 0.0 and dv > max_accel / f_lo:
        return dv / max_accel + 1.0 / f_hi
    return 1.0 / f_lo + 1.0 / f_hi


def notch_dist(v0, v1, f_lo, f_hi, max_accel=None):
    """Path distance to change speed v0 -> v1 under the ideal shape.
    Closed form: distance = mean speed * duration, exact because a
    symmetric velocity S-curve's time-average speed is (v0+v1)/2."""
    dv = abs(v1 - v0)
    if dv <= 1e-12:
        return 0.0
    return 0.5 * (v0 + v1) * notch_duration(dv, f_lo, f_hi, max_accel)


def _ramp_up(v0, v1, f_lo, f_hi, max_accel, jerk_dt, max_slices):
    # Jerk-limited acceleration ramp v0 -> v1 (v1 >= v0), integrated at
    # fixed jerk_dt: acceleration rises from ~0 toward a_peak in steps
    # bounded by J*dt, then falls back toward ~0 landing on v1. The "brake"
    # cap a_brake = sqrt(2*J*(v1-v)) has da/dt = -J exactly along it, so
    # following min(a_peak, a_brake, a+J*dt) guarantees both a bounded rise
    # and a jerk-feasible fall that lands exactly on v1 (the final slice's
    # duration is truncated to hit it, so velocity continuity is exact
    # rather than approximate).
    dv = v1 - v0
    if dv <= 1e-12:
        return [], True
    j = ramp_jerk(dv, f_lo, f_hi, max_accel)
    if j is None or j <= 0.0:
        return [], False
    a_peak = dv * f_lo
    if max_accel and max_accel > 0.0:
        a_peak = min(a_peak, max_accel)
    if a_peak <= 0.0:
        return [], False
    segs = []
    v = v0
    a = 0.0
    guard = 0
    while v < v1 - 1e-9:
        guard += 1
        if guard > max_slices:
            return [], False
        rem = v1 - v
        a_brake = math.sqrt(2.0 * j * rem)
        a_new = min(a_peak, a_brake, a + j * jerk_dt)
        if a_new <= 0.0:
            return [], False
        v_next = v + a_new * jerk_dt
        this_dt = jerk_dt
        if v_next >= v1:
            v_next = v1
            this_dt = rem / a_new
        dist = 0.5 * (v + v_next) * this_dt
        segs.append((this_dt, 0.0, 0.0, v, v_next, a_new, dist))
        v, a = v_next, a_new
    return segs, True


def _decel_from_accel(acc_slices):
    # Time-reverse an increasing-velocity ramp into a decel slice list: an
    # accel slice (dt,0,0, v_lo,v_hi, a, dist) becomes decel
    # (0,0,dt, v_hi,v_hi, a, dist), reversed order so the chain still runs
    # from the higher speed down to the lower one.
    dec = []
    for at, _ct, _dt, sv, cv, a, dist in reversed(acc_slices):
        dec.append((0.0, 0.0, at, cv, cv, a, dist))
    return dec


def build_notch_profile(
    vs, vc, ve, move_d, f_lo, f_hi, max_accel, jerk_dt, max_slices=4096
):
    """Emit a jerk-shaped profile for a move whose boundary speeds
    (vs, vc, ve) and length (move_d) were already planned by the stock
    lookahead -- this module does not re-plan them. Returns a list of
    (accel_t, cruise_t, decel_t, start_v, cruise_v, accel, dist) segments,
    or None when the shaped ramp would need more distance than move_d
    actually has (or hit max_slices): the caller must fall back to the
    stock hard-trapezoid emission for that one move rather than try to
    re-derive different boundary speeds, which is the toolhead lookahead's
    job, not this module's.
    """
    acc, ok = _ramp_up(vs, vc, f_lo, f_hi, max_accel, jerk_dt, max_slices)
    if not ok:
        return None
    dec, ok = _ramp_up(ve, vc, f_lo, f_hi, max_accel, jerk_dt, max_slices)
    if not ok:
        return None
    d_acc = math.fsum(s[6] for s in acc)
    d_dec = math.fsum(s[6] for s in dec)
    cruise_d = move_d - d_acc - d_dec
    if cruise_d < -1e-6 * max(1.0, move_d):
        return None
    cruise_d = max(0.0, cruise_d)
    segs = list(acc)
    if cruise_d > 1e-9 and vc > 1e-9:
        segs.append((0.0, cruise_d / vc, 0.0, vc, vc, 0.0, cruise_d))
    segs.extend(_decel_from_accel(dec))
    return segs
