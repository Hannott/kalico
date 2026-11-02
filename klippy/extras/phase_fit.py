# Fit phase stepping corrections to measured motor harmonics
#
# A correction shifts the coil current angle by mag * sin(h * angle + pha)
# for harmonic h, written as the complex number mag * exp(1j * pha). The
# vibration measured at that harmonic responds linearly to it, with a gain
# that depends on the speed through the printer's mechanics, so sweeps made
# at a few corrections determine the response at every speed and one
# correction can be chosen for a whole speed range. A correction is a
# (slow, fast) pair: its values at the two ends of the speed range, between
# which the micro-controller follows the speed linearly; a correction that
# is the same at every speed has equal ends.
#
# Copyright (C) 2026  Åsmund Collin <aakjaergaard@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import math

import numpy as np

PROBE_MIN = 0.02
PROBE_MAX = 0.06
# Fewest sweeps a speed must appear in to fit its offset and gain
MIN_POINTS = 2
# A step is worth a sweep when it should lower the worst vibration this much
MIN_IMPROVEMENT = 0.05
# A correction that changes with speed is used when it should lower the
# worst vibration this much below the best constant one
SPEED_DEPENDENT_GAIN = 0.1
# Weight, relative to the measurements, that pulls the two ends of a
# correction together where the measurements leave them free
SLOPE_RIDGE = 1e-3
# A speed counts as worse than before when it rose by more than this share
# of the earlier level, and by more than twice the noise
REGRESSION_SHARE = 0.2


def as_pair(correction):
    """A correction as its (slow, fast) ends."""
    if isinstance(correction, tuple):
        return correction
    return (complex(correction), complex(correction))


def probe_size(phasors, period, limit):
    """Probe correction magnitude (radians) for a harmonic from its
    uncorrected displacement phasors (mm): about the correction its largest
    displacement implies, at least PROBE_MIN so that a small harmonic still
    answers clearly above the noise, and at most PROBE_MAX, or limit, to
    stay in the linear range."""
    largest = max((abs(p) for p in phasors.values()), default=0.0)
    implied = 2.0 * math.pi * largest / period
    return min(limit, PROBE_MAX, max(PROBE_MIN, implied))


def probe_corrections(probes, limit):
    """The four probe settings {key: correction} for probes {key: size}:
    both signs of a real and of an imaginary step, so that the differences
    cancel every even order of the response. Probes are scaled down
    together when their sum would not fit in the correction table."""
    total = sum(probes.values())
    scale = min(1.0, limit / total) if total else 1.0
    return [
        {key: sign * size * scale for key, size in probes.items()}
        for sign in (1 + 0j, -1 + 0j, 1j, -1j)
    ]


class HarmonicResponse:
    """How one harmonic of one motor direction responds to its correction.

    add() takes a sweep made at a correction: {speed: phasor} and the
    noise level {speed: displacement}. fit() then finds, per speed, the
    offset r0 and gain g with vibration r(c) = r0 + g * c by weighted least
    squares over all sweeps so far, so that every sweep improves the
    model, and solve() picks the correction for the speed range.
    """

    def __init__(self, speed_min, speed_max, limit):
        self.speed_min, self.speed_max = speed_min, speed_max
        self.limit = limit
        self.sweeps = []
        self.speeds = np.zeros(0)
        self.offset = self.gain = self.noise = np.zeros(0, dtype=complex)

    def fraction(self, speeds):
        """Where speeds lie between the slow (0) and fast (1) end."""
        span = self.speed_max - self.speed_min
        if span <= 0.0:
            return np.zeros(np.shape(speeds))
        return np.clip((np.asarray(speeds) - self.speed_min) / span, 0.0, 1.0)

    def value(self, correction, speeds):
        """The correction in effect at speeds."""
        slow, fast = as_pair(correction)
        return slow + (fast - slow) * self.fraction(speeds)

    def add(self, correction, phasors, noises):
        self.sweeps.append((as_pair(correction), dict(phasors), dict(noises)))

    def measured(self, correction):
        """{speed: displacement} measured at correction in the speed range,
        averaged over the sweeps made at it."""
        wanted = as_pair(correction)
        levels = {}
        for c, phasors, _ in self.sweeps:
            if abs(c[0] - wanted[0]) > 1e-12 or abs(c[1] - wanted[1]) > 1e-12:
                continue
            for speed, phasor in phasors.items():
                if self.speed_min <= speed <= self.speed_max:
                    levels.setdefault(speed, []).append(abs(phasor))
        return {s: sum(v) / len(v) for s, v in levels.items()}

    def fit(self):
        """Fit offset and gain per speed; returns the number of speeds."""
        by_speed = {}
        for c, phasors, noises in self.sweeps:
            for speed, phasor in phasors.items():
                if not self.speed_min <= speed <= self.speed_max:
                    continue
                noise = noises.get(speed) or 0.0
                value = complex(self.value(c, speed))
                by_speed.setdefault(speed, []).append((value, phasor, noise))
        speeds, offsets, gains, noise = [], [], [], []
        for speed in sorted(by_speed):
            points = by_speed[speed]
            if len({c for c, _, _ in points}) < MIN_POINTS:
                continue
            level = max(n for _, _, n in points)
            # A noise level of zero (a test without one) weighs all alike
            weights = np.array(
                [1.0 / max(n, 1e-3 * level, 1e-12) for _, _, n in points]
            )
            design = (
                np.array([[1.0, c] for c, _, _ in points]) * weights[:, None]
            )
            target = np.array([p for _, p, _ in points]) * weights
            r0, g = np.linalg.lstsq(design, target, rcond=None)[0]
            speeds.append(speed)
            offsets.append(r0)
            gains.append(g)
            noise.append(level)
        self.speeds = np.array(speeds)
        self.offset = np.array(offsets, dtype=complex)
        self.gain = np.array(gains, dtype=complex)
        self.noise = np.array(noise)
        return len(speeds)

    def predict(self, correction):
        """Predicted displacement per speed at correction."""
        return np.abs(
            self.offset + self.gain * self.value(correction, self.speeds)
        )

    def worst(self, correction):
        return float(np.max(self.predict(correction)))

    def noise_floor(self):
        return float(np.max(self.noise)) if len(self.noise) else 0.0

    def solve(self, objective="minimax", speed_dependent=True):
        """The (slow, fast) correction that leaves the least worst predicted
        vibration over the speed range, or with objective "rms" the least
        total. A correction that changes with speed is returned only when
        it beats the best constant one by SPEED_DEPENDENT_GAIN and by more
        than noise can account for."""
        constant = self._solve(objective, False)
        if not speed_dependent:
            return constant
        varying = self._solve(objective, True)
        before = self.worst(constant)
        gain = before - self.worst(varying)
        if gain > max(SPEED_DEPENDENT_GAIN * before, 0.5 * self.noise_floor()):
            return varying
        return constant

    def _solve(self, objective, varying, iterations=100):
        """Lawson's algorithm: least squares, reweighted towards the speeds
        left worst, within the correction limit."""
        weights = np.ones(len(self.speeds))
        c = self._weighted(weights, varying)
        if objective == "minimax":
            for _ in range(iterations):
                left = weights * self.predict(c)
                total = np.sum(left)
                if not total:
                    break
                weights = left / total
                c = self._weighted(weights, varying)
        return c

    def _weighted(self, weights, varying):
        if not len(self.speeds):
            return (0j, 0j)
        if not varying:
            den = np.sum(weights * np.abs(self.gain) ** 2)
            if not den:
                return (0j, 0j)
            c = -np.sum(weights * np.conj(self.gain) * self.offset) / den
            return self._limited((complex(c), complex(c)))
        root = np.sqrt(weights)
        u = self.fraction(self.speeds)
        design = np.column_stack(
            [self.gain * (1.0 - u) * root, self.gain * u * root]
        )
        target = -self.offset * root
        # Where the measurements leave an end free (speeds not reaching it),
        # a small pull towards equal ends keeps it from running away
        ridge = math.sqrt(
            SLOPE_RIDGE * np.sum(weights * np.abs(self.gain) ** 2)
        )
        design = np.vstack([design, [[ridge, -ridge]]])
        target = np.append(target, 0j)
        slow, fast = np.linalg.lstsq(design, target, rcond=None)[0]
        return self._limited((complex(slow), complex(fast)))

    def _limited(self, pair):
        largest = max(abs(pair[0]), abs(pair[1]))
        if largest > self.limit:
            scale = self.limit / largest
            return (pair[0] * scale, pair[1] * scale)
        return pair

    def improves(self, candidate, applied):
        """Whether candidate should lower the worst vibration enough below
        what applied is predicted to leave to be worth another sweep: by
        MIN_IMPROVEMENT, and by more than noise can account for."""
        candidate, applied = as_pair(candidate), as_pair(applied)
        if max(abs(candidate[i] - applied[i]) for i in (0, 1)) < 1e-9:
            return False
        before = self.worst(applied)
        gain = before - self.worst(candidate)
        return gain > max(MIN_IMPROVEMENT * before, 0.5 * self.noise_floor())


def compare(before, after, noise):
    """Summarise {speed: displacement} after a calibration against before:
    (worst before, worst after, average before, average after, speeds that
    got worse). noise maps speed to the noise level."""
    speeds = sorted(set(before) & set(after))
    worse = []
    for speed in speeds:
        margin = max(
            2.0 * (noise.get(speed) or 0.0), REGRESSION_SHARE * before[speed]
        )
        if after[speed] > before[speed] + margin:
            worse.append(speed)
    b = [before[s] for s in speeds]
    a = [after[s] for s in speeds]
    if not speeds:
        return 0.0, 0.0, 0.0, 0.0, worse
    return max(b), max(a), sum(b) / len(b), sum(a) / len(a), worse
