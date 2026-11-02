# Fit phase stepping corrections to measured motor harmonics
#
# A correction shifts the coil current angle by mag * sin(h * angle + pha)
# for harmonic h, written as the complex number mag * exp(1j * pha). The
# vibration measured at that harmonic responds linearly to it, with a gain
# that depends on the speed through the printer's mechanics, so sweeps made
# at a few corrections determine the response at every speed and one
# correction can be chosen for a whole speed range.
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
# A speed counts as worse than before when it rose by more than this share
# of the earlier level, and by more than twice the noise
REGRESSION_SHARE = 0.2


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
        for sign in (1.0, -1.0, 1j, -1j)
    ]


class HarmonicResponse:
    """How one harmonic of one motor direction responds to its correction.

    add() takes a sweep made at a correction: {speed: phasor} and the
    noise level {speed: displacement}. fit() then finds, per speed, the
    offset r0 and gain g with vibration r(c) = r0 + g * c by weighted least
    squares over all sweeps so far, so that every sweep improves the
    model, and solve() picks the correction for a speed range.
    """

    def __init__(self, speed_min, speed_max, limit):
        self.speed_min, self.speed_max = speed_min, speed_max
        self.limit = limit
        self.sweeps = []
        self.speeds = np.zeros(0)
        self.offset = self.gain = self.noise = np.zeros(0, dtype=complex)

    def add(self, correction, phasors, noises):
        self.sweeps.append((complex(correction), dict(phasors), dict(noises)))

    def measured(self, correction):
        """{speed: displacement} measured at correction in the speed range,
        averaged over the sweeps made at it."""
        levels = {}
        for c, phasors, _ in self.sweeps:
            if abs(c - correction) > 1e-12:
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
                by_speed.setdefault(speed, []).append((c, phasor, noise))
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
        return np.abs(self.offset + self.gain * correction)

    def worst(self, correction):
        return float(np.max(self.predict(correction)))

    def noise_floor(self):
        return float(np.max(self.noise)) if len(self.noise) else 0.0

    def solve(self, objective="minimax", iterations=100):
        """Correction that leaves the least worst predicted vibration over
        the speed range (Lawson's algorithm: least squares, reweighted
        towards the speeds left worst), or with objective "rms" the least
        total, within the correction limit."""
        weights = np.ones(len(self.speeds))
        c = self._weighted(weights)
        if objective == "minimax":
            for _ in range(iterations):
                left = weights * self.predict(c)
                total = np.sum(left)
                if not total:
                    break
                weights = left / total
                c = self._weighted(weights)
        return c

    def _weighted(self, weights):
        den = np.sum(weights * np.abs(self.gain) ** 2)
        if not den:
            return 0j
        c = -np.sum(weights * np.conj(self.gain) * self.offset) / den
        if abs(c) > self.limit:
            c *= self.limit / abs(c)
        return complex(c)

    def improves(self, candidate, applied):
        """Whether candidate should lower the worst vibration enough below
        what applied is predicted to leave to be worth another sweep: by
        MIN_IMPROVEMENT, and by more than noise can account for."""
        if abs(candidate - applied) < 1e-9:
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
