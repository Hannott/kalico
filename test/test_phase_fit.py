# Tests for phase_fit.py: the response of a harmonic to its correction and
# the correction chosen for a speed range.
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "klippy", "extras"))
import motor_harmonics as mh  # noqa: E402
import phase_fit as pf  # noqa: E402

PERIOD = 0.8
LIMIT = 127 * 2.0 * math.pi / 4096.0
SPEEDS = [float(s) for s in range(40, 101, 5)]


def test_probe_size_follows_the_uncorrected_level():
    small = {s: 2e-5 * np.exp(1j * s) for s in SPEEDS}
    assert pf.probe_size(small, PERIOD, LIMIT) == pf.PROBE_MIN
    medium = {s: 5e-3 * np.exp(1j * s) for s in SPEEDS}
    assert pf.probe_size(medium, PERIOD, LIMIT) == pytest.approx(
        2.0 * math.pi * 5e-3 / PERIOD
    )
    large = {s: 2e-2 for s in SPEEDS}
    assert pf.probe_size(large, PERIOD, LIMIT) == pf.PROBE_MAX
    assert pf.probe_size(large, PERIOD, 0.03) == 0.03
    assert pf.probe_size({}, PERIOD, LIMIT) == pf.PROBE_MIN


def test_probe_corrections_fit_the_table():
    probes = {"a": 0.05, "b": 0.03}
    settings = pf.probe_corrections(probes, LIMIT)
    assert [s["a"] for s in settings] == [0.05, -0.05, 0.05j, -0.05j]
    assert [s["b"] for s in settings] == [0.03, -0.03, 0.03j, -0.03j]
    # Four large probes are scaled down together
    probes = {k: 0.06 for k in "abcd"}
    settings = pf.probe_corrections(probes, LIMIT)
    assert sum(abs(c) for c in settings[0].values()) == pytest.approx(LIMIT)
    assert settings[0]["a"] == settings[0]["b"]
    assert pf.probe_corrections({}, LIMIT) == [{}, {}, {}, {}]


def test_pairs_and_constants_are_the_same_correction():
    model = pf.HarmonicResponse(40.0, 100.0, LIMIT)
    assert pf.as_pair(0.01j) == (0.01j, 0.01j)
    assert pf.as_pair((0.01j, 0.02)) == (0.01j, 0.02)
    assert model.value((0j, 0.02), 70.0) == pytest.approx(0.01)
    assert model.value((0j, 0.02), 30.0) == 0.0
    assert model.value((0j, 0.02), 120.0) == pytest.approx(0.02)
    assert model.value(0.01j, 70.0) == 0.01j
    assert list(model.fraction([40.0, 55.0, 100.0])) == [0.0, 0.25, 1.0]
    assert list(pf.HarmonicResponse(50.0, 50.0, LIMIT).fraction([50.0])) == [
        0.0
    ]


def _linear_motor(targets, gains, noise=0.0, seed=1):
    """A sweep function: vibration gain * (c - target) per speed, plus
    noise, for a correction c, which may be a (slow, fast) pair."""
    rng = np.random.default_rng(seed)
    model = pf.HarmonicResponse(min(targets), max(targets), LIMIT)

    def sweep(c):
        phasors = {
            s: gains[s] * (complex(model.value(c, s)) - targets[s])
            + noise * (rng.normal() + 1j * rng.normal())
            for s in targets
        }
        return phasors, dict.fromkeys(targets, noise)

    return sweep


def _calibrate(sweep, iterations=2, speed_dependent=True):
    model = pf.HarmonicResponse(40.0, 100.0, LIMIT)
    phasors, noises = sweep(0j)
    model.add(0j, phasors, noises)
    probe = pf.probe_size(phasors, PERIOD, LIMIT)
    for setting in pf.probe_corrections({"k": probe}, LIMIT):
        model.add(setting["k"], *sweep(setting["k"]))
    applied = (0j, 0j)
    for _ in range(iterations + 1):
        assert model.fit() == len(SPEEDS)
        c = model.solve(speed_dependent=speed_dependent)
        if not model.improves(c, applied):
            break
        applied = c
        model.add(applied, *sweep(applied))
    return model, applied


def test_shared_optimum_is_found_exactly():
    target = 0.025 * np.exp(1.1j)
    gains = {s: 3e-2 * np.exp(1j * s / 20.0) for s in SPEEDS}
    sweep = _linear_motor(dict.fromkeys(SPEEDS, target), gains)
    model, applied = _calibrate(sweep)
    # A constant error gets a constant correction
    assert applied[0] == applied[1]
    assert applied[0] == pytest.approx(target, abs=1e-9)
    assert model.worst(applied) < 1e-12
    # Two sweeps at the same correction are averaged
    assert model.measured(applied)[40.0] == pytest.approx(0.0, abs=1e-12)
    assert len(model.sweeps) == 6


def test_minimax_minimises_the_worst_speed():
    # The best correction drifts with speed, as when the load angle grows,
    # and one speed vibrates far more than the others
    gains = {s: 1e-2 * np.exp(1j * s / 30.0) for s in SPEEDS}
    gains[70.0] *= 5.0
    targets = {s: 0.03 * np.exp(1j * (s - 40.0) / 50.0) for s in SPEEDS}
    sweep = _linear_motor(targets, gains)
    model, applied = _calibrate(sweep, speed_dependent=False)
    worst = model.worst(applied)
    grid = np.linspace(-0.05, 0.05, 201)

    def exact(c):
        return max(abs(gains[s] * (c - targets[s])) for s in SPEEDS)

    best_on_grid = min(exact(complex(re, im)) for re in grid for im in grid)
    assert exact(applied[0]) <= best_on_grid * 1.001
    assert worst == pytest.approx(exact(applied[0]), rel=1e-6)
    assert worst < 0.25 * model.worst(0j)
    # Least squares leaves the busy speed worse but the rest better
    rms = model.solve("rms", speed_dependent=False)
    assert exact(rms[0]) > exact(applied[0])
    assert np.mean(model.predict(rms)) < np.mean(model.predict(applied))
    # A correction that follows the speed does far better still
    varying = model.solve()
    assert varying[0] != varying[1]
    assert model.worst(varying) < 0.5 * worst


def test_noisy_sweeps_still_converge_and_stop():
    target = 0.02 * np.exp(-0.4j)
    gains = {s: 2e-2 * np.exp(1j * s / 25.0) for s in SPEEDS}
    sweep = _linear_motor(dict.fromkeys(SPEEDS, target), gains, noise=2e-5)
    model, applied = _calibrate(sweep, iterations=5)
    # Each sweep places the correction within noise / gain = 1e-3 rad, and
    # the noise does not earn a correction that changes with speed
    assert applied[0] == applied[1]
    assert applied[0] == pytest.approx(target, abs=5e-4)
    # Converged within the noise: refinement stopped before 5 iterations
    assert len(model.sweeps) < 11
    assert model.noise_floor() == pytest.approx(2e-5)
    assert model.worst(applied) < 4.0 * model.noise_floor()
    nudged = (applied[0] + 1e-6, applied[1] + 1e-6)
    assert not model.improves(nudged, applied)


def test_fit_uses_only_the_speed_range_and_enough_points():
    model = pf.HarmonicResponse(50.0, 80.0, LIMIT)
    gains = {s: 1e-2 for s in SPEEDS}
    sweep = _linear_motor(dict.fromkeys(SPEEDS, 0.01 + 0.01j), gains)
    model.add(0j, *sweep(0j))
    assert model.fit() == 0
    model.add(0.02, *sweep(0.02))
    assert model.fit() == 7
    assert list(model.speeds) == [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0]
    assert model.solve() == pytest.approx((0.01 + 0.01j, 0.01 + 0.01j))


def test_solve_stays_within_the_limit():
    gains = {s: 1e-2 for s in SPEEDS}
    sweep = _linear_motor(dict.fromkeys(SPEEDS, 0.5), gains)
    model = pf.HarmonicResponse(40.0, 100.0, LIMIT)
    model.add(0j, *sweep(0j))
    model.add(0.02, *sweep(0.02))
    model.fit()
    assert abs(model.solve()[0]) == pytest.approx(LIMIT)


def test_free_end_stays_with_the_measured_one():
    # Only 40-60 mm/s measured of a 40-100 range: the fast end, which the
    # measurements do not reach, must not run away from the slow end
    measured = [s for s in SPEEDS if s <= 60.0]
    gains = {s: 1e-2 * np.exp(1j * s / 30.0) for s in measured}
    sweep = _linear_motor(dict.fromkeys(measured, 0.02 + 0.01j), gains)
    model = pf.HarmonicResponse(40.0, 100.0, LIMIT)
    model.add(0j, *sweep(0j))
    model.add(0.02, *sweep(0.02))
    model.add(0.02j, *sweep(0.02j))
    model.fit()
    slow, fast = model._solve("minimax", True)
    assert slow == pytest.approx(0.02 + 0.01j, abs=1e-6)
    assert fast == pytest.approx(slow, abs=1e-6)


def test_compare_reports_speeds_that_got_worse():
    before = {40.0: 1e-3, 50.0: 2e-3, 60.0: 1e-4, 70.0: 5e-4}
    after = {40.0: 1e-4, 50.0: 1e-4, 60.0: 2e-4, 70.0: 5.5e-4}
    noise = dict.fromkeys(before, 2e-5)
    worst_b, worst_a, avg_b, avg_a, worse = pf.compare(before, after, noise)
    assert (worst_b, worst_a) == (2e-3, 5.5e-4)
    assert avg_b == pytest.approx(9e-4) and avg_a == pytest.approx(2.375e-4)
    # 60 rose by more than 20 % and more than twice the noise; 70 by 10 %
    assert worse == [60.0]
    assert pf.compare({}, {}, {})[4] == []


def _ramp_sweep(errors_at, noise, model):
    """Sweep function over simulated ramps: ramps up and down in both
    directions through 20-120 mm/s with the displacement errors
    errors_at(speed) plus the correction, which model maps to each speed,
    analysed on a speed grid."""
    grid = mh.plan_speeds(30.0, 110.0, 2.5)

    def sweep(c):
        rows = []
        for motor_dir in (1, -1):
            for up in (True, False):
                v1 = 20.0 if up else 120.0
                a = 50.0 if up else -50.0
                ramp = mh.Ramp(motor_dir, 10.0, v1, a, 2.0, -motor_dir * 60.0)
                times, accel = _varying_samples(
                    ramp, errors_at, lambda s: model.value(c, s), noise
                )
                rows += mh.analyze_ramp(
                    times, accel, ramp, PERIOD, (2,), grid, 2.5, belt=0.4
                )
        table = {}
        for speed, d, res in mh.merge_rows(rows):
            if res.phasor is not None:
                phasors, noises = table.setdefault(d, ({}, {}))
                phasors[speed] = res.phasor
                noises[speed] = res.noise_disp
        return table

    return sweep


def _varying_samples(ramp, errors_at, correction_at, noise):
    rng = np.random.default_rng(5)
    times = ramp.start_time + np.arange(0.0, ramp.duration, 1.0 / 1600.0)
    t = times - ramp.start_time
    speed = ramp.start_speed + ramp.accel * t
    pos = ramp.start_pos + ramp.motor_dir * (
        ramp.start_speed * t + 0.5 * ramp.accel * t * t
    )
    rate = 2.0 * math.pi * ramp.motor_dir * speed / PERIOD
    rate_change = 2.0 * math.pi * ramp.motor_dir * ramp.accel / PERIOD
    angle = 2.0 * math.pi * pos / PERIOD
    # The motor's error varies slowly with speed, so its derivatives are
    # neglected against those of the fast electrical angle
    c = errors_at(speed) + correction_at(speed)
    accel = rng.normal(0.0, noise, len(times)) + ramp.motor_dir * ramp.accel
    accel += np.imag(
        c * np.exp(2j * angle) * (2j * rate_change - (2.0 * rate) ** 2)
    )
    return times, accel


def test_calibration_from_simulated_ramps_cancels_the_error():
    target = 2e-3 * np.exp(0.8j)
    models = {d: pf.HarmonicResponse(40.0, 100.0, LIMIT) for d in (1, -1)}
    sweep = _ramp_sweep(lambda speed: target + 0.0 * speed, 20.0, models[1])
    base = sweep(0j)
    for d, model in models.items():
        model.add(0j, *base[d])
    probe = pf.probe_size(base[1][0], PERIOD, LIMIT)
    for setting in pf.probe_corrections({1: probe, -1: probe}, LIMIT):
        table = sweep(setting[1])
        for d, model in models.items():
            model.add(setting[d], *table[d])
    for d, model in models.items():
        assert model.fit() >= 20
        # The firmware shifts the angle by Im(c e^{ih angle}), the same form
        # as the simulated error, so the correction that cancels it is -C
        c = model.solve()
        assert c[0] == c[1]
        assert c[0] == pytest.approx(-target, rel=0.03)
        assert model.worst(c) < 0.05 * model.worst(0j)
        assert model.noise_floor() < 0.1 * model.worst(0j)


def test_drifting_error_gets_a_correction_that_follows_the_speed():
    # The error's phase turns with speed. A constant correction splits the
    # difference and leaves the ends of the range worst, as minimax does; a
    # correction that follows the speed cancels nearly all of it.
    model = pf.HarmonicResponse(40.0, 100.0, LIMIT)
    sweep = _ramp_sweep(
        lambda speed: 2e-3 * np.exp(1j * (0.8 + (speed - 70.0) / 100.0)),
        0.0,
        model,
    )
    base = sweep(0j)
    model.add(0j, *base[1])
    probe = pf.probe_size(base[1][0], PERIOD, LIMIT)
    for setting in pf.probe_corrections({1: probe}, LIMIT):
        model.add(setting[1], *sweep(setting[1])[1])
    model.fit()
    constant = model.solve(speed_dependent=False)
    residue = model.predict(constant)
    assert np.max(residue) < 0.35 * model.worst(0j)
    assert residue[0] == pytest.approx(residue[-1], rel=0.1)
    assert np.min(residue) < 0.5 * np.max(residue)
    varying = model.solve()
    assert varying[0] != varying[1]
    assert model.worst(varying) < 0.2 * np.max(residue)
    for speed in (40.0, 70.0, 100.0):
        expected = -2e-3 * np.exp(1j * (0.8 + (speed - 70.0) / 100.0))
        assert model.value(varying, speed) == pytest.approx(expected, rel=0.1)
    # Applied, the sweep confirms it
    model.add(varying, *sweep(varying)[1])
    assert max(model.measured(varying).values()) < 0.2 * np.max(residue)
