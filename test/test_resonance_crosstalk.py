# Tests for the accelerometer cross-axis leakage probe in resonance_tester.py:
# the fixed-frequency probing move and the low-passed leakage ratio estimate.
#
# Run: klippy-env/bin/python test/test_resonance_crosstalk.py
import importlib
import math
import os
import random
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KLIPPY = os.path.join(ROOT, "klippy")
STUB_MODULES = ["klippy"]
saved_modules = {name: sys.modules.get(name) for name in STUB_MODULES}
pkg = types.ModuleType("klippy")
pkg.__path__ = [KLIPPY]
sys.modules["klippy"] = pkg

resonance_tester = importlib.import_module("klippy.extras.resonance_tester")
for name in STUB_MODULES:
    if saved_modules[name] is None:
        sys.modules.pop(name, None)

FS = 1600
ACCEL = 150.0


def integrate(seq):
    v = x = last_t = 0.0
    trace = []
    for t, accel, _freq in seq:
        dt = t - last_t
        x += v * dt + 0.5 * accel * dt * dt
        v += accel * dt
        last_t = t
        trace.append((v, x))
    return trace


def accel_at(seq, t):
    for next_t, accel, _freq in seq:
        if t <= next_t:
            return accel
    return 0.0


def synth_samples(seq, tilt, mode_coupling, mode_freq=50.0, zeta=0.03):
    # Primary channel: commanded acceleration through a lightly damped
    # structural mode. Perpendicular channel: a rigid misalignment of the
    # whole reading plus a separate coupling of only the ringing.
    random.seed(1)
    w = 2.0 * math.pi * mode_freq
    sub = 8
    dt = 1.0 / (FS * sub)
    q = qd = 0.0
    samples = []
    end = seq[-1][0]
    for i in range(int((end + 0.5) * FS)):
        for k in range(sub):
            t = (i + k / sub) / FS
            a = accel_at(seq, t) if t < end else 0.0
            qd += (w * w * (a - q) - 2.0 * zeta * w * qd) * dt
            q += qd * dt
        t = i / FS
        a = accel_at(seq, t) if t < end else 0.0
        prim = q + random.gauss(0.0, 5.0)
        perp = tilt * q + mode_coupling * (q - a) + random.gauss(0.0, 5.0)
        samples.append((t, perp, prim, 0.0))
    return samples


def test_probe_move_reverses_and_returns_to_start():
    seq = resonance_tester._gen_fixed_freq_test(2.0, ACCEL, 4)
    trace = integrate(seq)
    assert min(v for v, _ in trace) < -1.0
    assert max(v for v, _ in trace) > 1.0
    assert abs(trace[-1][0]) < 1e-9
    assert abs(trace[-1][1]) < 1e-9
    print("  probe move reverses direction and ends where it began OK")


def test_probe_move_period_matches_requested_frequency():
    freq = 2.0
    seq = resonance_tester._gen_fixed_freq_test(freq, ACCEL, 4)
    zero_crossings = [
        t for (t, _a, _f), (v, _x) in zip(seq, integrate(seq)) if abs(v) < 1e-9
    ]
    assert abs(zero_crossings[1] - zero_crossings[0] - 0.5 / freq) < 1e-9
    assert abs(seq[-1][0] - 4 / freq) < 1e-9
    print("  probe move runs the requested number of cycles at freq OK")


def test_ratio_recovers_pure_misalignment():
    seq = resonance_tester._gen_fixed_freq_test(2.0, ACCEL, 4)
    tilt = math.tan(math.radians(3.0))
    ratio = resonance_tester._crosstalk_ratio(
        synth_samples(seq, tilt, 0.0), "y"
    )
    assert abs(ratio - tilt) < 0.005
    print("  3 degree misalignment is recovered to within 0.5% OK")


def test_ratio_ignores_structural_ringing_coupling():
    seq = resonance_tester._gen_fixed_freq_test(2.0, ACCEL, 4)
    ratio = resonance_tester._crosstalk_ratio(
        synth_samples(seq, 0.0, 0.15), "y"
    )
    assert abs(ratio) < 0.01
    print("  ringing-only cross coupling does not register as leakage OK")


def test_ratio_uses_correct_channels_per_axis():
    seq = resonance_tester._gen_fixed_freq_test(2.0, ACCEL, 4)
    samples = synth_samples(seq, 0.05, 0.0)
    y_ratio = resonance_tester._crosstalk_ratio(samples, "y")
    swapped = [(t, y, x, z) for t, x, y, z in samples]
    x_ratio = resonance_tester._crosstalk_ratio(swapped, "x")
    assert abs(y_ratio - x_ratio) < 1e-9
    print("  X and Y use mirrored primary/perpendicular channels OK")


def test_ratio_rejects_degenerate_input():
    assert (
        resonance_tester._crosstalk_ratio([(0.0, 0.0, 0.0, 0.0)], "x") is None
    )
    flat = [(i / FS, 0.0, 1.0, 0.0) for i in range(100)]
    assert resonance_tester._crosstalk_ratio(flat, "y") is None
    same_time = [(0.0, 0.0, float(i), 0.0) for i in range(10)]
    assert resonance_tester._crosstalk_ratio(same_time, "y") is None
    print("  too-short, motionless or untimed samples give no ratio OK")


if __name__ == "__main__":
    print("resonance crosstalk probe")
    test_probe_move_reverses_and_returns_to_start()
    test_probe_move_period_matches_requested_frequency()
    test_ratio_recovers_pure_misalignment()
    test_ratio_ignores_structural_ringing_coupling()
    test_ratio_uses_correct_channels_per_axis()
    test_ratio_rejects_degenerate_input()
    print("all passed")
