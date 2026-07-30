# Tests for resonance_model.py and ShaperCalibrate.extract_resonance_peaks:
# turning a resonance test's measured PSD into a compact, persistable peak
# model, and the SAVE_RESONANCE_MODEL command that drives it.
#
# Requires numpy (like shaper_calibrate.py itself does).
# Run: klippy-env/bin/python test/test_resonance_model.py
import importlib
import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KLIPPY = os.path.join(ROOT, "klippy")
STUB_MODULES = ["klippy"]
saved_modules = {name: sys.modules.get(name) for name in STUB_MODULES}
pkg = types.ModuleType("klippy")
pkg.__path__ = [KLIPPY]
sys.modules["klippy"] = pkg

shaper_calibrate = importlib.import_module("klippy.extras.shaper_calibrate")
resonance_ramp = importlib.import_module("klippy.extras.resonance_ramp")
resonance_model = importlib.import_module("klippy.extras.resonance_model")
for name in STUB_MODULES:
    if saved_modules[name] is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = saved_modules[name]

np = shaper_calibrate.ShaperCalibrate(None).numpy


def synthetic_calibration_data(peaks, min_freq=5.0, max_freq=200.0, n=2000):
    """A CalibrationData whose psd_sum is the sum of a few Lorentzian-like
    bumps at the given (freq, damping, amplitude) triples, plus a small
    noise floor -- enough to exercise real peak detection without needing
    an actual accelerometer capture."""
    freq_bins = np.linspace(min_freq, max_freq, n)
    psd = np.full(n, 0.001)
    for f0, damping, amp in peaks:
        r = freq_bins / f0
        denom = (1.0 - r * r) ** 2 + (2.0 * damping * r) ** 2
        psd = psd + amp / np.maximum(denom, 1e-9)
    zeros = np.zeros(n)
    return shaper_calibrate.CalibrationData(freq_bins, psd, zeros, zeros, zeros)


def test_extract_resonance_peaks_finds_configured_peaks():
    helper = shaper_calibrate.ShaperCalibrate(None)
    data = synthetic_calibration_data([(55.0, 0.03, 5.0), (90.0, 0.04, 2.0)])
    peaks = helper.extract_resonance_peaks(data)
    freqs = sorted(f for f, _d, _w in peaks)
    assert len(freqs) == 2, freqs
    assert abs(freqs[0] - 55.0) < 2.0, freqs
    assert abs(freqs[1] - 90.0) < 2.0, freqs
    print("  extract_resonance_peaks finds both configured peaks OK")


def test_extract_resonance_peaks_weight_is_relative():
    helper = shaper_calibrate.ShaperCalibrate(None)
    data = synthetic_calibration_data([(55.0, 0.03, 5.0), (90.0, 0.04, 2.0)])
    peaks = helper.extract_resonance_peaks(data)
    weights = sorted((w, f) for f, _d, w in peaks)
    assert abs(weights[-1][0] - 1.0) < 1e-6, weights
    assert 0.0 < weights[0][0] < 1.0, weights
    print("  the tallest peak's weight normalizes to 1.0 OK")


def test_extract_resonance_peaks_ignores_flat_noise_floor():
    helper = shaper_calibrate.ShaperCalibrate(None)
    n = 500
    freq_bins = np.linspace(5.0, 200.0, n)
    psd = np.full(n, 0.001)
    data = shaper_calibrate.CalibrationData(
        freq_bins, psd, psd * 0, psd * 0, psd * 0
    )
    peaks = helper.extract_resonance_peaks(data)
    assert peaks == []
    print("  a flat noise floor with no real peak yields nothing OK")


def test_extract_resonance_peaks_respects_max_peaks():
    helper = shaper_calibrate.ShaperCalibrate(None)
    data = synthetic_calibration_data(
        [
            (30.0, 0.03, 5.0),
            (55.0, 0.03, 4.0),
            (90.0, 0.03, 3.0),
            (140.0, 0.03, 2.0),
        ]
    )
    peaks = helper.extract_resonance_peaks(data, max_peaks=2)
    assert len(peaks) <= 2
    print("  max_peaks caps how many peaks are returned OK")


def test_shoulder_peak_damping_is_not_inflated_by_its_neighbour():
    # A mode sitting on the flank of a taller one measures the NEIGHBOUR's
    # slope as its own half-power width unless the search is bounded short
    # of it. On a real capture (48.5 Hz shoulder beside a dominant 57.5 Hz
    # peak) the unbounded estimate came out 0.165 -- about 3x any plausible
    # printer damping ratio -- and that value is persisted to the config.
    helper = shaper_calibrate.ShaperCalibrate(None)
    data = synthetic_calibration_data(
        [(48.5, 0.055, 4.0), (57.7, 0.048, 9.0), (92.0, 0.045, 1.0)]
    )
    peaks = helper.extract_resonance_peaks(data)
    by_freq = {round(f): d for f, d, _w in peaks}
    shoulder = [d for f, d in by_freq.items() if 46 <= f <= 51]
    assert shoulder, by_freq
    assert shoulder[0] <= 0.12, shoulder
    print("  a shoulder peak's damping is not inflated by its neighbour OK")


class FakeGCmd:
    def __init__(self, params):
        self.params = params
        self.infos = []

    def get(self, name, default=None):
        return self.params.get(name, default)

    def error(self, msg):
        return RuntimeError(msg)

    def respond_info(self, msg):
        self.infos.append(msg)


class FakeConfigFile:
    def __init__(self):
        self.set_calls = []

    def set(self, section, option, value):
        self.set_calls.append((section, option, value))


class FakeResonanceTester:
    def __init__(self, last_calibration_data):
        self.last_calibration_data = last_calibration_data


class FakePrinter:
    def __init__(self, objects):
        self._objects = objects

    def command_error(self, msg):
        return RuntimeError(msg)

    def lookup_object(self, name, default=0):
        if name in self._objects:
            return self._objects[name]
        if default != 0:
            return default
        raise RuntimeError("no object " + name)


class FakeConfig:
    def __init__(self, printer, values=None):
        self._printer = printer
        self._values = values or {}

    def get_printer(self):
        return self._printer

    def getfloatlist(self, option, default=()):
        return tuple(self._values.get(option, default))

    def error(self, msg):
        return RuntimeError(msg)


def make_model(last_calibration_data=None, config_values=None):
    printer = FakePrinter(
        {
            "gcode": types.SimpleNamespace(
                register_command=lambda *a, **k: None
            ),
            "resonance_tester": FakeResonanceTester(
                last_calibration_data or {}
            ),
            "configfile": FakeConfigFile(),
        }
    )
    config = FakeConfig(printer, config_values)
    model = resonance_model.ResonanceModel(config)
    return model, printer


def test_get_model_empty_when_nothing_saved():
    model, _printer = make_model()
    pm = model.get_model("x")
    assert isinstance(pm, resonance_ramp.PeakModel)
    assert pm.is_empty()
    print("  get_model returns an empty PeakModel when nothing is saved OK")


def test_config_loads_persisted_peaks():
    model, _printer = make_model(
        config_values={
            "peak_freqs_x": (55.0, 90.0),
            "peak_dampings_x": (0.03, 0.04),
            "peak_weights_x": (1.0, 0.4),
        }
    )
    pm = model.get_model("x")
    assert not pm.is_empty()
    assert pm.dominant_freq() == 55.0
    print("  persisted peaks are loaded back into a usable PeakModel OK")


def test_mismatched_list_lengths_are_a_config_error():
    try:
        make_model(
            config_values={
                "peak_freqs_x": (55.0, 90.0),
                "peak_dampings_x": (0.03,),
                "peak_weights_x": (1.0, 0.4),
            }
        )
    except RuntimeError as e:
        assert "same number of entries" in str(e)
    else:
        raise AssertionError("mismatched list lengths were accepted")
    print("  mismatched peak_freqs/dampings/weights lengths are rejected OK")


def test_save_resonance_model_persists_and_updates_live_state():
    data = synthetic_calibration_data([(55.0, 0.03, 5.0)])
    model, printer = make_model(last_calibration_data={"x": data})
    gcmd = FakeGCmd({"AXIS": "x"})
    model.cmd_SAVE_RESONANCE_MODEL(gcmd)
    configfile = printer.lookup_object("configfile")
    keys = {opt for _sec, opt, _val in configfile.set_calls}
    assert "peak_freqs_x" in keys
    assert "peak_dampings_x" in keys
    assert "peak_weights_x" in keys
    assert not model.get_model("x").is_empty()
    print(
        "  SAVE_RESONANCE_MODEL persists via configfile.set and updates state OK"
    )


def test_save_resonance_model_requires_resonance_tester():
    printer = FakePrinter(
        {"gcode": types.SimpleNamespace(register_command=lambda *a, **k: None)}
    )
    model = resonance_model.ResonanceModel(FakeConfig(printer))
    try:
        model.cmd_SAVE_RESONANCE_MODEL(FakeGCmd({}))
    except RuntimeError as e:
        assert "resonance_tester" in str(e)
    else:
        raise AssertionError("missing [resonance_tester] was accepted")
    print(
        "  SAVE_RESONANCE_MODEL requires [resonance_tester] to be configured OK"
    )


def test_save_resonance_model_requires_calibration_data():
    model, _printer = make_model(last_calibration_data={})
    try:
        model.cmd_SAVE_RESONANCE_MODEL(FakeGCmd({}))
    except RuntimeError as e:
        assert "SHAPER_CALIBRATE" in str(e)
    else:
        raise AssertionError("missing calibration data was accepted")
    print("  SAVE_RESONANCE_MODEL requires prior calibration data OK")


def main():
    test_extract_resonance_peaks_finds_configured_peaks()
    test_extract_resonance_peaks_weight_is_relative()
    test_extract_resonance_peaks_ignores_flat_noise_floor()
    test_extract_resonance_peaks_respects_max_peaks()
    test_shoulder_peak_damping_is_not_inflated_by_its_neighbour()
    test_get_model_empty_when_nothing_saved()
    test_config_loads_persisted_peaks()
    test_mismatched_list_lengths_are_a_config_error()
    test_save_resonance_model_persists_and_updates_live_state()
    test_save_resonance_model_requires_resonance_tester()
    test_save_resonance_model_requires_calibration_data()
    print("ALL PASS")


if __name__ == "__main__":
    main()
