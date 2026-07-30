# Standalone regression tests for the opt-in resonance-shaping path added to
# ToolHead: resolving a notch pair from [resonance_model] at klippy:connect,
# and _process_moves emitting a shaped profile when it fits (falling back to
# the stock hard trapezoid, unchanged, when it does not).
#
# Run: klippy-env/bin/python test/test_toolhead_resonance_shaping.py
import importlib
import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KLIPPY = os.path.join(ROOT, "klippy")
STUB_MODULES = [
    "klippy",
    "klippy.chelper",
    "klippy.kinematics",
    "klippy.kinematics.extruder",
    "klippy.toolhead",
]
saved_modules = {name: sys.modules.get(name) for name in STUB_MODULES}
pkg = types.ModuleType("klippy")
pkg.__path__ = [KLIPPY]
sys.modules["klippy"] = pkg
sys.modules.setdefault("klippy.chelper", types.ModuleType("klippy.chelper"))
kin_pkg = types.ModuleType("klippy.kinematics")
kin_pkg.__path__ = [os.path.join(KLIPPY, "kinematics")]
sys.modules["klippy.kinematics"] = kin_pkg
extruder_stub = types.ModuleType("klippy.kinematics.extruder")
extruder_stub.DummyExtruder = object
extruder_stub.add_printer_objects = lambda config: None
sys.modules["klippy.kinematics.extruder"] = extruder_stub

toolhead = importlib.import_module("klippy.toolhead")
resonance_ramp = importlib.import_module("klippy.extras.resonance_ramp")
for name in STUB_MODULES:
    if saved_modules[name] is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = saved_modules[name]


class FakePrinter:
    def __init__(self, objects):
        self._objects = objects

    def lookup_object(self, name, default=0):
        if name in self._objects:
            return self._objects[name]
        if default != 0:
            return default
        raise RuntimeError("no object " + name)


class FakeResonanceModel:
    def __init__(self, peaks_x=(), peaks_y=()):
        self._peaks = {"x": list(peaks_x), "y": list(peaks_y)}

    def get_model(self, axis):
        return resonance_ramp.PeakModel(self._peaks.get(axis, []))


def make_toolhead(resonance_shaping, resonance_model=None):
    th = object.__new__(toolhead.ToolHead)
    th.resonance_shaping = resonance_shaping
    th.resonance_jerk_dt = 0.001
    th._resonance_notch_pair = None
    objects = {}
    if resonance_model is not None:
        objects["resonance_model"] = resonance_model
    th.printer = FakePrinter(objects)
    return th


def test_connect_resolves_pair_from_the_model():
    model = FakeResonanceModel(peaks_x=[(55.0, 0.03, 1.0)])
    th = make_toolhead(True, model)
    th._handle_resonance_shaping_connect()
    assert th._resonance_notch_pair == (55.0, 55.0)
    print("  connect resolves a notch pair from [resonance_model] OK")


def test_connect_stays_inert_when_shaping_disabled():
    model = FakeResonanceModel(peaks_x=[(55.0, 0.03, 1.0)])
    th = make_toolhead(False, model)
    th._handle_resonance_shaping_connect()
    assert th._resonance_notch_pair is None
    print("  connect leaves the pair unresolved when shaping is disabled OK")


def test_connect_stays_inert_without_resonance_model():
    th = make_toolhead(True, resonance_model=None)
    th._handle_resonance_shaping_connect()
    assert th._resonance_notch_pair is None
    print("  connect degrades gracefully with no [resonance_model] section OK")


def test_connect_stays_inert_with_an_empty_model():
    th = make_toolhead(True, FakeResonanceModel())
    th._handle_resonance_shaping_connect()
    assert th._resonance_notch_pair is None
    print("  connect degrades gracefully when the model has no saved peaks OK")


class FakeExtruder:
    def __init__(self):
        self.move_calls = []
        self.move_segment_calls = []

    def move(self, print_time, move):
        self.move_calls.append((print_time, move))

    def move_segment(
        self,
        print_time,
        move,
        accel_t,
        cruise_t,
        decel_t,
        start_v,
        cruise_v,
        accel,
        seg_dist,
    ):
        self.move_segment_calls.append(print_time)


class FakeMove:
    def __init__(
        self, start_v, cruise_v, end_v, move_d, accel, has_extrude=False
    ):
        self.start_v = start_v
        self.cruise_v = cruise_v
        self.end_v = end_v
        self.move_d = move_d
        self.accel = accel
        self.is_kinematic_move = True
        self.start_pos = (0.0, 0.0, 0.0, 0.0)
        self.axes_r = [1.0, 0.0, 0.0, 1.0 if has_extrude else 0.0]
        self.axes_d = [move_d, 0.0, 0.0, 1.0 if has_extrude else 0.0]
        self.accel_t = 0.1
        self.cruise_t = 0.1
        self.decel_t = 0.1
        self.timing_callbacks = []


def make_process_moves_toolhead(f_lo, f_hi):
    th = object.__new__(toolhead.ToolHead)
    th.resonance_shaping = True
    th.resonance_jerk_dt = 0.001
    th._resonance_notch_pair = (f_lo, f_hi)
    th.special_queuing_state = ""
    th.print_time = 0.0
    th.trapq = object()
    th.trapq_calls = []
    th.trapq_append = lambda *a: th.trapq_calls.append(a)
    th.extruder = FakeExtruder()
    th.kin_flush_delay = 0.0
    th.note_mcu_movequeue_activity = lambda *a, **k: None
    th._advance_move_time = lambda t: None
    return th


def test_process_moves_uses_the_shaped_profile_when_it_fits():
    th = make_process_moves_toolhead(55.0, 75.0)
    move = FakeMove(0.0, 100.0, 0.0, 50.0, 20000.0, has_extrude=True)
    th._process_moves([move])
    assert len(th.trapq_calls) > 1
    assert th.extruder.move_calls == []
    assert len(th.extruder.move_segment_calls) == len(th.trapq_calls)
    print("  a move that fits the shaped ramp is emitted as slices OK")


def test_process_moves_falls_back_when_the_ramp_does_not_fit():
    th = make_process_moves_toolhead(55.0, 75.0)
    # Far too short a move for the ramp's own minimum runway.
    move = FakeMove(0.0, 300.0, 0.0, 0.5, 20000.0, has_extrude=True)
    th._process_moves([move])
    assert len(th.trapq_calls) == 1
    at, ct, dt = (
        th.trapq_calls[0][2],
        th.trapq_calls[0][3],
        th.trapq_calls[0][4],
    )
    assert (at, ct, dt) == (move.accel_t, move.cruise_t, move.decel_t)
    assert th.extruder.move_calls and not th.extruder.move_segment_calls
    print(
        "  a move too short for the ramp falls back to the stock trapezoid OK"
    )


def test_process_moves_stock_path_when_shaping_disabled():
    th = make_process_moves_toolhead(55.0, 75.0)
    th.resonance_shaping = False
    move = FakeMove(0.0, 100.0, 0.0, 50.0, 20000.0)
    th._process_moves([move])
    assert len(th.trapq_calls) == 1
    print(
        "  shaping disabled uses the stock trapezoid even if a pair is set OK"
    )


def main():
    test_connect_resolves_pair_from_the_model()
    test_connect_stays_inert_when_shaping_disabled()
    test_connect_stays_inert_without_resonance_model()
    test_connect_stays_inert_with_an_empty_model()
    test_process_moves_uses_the_shaped_profile_when_it_fits()
    test_process_moves_falls_back_when_the_ramp_does_not_fit()
    test_process_moves_stock_path_when_shaping_disabled()
    print("ALL PASS")


if __name__ == "__main__":
    main()
