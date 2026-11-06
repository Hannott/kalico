# Standalone regression tests for the opt-in resonance-shaping path added to
# ToolHead: resolving a notch pair from [resonance_model] at klippy:connect,
# and _process_lookahead emitting a shaped profile when it fits (falling back to
# the stock hard trapezoid, unchanged, when it does not).
#
# Run: klippy-env/bin/python test/test_toolhead_resonance_shaping.py
import contextlib
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
    th.resonance_single_axis_ratio = 0.2
    th._resonance_notch_pair = None
    th._resonance_axis_pairs = (None, None)
    th._resonance_shaping_suspended = False
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


PEAKS_X = [
    (56.127, 0.0638, 1.0),
    (110.696, 0.0788, 0.7205),
    (148.114, 0.1, 0.43),
]
PEAKS_Y = [(40.537, 0.0685, 1.0), (148.115, 0.0157, 0.155)]


def test_connect_resolves_single_axis_pairs_with_shorter_ramps():
    th = make_toolhead(True, FakeResonanceModel(PEAKS_X, PEAKS_Y))
    th._handle_resonance_shaping_connect()
    mixed = th._resonance_notch_pair
    pair_x, pair_y = th._resonance_axis_pairs
    assert mixed == (40.537, 56.127)
    assert pair_x[0] == 56.127 and pair_y[0] == 40.537
    ramp_x, ramp_y = (1.0 / p[0] + 1.0 / p[1] for p in (pair_x, pair_y))
    ramp_mixed = 1.0 / mixed[0] + 1.0 / mixed[1]
    assert ramp_x < ramp_mixed and ramp_y < ramp_mixed
    print("  connect resolves shorter single-axis pairs from the model OK")


def test_connect_leaves_an_axis_pair_unset_without_peaks_for_it():
    th = make_toolhead(True, FakeResonanceModel(peaks_x=[(55.0, 0.03, 1.0)]))
    th._handle_resonance_shaping_connect()
    assert th._resonance_axis_pairs == ((55.0, 55.0), None)
    print("  an axis without saved peaks has no single-axis pair OK")


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


class FakeReactor:
    def assert_no_pause(self):
        return contextlib.nullcontext()


class FakeLookahead:
    def __init__(self):
        self.moves = []

    def flush(self, lazy=False):
        moves, self.moves = self.moves, []
        return moves


class FakeMotionQueuing:
    def note_mcu_movequeue_activity(self, *args, **kwargs):
        pass


class FakeExtruder:
    def __init__(self):
        self.move_calls = []
        self.move_segment_calls = []

    def process_move(self, print_time, move, ea_index):
        self.move_calls.append((print_time, move, ea_index))

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


def make_process_moves_toolhead(f_lo, f_hi, max_slide=2.0):
    th = object.__new__(toolhead.ToolHead)
    th.resonance_shaping = True
    th.resonance_max_notch_slide = max_slide
    th.resonance_single_axis_ratio = 0.2
    th._resonance_axis_pairs = (None, None)
    th._resonance_shaping_suspended = False
    th._resonance_stats = {"shaped": 0, "slid": 0, "fallback": 0}
    th.resonance_jerk_dt = 0.001
    th._resonance_notch_pair = (f_lo, f_hi)
    th.special_queuing_state = ""
    th.print_time = 0.0
    th.trapq = object()
    th.trapq_calls = []
    th.trapq_append = lambda *a: th.trapq_calls.append(a)
    th.extra_axes = [FakeExtruder()]
    th.reactor = FakeReactor()
    th.lookahead = FakeLookahead()
    th.motion_queuing = FakeMotionQueuing()
    th.kin_flush_delay = 0.0
    th._advance_move_time = lambda t: None
    return th


def process_moves(th, moves):
    for move in moves:
        if not hasattr(move, "notch_pair"):
            move.notch_pair = th.notch_pair_for(move)
    th.lookahead.moves = list(moves)
    th._process_lookahead()


def test_suspending_shaping_only_affects_new_moves():
    th = make_process_moves_toolhead(55.0, 75.0)
    planned_before = FakeMove(0.0, 100.0, 0.0, 50.0, 20000.0)
    planned_before.notch_pair = th.notch_pair_for(planned_before)
    assert planned_before.notch_pair == (55.0, 75.0)
    assert th.disable_resonance_shaping() is True
    assert th.disable_resonance_shaping() is False
    planned_after = FakeMove(0.0, 100.0, 0.0, 50.0, 20000.0)
    planned_after.notch_pair = th.notch_pair_for(planned_after)
    assert planned_after.notch_pair is None
    process_moves(th, [planned_before, planned_after])
    assert th._resonance_stats["shaped"] == 1
    th.enable_resonance_shaping()
    assert th.notch_pair_for(planned_after) == (55.0, 75.0)
    th.resonance_shaping = False
    assert th.disable_resonance_shaping() is False
    print("  suspending shaping keeps queued moves on their planned pair OK")


class FakeGcode:
    def __init__(self):
        self.messages = []
        self.scripts = []

    def respond_info(self, msg):
        self.messages.append(msg)

    def run_script_from_command(self, script):
        self.scripts.append(script)


class FakeReactorTime:
    def monotonic(self):
        return 0.0


def test_resonance_test_suspends_shaping_unless_input_shaping_is_kept():
    resonance_tester = importlib.import_module("klippy.extras.resonance_tester")
    th = make_process_moves_toolhead(55.0, 75.0)
    th.max_velocity, th.max_accel, th.min_cruise_ratio = 300.0, 5000.0, 0.0
    th.get_status = lambda eventtime: {
        "max_accel": 5000.0,
        "minimum_cruise_ratio": 0.0,
        "max_velocity": 300.0,
    }
    th.get_kinematics = lambda: object()
    gcode = FakeGcode()
    printer = FakePrinter({"toolhead": th, "gcode": gcode})
    printer.get_reactor = lambda: FakeReactorTime()
    with resonance_tester.suspend_limits(printer, 1000.0, 50.0, 0):
        assert th._resonance_shaping_suspended
    assert not th._resonance_shaping_suspended
    assert any("Disabled resonance_shaping" in m for m in gcode.messages)
    assert any("Re-enabled resonance_shaping" in m for m in gcode.messages)
    gcode.messages = []
    with resonance_tester.suspend_limits(printer, 1000.0, 50.0, 1):
        assert not th._resonance_shaping_suspended
    assert not any("resonance_shaping" in m for m in gcode.messages)
    print("  resonance tests suspend shaping unless INPUT_SHAPING=1 OK")


def test_process_moves_uses_the_shaped_profile_when_it_fits():
    th = make_process_moves_toolhead(55.0, 75.0)
    move = FakeMove(0.0, 100.0, 0.0, 50.0, 20000.0, has_extrude=True)
    process_moves(th, [move])
    assert len(th.trapq_calls) > 1
    assert th.extra_axes[0].move_calls == []
    assert len(th.extra_axes[0].move_segment_calls) == len(th.trapq_calls)
    print("  a move that fits the shaped ramp is emitted as slices OK")


def test_process_moves_falls_back_when_the_ramp_does_not_fit():
    th = make_process_moves_toolhead(55.0, 75.0)
    # Far too short a move for the ramp's own minimum runway.
    move = FakeMove(0.0, 300.0, 0.0, 0.5, 20000.0, has_extrude=True)
    process_moves(th, [move])
    assert th._resonance_stats == {"shaped": 0, "slid": 0, "fallback": 1}
    assert len(th.trapq_calls) == 1
    at, ct, dt = (
        th.trapq_calls[0][2],
        th.trapq_calls[0][3],
        th.trapq_calls[0][4],
    )
    assert (at, ct, dt) == (move.accel_t, move.cruise_t, move.decel_t)
    assert (
        th.extra_axes[0].move_calls and not th.extra_axes[0].move_segment_calls
    )
    print(
        "  a move too short for the ramp falls back to the stock trapezoid OK"
    )


def test_process_moves_stock_path_when_shaping_disabled():
    th = make_process_moves_toolhead(55.0, 75.0)
    th.resonance_shaping = False
    move = FakeMove(0.0, 100.0, 0.0, 50.0, 20000.0)
    process_moves(th, [move])
    assert len(th.trapq_calls) == 1
    print(
        "  shaping disabled uses the stock trapezoid even if a pair is set OK"
    )


def test_process_moves_slides_the_pair_for_a_short_move():
    # Needs (0+100)/f_eq*... of runway: 100/55 + 100/75 ~= 3.15 mm at the
    # pair as chosen; 2.5 mm fits once the pair is raised by ~1.3x.
    th = make_process_moves_toolhead(55.0, 75.0)
    move = FakeMove(0.0, 100.0, 0.0, 2.5, 20000.0, has_extrude=True)
    process_moves(th, [move])
    assert th._resonance_stats == {"shaped": 0, "slid": 1, "fallback": 0}
    assert len(th.trapq_calls) > 1
    assert len(th.extra_axes[0].move_segment_calls) == len(th.trapq_calls)
    print("  a slightly short move is shaped with a raised notch pair OK")


def test_process_moves_slide_is_bounded():
    th = make_process_moves_toolhead(55.0, 75.0, max_slide=1.1)
    move = FakeMove(0.0, 100.0, 0.0, 2.5, 20000.0, has_extrude=True)
    process_moves(th, [move])
    assert th._resonance_stats == {"shaped": 0, "slid": 0, "fallback": 1}
    assert len(th.trapq_calls) == 1
    print("  the slide never exceeds resonance_max_notch_slide OK")


def test_process_moves_skips_moves_without_xy_motion():
    th = make_process_moves_toolhead(55.0, 75.0)
    move = FakeMove(0.0, 20.0, 0.0, 5.0, 500.0)
    move.axes_r = [0.0, 0.0, 1.0, 0.0]
    move.axes_d = [0.0, 0.0, 5.0, 0.0]
    process_moves(th, [move])
    assert th._resonance_stats == {"shaped": 0, "slid": 0, "fallback": 0}
    assert len(th.trapq_calls) == 1
    print("  a Z-only move is emitted as the stock trapezoid OK")


def test_shaped_moves_are_counted():
    th = make_process_moves_toolhead(55.0, 75.0)
    process_moves(th, [FakeMove(0.0, 100.0, 0.0, 50.0, 20000.0)])
    assert th._resonance_stats == {"shaped": 1, "slid": 0, "fallback": 0}
    print("  shaped moves are counted for status OK")


def test_single_axis_moves_use_their_own_pair():
    th = make_process_moves_toolhead(40.537, 56.127)
    th._resonance_axis_pairs = ((56.127, 148.114), (40.537, 148.115))
    along_x = FakeMove(0.0, 100.0, 0.0, 50.0, 20000.0)
    along_y = FakeMove(0.0, 100.0, 0.0, 50.0, 20000.0)
    along_y.axes_r = [0.0, 1.0, 0.0, 0.0]
    along_y.axes_d = [0.0, 50.0, 0.0, 0.0]
    mostly_x = FakeMove(0.0, 100.0, 0.0, 50.0, 20000.0)
    mostly_x.axes_r = [0.98, 0.19, 0.0, 0.0]
    mostly_x.axes_d = [49.0, 9.5, 0.0, 0.0]
    diagonal = FakeMove(0.0, 100.0, 0.0, 50.0, 20000.0)
    diagonal.axes_r = [0.7071, 0.7071, 0.0, 0.0]
    diagonal.axes_d = [35.35, 35.35, 0.0, 0.0]
    assert th.notch_pair_for(along_x) == (56.127, 148.114)
    assert th.notch_pair_for(mostly_x) == (56.127, 148.114)
    assert th.notch_pair_for(along_y) == (40.537, 148.115)
    assert th.notch_pair_for(diagonal) == (40.537, 56.127)
    th.resonance_single_axis_ratio = 0.0
    assert th.notch_pair_for(along_x) == (40.537, 56.127)
    th.resonance_single_axis_ratio = 0.2
    th._resonance_axis_pairs = (None, None)
    th._resonance_shaping_suspended = False
    assert th.notch_pair_for(along_x) == (40.537, 56.127)
    print("  single-axis moves pick their axis's pair, others the mixed one OK")


def test_emitter_shapes_with_the_moves_own_pair():
    th = make_process_moves_toolhead(40.537, 56.127)
    th._resonance_axis_pairs = ((56.127, 148.114), (40.537, 148.115))
    along_x = FakeMove(0.0, 100.0, 0.0, 50.0, 20000.0)
    process_moves(th, [along_x])
    slices_x = len(th.trapq_calls)
    th.trapq_calls = []
    diagonal = FakeMove(0.0, 100.0, 0.0, 50.0, 20000.0)
    diagonal.axes_r = [0.7071, 0.7071, 0.0, 0.0]
    diagonal.axes_d = [35.35, 35.35, 0.0, 0.0]
    process_moves(th, [diagonal])
    slices_mixed = len(th.trapq_calls)
    # Shorter ramps mean fewer 1 ms slices
    assert 0 < slices_x < slices_mixed, (slices_x, slices_mixed)
    assert th._resonance_stats["shaped"] == 2
    print("  the emitter renders each move with the pair it was planned on OK")


def make_planning_toolhead(pair, max_accel=5000.0):
    th = object.__new__(toolhead.ToolHead)
    th.resonance_single_axis_ratio = 0.2
    th._resonance_axis_pairs = (None, None)
    th._resonance_shaping_suspended = False
    th.max_velocity = 300.0
    th.max_accel = max_accel
    th.max_accel_to_decel = max_accel * 0.5
    th.junction_deviation = 0.01
    th.extra_axes = []
    th.resonance_shaping = pair is not None
    th._resonance_notch_pair = pair
    return th


def plan(th, points, speed):
    queue = toolhead.LookAheadQueue()
    pos = (0.0, 0.0, 0.0, 0.0)
    moves = []
    for x, y in points:
        end = (x, y, 0.0, 0.0)
        move = toolhead.Move(th, pos, end, speed)
        queue.add_move(move)
        moves.append(move)
        pos = end
    flushed = queue.flush()
    assert flushed == moves
    return moves


def shaped_speeds_fit(moves, pair, max_accel):
    f_lo, f_hi = pair
    return [
        resonance_ramp.profile_dist(
            m.start_v, m.cruise_v, m.end_v, f_lo, f_hi, max_accel
        )
        <= m.move_d * (1.0 + 1e-6)
        for m in moves
    ]


def chord_path(n, length):
    return [((i + 1) * length, 0.0) for i in range(n)]


def test_lookahead_plans_speeds_the_shaped_ramp_can_render():
    pair = (45.2, 76.4)
    for path in (
        chord_path(40, 0.2),
        chord_path(6, 3.0),
        [(50.0, 0.0), (50.0, 50.0), (0.0, 50.0), (0.0, 0.0)],
        [(2.0, 0.0), (2.5, 0.3), (3.0, 0.9), (3.3, 1.8), (3.3, 3.0)],
    ):
        th = make_planning_toolhead(pair)
        moves = plan(th, path, 200.0)
        assert all(shaped_speeds_fit(moves, pair, th.max_accel)), path
        for prev, move in zip(moves, moves[1:]):
            assert abs(prev.end_v - move.start_v) < 1e-9
        assert moves[-1].end_v < 1e-9
        for m in moves:
            assert m.cruise_v >= m.start_v - 1e-9
            assert m.cruise_v >= m.end_v - 1e-9
            segs = resonance_ramp.build_notch_profile(
                m.start_v,
                m.cruise_v,
                m.end_v,
                m.move_d,
                pair[0],
                pair[1],
                m.accel,
                0.001,
            )
            assert segs is not None
    print("  the lookahead only plans speeds the shaped ramp can render OK")


def test_stock_lookahead_plans_speeds_the_ramp_cannot_render():
    pair = (45.2, 76.4)
    th = make_planning_toolhead(None)
    moves = plan(th, chord_path(40, 0.2), 200.0)
    assert not all(shaped_speeds_fit(moves, pair, th.max_accel))
    print("  without the notch reach, short chords would need the fallback OK")


def test_lookahead_without_shaping_is_unchanged():
    th = make_planning_toolhead(None)
    moves = plan(th, [(50.0, 0.0), (50.0, 50.0)], 200.0)
    first = moves[0]
    assert first.notch_pair is None
    assert abs(first.cruise_v - 200.0) < 1e-9
    assert first.start_v == 0.0
    print("  the stock planner is untouched when shaping is off OK")


def main():
    test_connect_resolves_pair_from_the_model()
    test_connect_resolves_single_axis_pairs_with_shorter_ramps()
    test_connect_leaves_an_axis_pair_unset_without_peaks_for_it()
    test_connect_stays_inert_when_shaping_disabled()
    test_connect_stays_inert_without_resonance_model()
    test_connect_stays_inert_with_an_empty_model()
    test_process_moves_uses_the_shaped_profile_when_it_fits()
    test_process_moves_falls_back_when_the_ramp_does_not_fit()
    test_process_moves_stock_path_when_shaping_disabled()
    test_process_moves_slides_the_pair_for_a_short_move()
    test_process_moves_slide_is_bounded()
    test_process_moves_skips_moves_without_xy_motion()
    test_shaped_moves_are_counted()
    test_single_axis_moves_use_their_own_pair()
    test_emitter_shapes_with_the_moves_own_pair()
    test_suspending_shaping_only_affects_new_moves()
    test_resonance_test_suspends_shaping_unless_input_shaping_is_kept()
    test_lookahead_plans_speeds_the_shaped_ramp_can_render()
    test_stock_lookahead_plans_speeds_the_ramp_cannot_render()
    test_lookahead_without_shaping_is_unchanged()
    print("ALL PASS")


if __name__ == "__main__":
    main()
