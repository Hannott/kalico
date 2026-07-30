# PrinterExtruder.move_segment must be equivalent to PrinterExtruder.move.
#
# move() queues one trapq entry for a whole move; move_segment() queues one
# per slice of the resonance-shaped profile. Both feed self.last_position
# straight back in as the NEXT entry's start position, so if the two
# disagree about how far the extruder advanced, the queue gets a position
# discontinuity and step generation fails with an internal error on the
# first extruding G1 -- which is exactly what happened when move_segment
# accumulated toolhead path distance where move() applies extruder
# distance (abs(axes_r[3]) * path distance).
#
# Run: klippy-env/bin/python test/test_extruder_move_segment.py
import importlib
import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KLIPPY = os.path.join(ROOT, "klippy")
STUB = ["klippy", "klippy.chelper", "klippy.stepper"]
saved = {n: sys.modules.get(n) for n in STUB}
pkg = types.ModuleType("klippy")
pkg.__path__ = [KLIPPY]
sys.modules["klippy"] = pkg
chelper = types.ModuleType("klippy.chelper")
chelper.get_ffi = lambda: (None, None)
sys.modules["klippy.chelper"] = chelper
sys.modules["klippy.stepper"] = types.ModuleType("klippy.stepper")

extruder = importlib.import_module("klippy.kinematics.extruder")
resonance_ramp = importlib.import_module("klippy.extras.resonance_ramp")
for n in STUB:
    if saved[n] is None:
        sys.modules.pop(n, None)
    else:
        sys.modules[n] = saved[n]


class FakeMove:
    def __init__(self, move_d, extrude_ratio, vs, vc, ve, accel):
        self.move_d = move_d
        self.is_kinematic_move = True
        self.start_v, self.cruise_v, self.end_v = vs, vc, ve
        self.accel = accel
        self.axes_r = [1.0, 0.0, 0.0, extrude_ratio]
        self.axes_d = [move_d, 0.0, 0.0, extrude_ratio * move_d]
        self.accel_t = self.cruise_t = self.decel_t = 0.05
        self.start_pos = (0.0, 0.0, 0.0, 0.0)


def make_extruder():
    e = object.__new__(extruder.PrinterExtruder)
    e.last_position = [0.0, 0.0, 0.0]
    e._seg_move = None
    e._seg_move_start = None
    e._seg_pos = 0.0
    e.trapq = object()
    e.appends = []
    e.trapq_append = lambda *a: e.appends.append(a)
    e.extruder_stepper = None
    return e


def test_segmented_and_whole_move_agree_on_final_position():
    for move_d, ratio in (
        (50.0, 0.04),  # ordinary print move: ratio is small, which is
        (20.0, 0.012),  # exactly where dropping abs_axis_r hurts most
        (5.0, 0.15),
        (30.0, 1.0),
    ):
        move = FakeMove(move_d, ratio, 0.0, 100.0, 0.0, 20000.0)
        segs = resonance_ramp.build_notch_profile(
            move.start_v,
            move.cruise_v,
            move.end_v,
            move.move_d,
            55.0,
            75.0,
            move.accel,
            0.001,
        )
        assert segs, (move_d, ratio)

        whole = make_extruder()
        whole.move(0.0, move)

        sliced = make_extruder()
        t = 0.0
        for at, ct, dt, sv, cv, a, dist in segs:
            sliced.move_segment(t, move, at, ct, dt, sv, cv, a, dist)
            t += at + ct + dt

        for i in range(3):
            assert (
                abs(whole.last_position[i] - sliced.last_position[i]) < 1e-9
            ), (
                move_d,
                ratio,
                i,
                whole.last_position,
                sliced.last_position,
            )
    print(
        "  segmented emission lands on the same extruder position as move() OK"
    )


def test_consecutive_moves_stay_continuous():
    # The failure mode is cumulative: each move hands last_position to the
    # next as its start position, so an error per move compounds.
    e = make_extruder()
    ref = make_extruder()
    t = 0.0
    for _ in range(5):
        move = FakeMove(40.0, 0.035, 0.0, 120.0, 0.0, 20000.0)
        segs = resonance_ramp.build_notch_profile(
            0.0, 120.0, 0.0, 40.0, 55.0, 75.0, move.accel, 0.001
        )
        for at, ct, dt, sv, cv, a, dist in segs:
            e.move_segment(t, move, at, ct, dt, sv, cv, a, dist)
            t += at + ct + dt
        ref.move(0.0, move)
    for i in range(3):
        assert abs(e.last_position[i] - ref.last_position[i]) < 1e-9, (
            i,
            e.last_position,
            ref.last_position,
        )
    print(
        "  five chained moves do not accumulate any extruder position drift OK"
    )


def test_each_slice_starts_where_the_previous_ended():
    # Every queued entry's start position must be the previous entry's end,
    # or the motion queue has a jump in it.
    e = make_extruder()
    move = FakeMove(50.0, 0.04, 0.0, 100.0, 0.0, 20000.0)
    segs = resonance_ramp.build_notch_profile(
        0.0, 100.0, 0.0, 50.0, 55.0, 75.0, move.accel, 0.001
    )
    t = 0.0
    positions = []
    for at, ct, dt, sv, cv, a, dist in segs:
        e.move_segment(t, move, at, ct, dt, sv, cv, a, dist)
        positions.append(list(e.last_position))
        t += at + ct + dt
    # start position recorded in each append (args 5,6,7) must equal the
    # running position after the previous slice
    prev = [0.0, 0.0, 0.0]
    for args, after in zip(e.appends, positions):
        start = list(args[5:8])
        for i in range(3):
            assert abs(start[i] - prev[i]) < 1e-9, (start, prev)
        prev = after
    print("  each queued slice starts exactly where the previous one ended OK")


def main():
    test_segmented_and_whole_move_agree_on_final_position()
    test_consecutive_moves_stay_continuous()
    test_each_slice_starts_where_the_previous_ended()
    print("ALL PASS")


if __name__ == "__main__":
    main()
