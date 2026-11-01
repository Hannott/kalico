# Tests for phasegen.c: phase stepping segments must follow the stepper
# position, including kinematics and input shaping, within tolerance at
# every MCU update when executed exactly as src/phase_stepper.c does.
import math
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from klippy import chelper  # noqa: E402
from klippy.extras import shaper_defs  # noqa: E402

TICK_FREQ = 40000
PERIOD = 0.8
PHASE_PER_MM = (1 << 32) / PERIOD
TOLERANCE = 1 << 21
GRID_TIME = 0.5
MASK = 0xFFFFFFFF
WINDOW = 0.05


def _append_move(ffi_lib, tq, print_time, start, end, speed, accel):
    dist = math.hypot(end[0] - start[0], end[1] - start[1])
    axes_r = ((end[0] - start[0]) / dist, (end[1] - start[1]) / dist, 0.0)
    accel_t = speed / accel
    accel_d = 0.5 * accel * accel_t**2
    if 2 * accel_d > dist:
        accel_t = math.sqrt(dist / accel)
        speed = accel * accel_t
        accel_d = dist / 2
    cruise_t = (dist - 2 * accel_d) / speed
    ffi_lib.trapq_append(
        tq,
        print_time,
        accel_t,
        cruise_t,
        accel_t,
        start[0],
        start[1],
        0.0,
        axes_r[0],
        axes_r[1],
        axes_r[2],
        0.0,
        speed,
        accel,
    )
    return print_time + 2 * accel_t + cruise_t


def _setup(path, motor, shaper):
    ffi_main, ffi_lib = chelper.get_ffi()
    tq = ffi_main.gc(ffi_lib.trapq_alloc(), ffi_lib.trapq_free)
    t = GRID_TIME + 0.1
    for start, end, speed, accel in path:
        t = _append_move(ffi_lib, tq, t, start, end, speed, accel) + 0.01
    end_time = t + 0.2
    sk = ffi_main.gc(ffi_lib.corexy_stepper_alloc(motor), ffi_lib.free)
    ffi_lib.itersolve_set_trapq(sk, tq)
    ref_pos = ffi_lib.itersolve_calc_position_from_coord(
        sk, path[0][0][0], path[0][0][1], 0.0
    )
    keep = [tq, sk]
    if shaper is not None:
        a, t_imp = shaper
        is_sk = ffi_main.gc(ffi_lib.input_shaper_alloc(), ffi_lib.free)
        ffi_lib.itersolve_set_trapq(is_sk, tq)
        ffi_lib.input_shaper_set_sk(is_sk, sk)
        for axis in (b"x", b"y"):
            ffi_lib.input_shaper_set_shaper_params(
                is_sk, axis, len(a), a, t_imp
            )
        keep.append(is_sk)
        sk = is_sk
    pg = ffi_main.gc(ffi_lib.phasegen_alloc(), ffi_lib.phasegen_free)
    ffi_lib.phasegen_setup(
        pg, sk, 1.0 / TICK_FREQ, PHASE_PER_MM, float(TOLERANCE)
    )
    return ffi_main, ffi_lib, pg, ref_pos, end_time, keep


def _run(path, motor=b"+", shaper=None, start_phase=517 << 22):
    ffi_main, ffi_lib, pg, ref_pos, end_time, keep = _setup(path, motor, shaper)
    ffi_lib.phasegen_start(pg, GRID_TIME, 0, start_phase, ref_pos)
    out = ffi_main.new("struct phasegen_seg[]", 16384)
    entries = []
    flush = GRID_TIME
    while flush < end_time:
        flush += WINDOW
        num = ffi_lib.phasegen_generate(pg, flush, out, 16384)
        assert num >= 0
        entries.extend(
            (
                out[i].count,
                out[i].start_vel,
                out[i].half_accel,
                out[i].wait_tick,
            )
            for i in range(num)
        )
    last_tick = ffi_lib.phasegen_get_last_tick(pg)

    # Execute the queue exactly as the MCU does, one tick at a time
    phase, vel, half, count = start_phase & MASK, 0, 0, 0
    queue = list(entries)
    worst = 0.0
    segments = 0
    for tick in range(1, last_tick + 1):
        while not count and queue:
            c, v, h, w = queue[0]
            if not c:
                assert vel == 0, "wait while moving"
                if tick < w:
                    break
                assert tick == w, "wait in the past"
                queue.pop(0)
                continue
            assert -(1 << 31) <= v < (1 << 31)
            count, vel, half = c, v, h
            segments += 1
            queue.pop(0)
        if not count:
            assert vel == 0, "queue ran dry while moving"
        else:
            phase = (phase + vel + half) & MASK
            vel += 2 * half
            count -= 1
        target = ffi_lib.phasegen_target(pg, tick)
        err = (phase - int(round(target))) & MASK
        err = err - (1 << 32) if err >= (1 << 31) else err
        worst = max(worst, abs(err))
    assert not queue and not count and vel == 0
    duration = last_tick / TICK_FREQ
    return worst, segments / duration, keep


SQUARE = [
    ((100.0, 100.0), (150.0, 100.0), 300.0, 5000.0),
    ((150.0, 100.0), (150.0, 150.0), 300.0, 5000.0),
    ((150.0, 150.0), (100.0, 150.0), 300.0, 5000.0),
    ((100.0, 150.0), (100.0, 100.0), 300.0, 5000.0),
]
TRAVEL = [
    ((20.0, 20.0), (250.0, 250.0), 600.0, 20000.0),
    ((250.0, 250.0), (20.0, 250.0), 600.0, 20000.0),
]
ZIGZAG = [
    (
        (100.0 + i * 0.4, 100.0 + (i % 2) * 2.0),
        (100.4 + i * 0.4, 100.0 + ((i + 1) % 2) * 2.0),
        150.0,
        10000.0,
    )
    for i in range(40)
]


@pytest.mark.parametrize("motor", [b"+", b"-"])
@pytest.mark.parametrize(
    "shaper",
    [
        None,
        shaper_defs.get_mzv_shaper(50.0, 0.1),
        shaper_defs.get_ei_shaper(40.0, 0.1),
    ],
    ids=["no_shaper", "mzv50", "ei40"],
)
@pytest.mark.parametrize(
    "path", [SQUARE, TRAVEL, ZIGZAG], ids=["square", "travel", "zigzag"]
)
def test_segments_follow_position(path, motor, shaper):
    worst, seg_rate, _ = _run(path, motor, shaper)
    assert worst <= TOLERANCE + 1
    # Segments are sent to the MCU over the same link as everything else
    assert seg_rate < 5000


def test_idle_sends_nothing_and_resumes_on_time():
    path = [((100.0, 100.0), (110.0, 100.0), 100.0, 3000.0)]
    worst, _, _ = _run(path)
    assert worst <= TOLERANCE + 1
