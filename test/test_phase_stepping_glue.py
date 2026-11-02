# Tests for the phase_stepping.py glue between the motion planner and the
# MCU: replay the exact commands sent to a fake MCU the way
# src/phase_stepper.c executes them and check the timing, queue flow
# control and position resets against the planned motion.
import contextlib
import math
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from klippy import chelper  # noqa: E402
from klippy.extras import (  # noqa: E402
    phase_stepping,
    shaper_defs,
    tmc,
    tmc5160,
)

CLOCK_FREQ = 180000000
TICK_FREQ = 40000
TICK = CLOCK_FREQ // TICK_FREQ
QUEUE_SIZE = 8
MASK = 0xFFFFFFFF
MSCNT = 517


class Error(Exception):
    pass


class Reactor:
    def __init__(self):
        self.t = 10.0

    def monotonic(self):
        return self.t

    def pause(self, waketime):
        self.t = max(self.t + 0.001, waketime)

    def mutex(self):
        return contextlib.nullcontext()


class Cmd:
    def __init__(self, mcu, name):
        self.mcu, self.name = mcu, name

    def send(self, data, minclock=0, reqclock=0):
        self.mcu.sent.append((self.name, list(data), minclock, reqclock))


class Query:
    def __init__(self, mcu):
        self.mcu = mcu

    def send(self, data):
        return {"tick_clock": self.mcu.grid_clock & MASK, "pending": 0}


class FakeMCU:
    def __init__(self, reactor):
        self.reactor = reactor
        self.sent = []
        self.grid_clock = None

    def get_name(self):
        return "mcu"

    def create_oid(self):
        return 3

    def alloc_command_queue(self):
        return None

    def register_config_callback(self, cb):
        self.config_cb = cb

    def add_config_cmd(self, cmd):
        pass

    def get_constants(self):
        return {
            "PHASE_STEPPER_TICK_FREQ": TICK_FREQ,
            "PHASE_STEPPER_QUEUE_SIZE": QUEUE_SIZE,
            "CLOCK_FREQ": CLOCK_FREQ,
        }

    def lookup_command(self, fmt, cq=None):
        cmd = Cmd(self, fmt.split()[0])
        if cmd.name == "phase_stepper_start":
            mcu = self

            class Start(Cmd):
                def send(self, data, minclock=0, reqclock=0):
                    Cmd.send(self, data, minclock, reqclock)
                    # The MCU timer starts on an arbitrary clock
                    now = mcu.print_time_to_clock(mcu.reactor.t)
                    mcu.grid_clock = now + 1234

            return Start(self, cmd.name)
        return cmd

    def lookup_query_command(self, fmt, resp, oid=None, cq=None):
        return Query(self)

    def estimated_print_time(self, eventtime):
        return eventtime

    def print_time_to_clock(self, print_time):
        return int(print_time * CLOCK_FREQ)

    def clock_to_print_time(self, clock):
        return clock / CLOCK_FREQ

    def clock32_to_clock64(self, clock32):
        base = self.print_time_to_clock(self.reactor.t)
        clock = (base & ~MASK) | clock32
        return clock + (1 << 32) if clock < base - (1 << 31) else clock


class FakeMutex:
    held = False

    def __enter__(self):
        assert not self.held
        self.held = True

    def __exit__(self, *exc):
        self.held = False


class FakeTMCSPI:
    chain_len = 1

    def __init__(self):
        self.mutex = FakeMutex()

    class spi:
        config_fmt_ticks = None

        @staticmethod
        def get_oid():
            return 1

    def reg_write(self, reg, val, chain_pos, print_time=None):
        assert self.mutex.held, "SPI transfer without the driver's mutex"
        return val


class FakeMCUTMC:
    def __init__(self, mcu):
        self.fields = tmc.FieldHelper(
            tmc5160.Fields, tmc5160.SignedFields, tmc5160.FieldFormatters
        )
        self.name_to_reg = tmc5160.Registers
        self.tmc_spi = FakeTMCSPI()
        self.mutex = self.tmc_spi.mutex
        self.chain_pos = 1
        self.mcu = mcu
        f = self.fields
        self.fields.registers["GCONF"] = 0
        f.set_field("irun", 20)
        f.set_field("ihold", 10)
        angle = 2 * math.pi * MSCNT / 1024
        cur = f.set_field("cur_a", round(247 * math.sin(angle)), 0)
        self.regs = {
            "GCONF": 0,
            "MSCNT": MSCNT,
            "MSCURACT": f.set_field("cur_b", round(247 * math.cos(angle)), cur),
        }

    def get_fields(self):
        return self.fields

    def get_mcu(self):
        return self.mcu

    def get_register(self, name):
        return self.regs.get(name, 0)

    def set_register(self, name, val, print_time=None):
        self.regs[name] = val


class Stepper:
    def __init__(self, sk, ffi_lib):
        self.sk, self.ffi_lib = sk, ffi_lib

    def get_name(self):
        return "stepper_x"

    def get_stepper_kinematics(self):
        return self.sk

    def get_commanded_position(self):
        # A freshly wrapped input shaper kinematics reports positions in
        # its own frame; phase stepping must not depend on this value.
        return 0.0

    def get_dir_inverted(self):
        return True, True

    def calc_position_from_coord(self, coord):
        return self.ffi_lib.itersolve_calc_position_from_coord(
            self.sk, coord[0], coord[1], coord[2]
        )

    def get_rotation_distance(self):
        return 40.0, 200 * 16


class Kin:
    def __init__(self, stepper):
        self.stepper = stepper

    def get_steppers(self):
        return [self.stepper]


class Toolhead:
    def __init__(self, stepper):
        self.kin = Kin(stepper)
        self.generators = []

        self.last_move_time = 0.0
        self.position = [100.0, 100.0, 0.0, 0.0]

    def get_position(self):
        return list(self.position)

    def wait_moves(self):
        pass

    def get_last_move_time(self):
        return self.last_move_time

    def dwell(self, delay):
        pass

    def get_kinematics(self):
        return self.kin

    def register_step_generator(self, handler):
        self.generators.append(handler)


class Enable:
    def is_motor_enabled(self):
        return True


class StepperEnable:
    def lookup_enable(self, name):
        return Enable()


class Configfile:
    def get_status(self, eventtime):
        return {"settings": {"stepper_x": {"microsteps": 16}}}


class Printer:
    config_error = command_error = Error

    def __init__(self, mcu_tmc, toolhead, reactor):
        self.tmc_obj = type("T", (), {"mcu_tmc": mcu_tmc})()
        self.reactor = reactor
        self.objs = {
            "toolhead": toolhead,
            "stepper_enable": StepperEnable(),
            "gcode": type(
                "G",
                (),
                {
                    "register_mux_command": lambda *a, **k: 0,
                    "register_command": lambda *a, **k: None,
                },
            )(),
            "configfile": Configfile(),
        }

    def load_object(self, config, name):
        return self.tmc_obj

    def lookup_object(self, name, default=None):
        return self.objs.get(name, default)

    def lookup_objects(self, module=None):
        return []

    def get_reactor(self):
        return self.reactor

    def register_event_handler(self, event, handler):
        pass

    def invoke_shutdown(self, msg):
        raise Error(msg)


class Config:
    error = Error

    def __init__(self, printer):
        self.printer = printer

    def get_printer(self):
        return self.printer

    def get_name(self):
        return "phase_stepping stepper_x"

    def get(self, k, default=None):
        return default

    def getint(self, k, default, **kw):
        return default

    def getboolean(self, k, default):
        return default

    def get_prefix_sections(self, prefix):
        return [type("S", (), {"get_name": lambda s: "tmc5160 stepper_x"})()]


def _append(ffi_lib, tq, t, start, end, speed, accel):
    dist = math.hypot(end[0] - start[0], end[1] - start[1])
    r = ((end[0] - start[0]) / dist, (end[1] - start[1]) / dist)
    accel_t = speed / accel
    if accel * accel_t**2 > dist:
        accel_t = math.sqrt(dist / accel)
        speed = accel * accel_t
    cruise_t = (dist - accel * accel_t**2) / speed
    ffi_lib.trapq_append(
        tq,
        t,
        accel_t,
        cruise_t,
        accel_t,
        start[0],
        start[1],
        0.0,
        r[0],
        r[1],
        0.0,
        0.0,
        speed,
        accel,
    )
    return t + 2 * accel_t + cruise_t


def _replay(mcu, ps, start_phase, targets):
    """Execute the sent commands like the MCU; return the worst error and
    the most entries queued at once."""
    starts = [s for s in mcu.sent if s[0] == "phase_stepper_start"]
    entries = [s for s in mcu.sent if s[0].startswith("phase_stepper_queue")]
    start_clock = starts[-1][1][1]
    grid = mcu.grid_clock
    first = -(-(start_clock - grid) // TICK)
    # Occupancy: an entry may arrive at its minclock and leaves at its start
    events = []
    for name, data, minclock, reqclock in entries:
        events.append((minclock, 1))
        events.append((reqclock, -1))
    occupancy = peak = 0
    for clock, delta in sorted(events, key=lambda e: (e[0], e[1])):
        occupancy += delta
        peak = max(peak, occupancy)
    queue = [(n, d) for n, d, _, _ in entries]
    phase, vel, half, count = start_phase, 0, 0, 0
    worst = 0
    last_tick = ps.ffi_lib.phasegen_get_last_tick(ps.pg)
    for tick in range(first, last_tick + 1):
        now = grid + tick * TICK
        while not count and queue:
            name, d = queue[0]
            if name == "phase_stepper_queue_wait":
                assert vel == 0
                clock = d[1]
                if (now - clock) & MASK >= 1 << 31:
                    break
                assert (now - clock) & MASK < TICK, "late"
                queue.pop(0)
                continue
            count, vel, half = d[1], d[2], d[3]
            queue.pop(0)
        if count:
            phase = (phase + vel + half) & MASK
            vel += 2 * half
            count -= 1
        else:
            assert vel == 0
        target = targets[tick]
        err = (phase - int(round(target))) & MASK
        err = err - (1 << 32) if err >= 1 << 31 else err
        worst = max(worst, abs(err))
    assert not queue
    return worst, peak


@pytest.mark.parametrize("shaped", [False, True], ids=["plain", "mzv"])
def test_glue_follows_planned_motion(shaped):
    ffi_main, ffi_lib = chelper.get_ffi()
    reactor = Reactor()
    mcu = FakeMCU(reactor)
    tq = ffi_main.gc(ffi_lib.trapq_alloc(), ffi_lib.trapq_free)
    base_sk = ffi_main.gc(ffi_lib.corexy_stepper_alloc(b"+"), ffi_lib.free)
    ffi_lib.itersolve_set_trapq(base_sk, tq)
    ffi_lib.itersolve_set_position(base_sk, 100.0, 100.0, 0.0)
    sk = base_sk
    if shaped:
        a, t = shaper_defs.get_mzv_shaper(50.0, 0.1)
        sk = ffi_main.gc(ffi_lib.input_shaper_alloc(), ffi_lib.free)
        ffi_lib.itersolve_set_trapq(sk, tq)
        ffi_lib.input_shaper_set_sk(sk, base_sk)
        for axis in (b"x", b"y"):
            ffi_lib.input_shaper_set_shaper_params(sk, axis, len(a), a, t)
        ffi_lib.itersolve_set_position(sk, 100.0, 100.0, 0.0)
    stepper = Stepper(sk, ffi_lib)
    toolhead = Toolhead(stepper)
    mcu_tmc = FakeMCUTMC(mcu)
    printer = Printer(mcu_tmc, toolhead, reactor)
    ps = phase_stepping.PhaseStepping(Config(printer))
    mcu.config_cb()
    ps._handle_connect()
    ps._enable()
    assert ps.active and mcu_tmc.regs["GCONF"] & (1 << 16)

    # Moves start after the MCU takes over; a long pause in between makes
    # the second burst start from an idle queue.
    t = reactor.t + 0.5
    targets = {}
    state = {"flush": reactor.t}

    def generate_until(end):
        while state["flush"] < end:
            state["flush"] += 0.05
            first = ffi_lib.phasegen_get_last_tick(ps.pg) + 1
            for gen in toolhead.generators:
                gen(state["flush"])
            last = ffi_lib.phasegen_get_last_tick(ps.pg)
            # Targets as seen at generation time, before any later reset
            for tick in range(first, last + 1):
                targets[tick] = ffi_lib.phasegen_target(ps.pg, tick)

    t = _append(ffi_lib, tq, t, (100.0, 100.0), (180.0, 140.0), 400.0, 8000.0)
    # SET_KINEMATIC_POSITION at rest: the same spot becomes (0, 0)
    reset_time = t + 0.5
    generate_until(reset_time)
    ffi_lib.trapq_set_position(tq, reset_time, 0.0, 0.0, 0.0)
    toolhead.last_move_time = reset_time
    toolhead.position = [0.0, 0.0, 0.0, 0.0]
    ps._handle_set_position()
    t = _append(
        ffi_lib,
        tq,
        reset_time + 0.5,
        (0.0, 0.0),
        (-80.0, -40.0),
        600.0,
        20000.0,
    )
    generate_until(t + 0.3)
    worst, peak = _replay(mcu, ps, (MSCNT << 22) & MASK, targets)
    assert worst <= phase_stepping.TOLERANCE + 1
    assert peak <= QUEUE_SIZE

    diff = ps._disable(wait=True)
    assert not ps.active
    assert mcu_tmc.regs["GCONF"] == 0
    assert diff is not None


def test_flow_control_holds_entries_back():
    flow = phase_stepping.FlowControl(3, 100)
    placed = [flow.place(c, w) for c, w in ((0, 120), (10, 0), (5, 0), (2, 0))]
    assert [p[0] for p in placed] == [120, 120, 130, 135]
    assert [p[1] for p in placed] == [None, None, None, 120]
