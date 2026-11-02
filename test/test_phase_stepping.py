# Tests for the host side helpers of phase_stepping.py
import cmath
import math
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from klippy.extras import phase_stepping as ps  # noqa: E402


def _integrate(moves, phase=0):
    """Integer phase path exactly as the MCU's stepper_tick() computes it."""
    for count, vel, half in moves:
        for _ in range(count):
            phase += vel + half
            vel += 2 * half
    return phase, vel


def test_scaled_sine_matches_a_sine():
    for angle in range(0, 4096, 7):
        expected = 248 * math.sin(2 * math.pi * angle / 4096)
        assert ps.scaled_sine(angle, 248) == pytest.approx(expected, abs=1.0)


def test_coil_value_packs_9_bit_currents():
    # MSCNT 0: sine 0, cosine +248
    assert ps.coil_value(0, 248, False) == 248 << 16
    assert ps.coil_value(0, 248, True) == 248
    # MSCNT 768 (270 degrees): sine -248, cosine 0
    value = ps.coil_value(768 << ps.MSCNT_SHIFT, 248, False)
    assert value & 0x1FF == (-248) & 0x1FF
    assert value >> 16 == 0


def test_check_coil_mapping():
    assert ps.check_coil_mapping(0, 0, 247) is None
    assert ps.check_coil_mapping(256, 247, 0) is None
    assert ps.check_coil_mapping(128, 175, 175) is None
    assert ps.check_coil_mapping(256, 0, 247) is not None
    assert ps.check_coil_mapping(0, 0, 3) is not None


def test_trapezoid_covers_the_planned_distance_and_stops():
    moves, travel = ps.plan_trapezoid(5.4e10, 2.7e6, 1677.0)
    phase, vel = _integrate(moves)
    assert phase == travel
    assert vel == 0
    assert travel == pytest.approx(5.4e10, rel=0.001)
    assert len(moves) == 3


def test_trapezoid_without_cruise():
    moves, travel = ps.plan_trapezoid(1.0e6, 2.7e6, 1677.0)
    phase, vel = _integrate(moves)
    assert (phase, vel) == (travel, 0)
    assert travel <= 1.0e6
    assert len(moves) == 2


def test_reverse_returns_exactly_to_the_start():
    moves, _ = ps.plan_trapezoid(5.3e9, 3.1e6, 1201.0)
    full = moves + [(400, 0, 0)] + ps.reverse_moves(moves)
    phase, vel = _integrate(full, phase=123456789)
    assert (phase, vel) == (123456789, 0)


def test_trapezoid_rejects_impossible_moves():
    with pytest.raises(ValueError):
        ps.plan_trapezoid(10.0, 2.7e6, 1677.0)
    with pytest.raises(ValueError):
        ps.plan_trapezoid(1e15, 3e9, 1e6)


def test_decode_coils_inverts_coil_value():
    for mscnt in (0, 100, 256, 517, 900):
        phase = mscnt << ps.MSCNT_SHIFT
        for swap in (False, True):
            sin_val, cos_val = ps.decode_coils(
                ps.coil_value(phase, 248, swap), swap
            )
            assert ps.coil_angle(sin_val, cos_val) == pytest.approx(
                mscnt, abs=1
            )
            assert math.hypot(sin_val, cos_val) == pytest.approx(248, abs=2)


class _Fields:
    def __init__(self, values):
        self.values = values

    def get_field(self, name, reg_value):
        return self.values.get(name, lambda v: 0)(reg_value)

    def lookup_register(self, name):
        return name in ("ot",)


def test_summarize_samples_reports_coils_and_driver_state():
    fields = _Fields(
        {
            "direct_mode": lambda v: v,
            "cs_actual": lambda v: 31,
            "ot": lambda v: 0,
        }
    )
    samples = [
        {
            "direct": ps.coil_value(m << ps.MSCNT_SHIFT, 248, True),
            "GCONF": 1 if m < 400 else 0,
            "DRV_STATUS": 0,
            "MSCURACT": 0,
        }
        for m in (0, 128, 256, 512)
    ]
    lines, warnings = ps.summarize_samples(samples, fields, True)
    assert "4 different values in 4 reads" in lines[0]
    assert "angles 0 128 256 512" in lines[0]
    assert "direct_mode on in 3/4 samples" in lines[1]
    assert warnings == ["the driver was not in direct mode for the whole test"]
    assert ps.summarize_samples([], fields, True) == ([], [])


def test_with_field_leaves_the_register_cache_alone():
    from klippy.extras import tmc, tmc5160

    fields = tmc.FieldHelper(tmc5160.Fields, tmc5160.SignedFields)
    fields.registers["IHOLD_IRUN"] = cached = fields.set_field("irun", 20, 0)
    value = ps.with_field(fields, "ihold", 20, cached)
    assert fields.get_field("ihold", value) == 20
    assert fields.get_field("irun", value) == 20
    assert fields.registers["IHOLD_IRUN"] == cached


def test_correction_parse_format_round_trip():
    corr, speeds = ps.parse_correction("2:0.0123:1.5708, 4:0.004:-0.3")
    assert speeds is None
    assert corr[2] == pytest.approx((cmath.rect(0.0123, 1.5708),) * 2)
    assert corr[4] == pytest.approx((cmath.rect(0.004, -0.3),) * 2)
    text = ps.format_correction(corr)
    assert text == "2:0.01230:1.5708 4:0.00400:-0.3000"
    again, speeds = ps.parse_correction(text)
    assert speeds is None
    assert again[2] == pytest.approx(corr[2]) and again[4] == pytest.approx(
        corr[4]
    )
    assert ps.parse_correction("") == ({}, None)
    with pytest.raises(ValueError):
        ps.parse_correction("2:0.01")


def test_speed_dependent_correction_round_trip():
    text = "2:0.03000:2.2000:50 2:0.02500:2.4000:100 4:0.00100:-0.4000"
    corr, speeds = ps.parse_correction(text)
    assert speeds == (50.0, 100.0)
    assert corr[2][0] == pytest.approx(cmath.rect(0.03, 2.2))
    assert corr[2][1] == pytest.approx(cmath.rect(0.025, 2.4))
    assert corr[4][0] == corr[4][1]
    assert ps.format_correction(corr, speeds) == text
    # The order of the entries does not matter
    assert ps.parse_correction("2:0.025:2.4:100 2:0.03:2.2:50")[1] == (
        50.0,
        100.0,
    )
    for bad in (
        "2:0.03:2.2:50",
        "2:0.03:2.2:50 2:0.02:2.4:50",
        "2:0.03:2.2:50 2:0.02:2.4:100 2:0.01:0:75",
        "2:0.03:2.2:50 2:0.02:2.4:100 4:0.01:0:60 4:0.01:0:100",
    ):
        with pytest.raises(ValueError):
            ps.parse_correction(bad)


def test_correction_at_interpolates_between_the_ends():
    pair, speeds = (0.02 + 0j, 0.04j), (50.0, 100.0)
    assert ps.correction_at(pair, speeds, 40.0) == pair[0]
    assert ps.correction_at(pair, speeds, 50.0) == pair[0]
    assert ps.correction_at(pair, speeds, 75.0) == pytest.approx(0.01 + 0.02j)
    assert ps.correction_at(pair, speeds, 100.0) == pair[1]
    assert ps.correction_at(pair, speeds, 120.0) == pair[1]
    assert ps.correction_at(pair, None, 120.0) == pair[0]


def test_set_correction_speeds_keeps_the_corrections_in_place():
    obj = ps.PhaseStepping.__new__(ps.PhaseStepping)
    obj.corrections = {
        "forward": {2: (0.02 + 0j, 0.04j)},
        "backward": {4: (0.01j, 0.01j)},
    }
    obj.correction_speeds = (50.0, 100.0)
    obj.set_correction_speeds((75.0, 100.0))
    assert obj.corrections["forward"][2] == pytest.approx((0.01 + 0.02j, 0.04j))
    assert obj.corrections["backward"][4] == (0.01j, 0.01j)
    obj.set_correction_speeds(None)
    assert obj.corrections["forward"][2] == pytest.approx(
        (0.01 + 0.02j, 0.01 + 0.02j)
    )
    with pytest.raises(ValueError):
        obj.set_correction_speeds((100.0, 50.0))
    obj.set_harmonic("forward", 2, 0.5 + 0j)
    assert obj.get_harmonic("forward", 2) == pytest.approx(
        (ps.CORRECTION_MAX, ps.CORRECTION_MAX)
    )
    obj.set_harmonic("forward", 2, (0j, 0j))
    assert obj.get_harmonic("forward", 2) == (0j, 0j)
    assert 2 not in obj.corrections["forward"]


def test_correction_speed_params_map_the_range_to_16_bits():
    period, tick_freq = 0.8, 40000
    per_mm_s = 2.0**32 / period / tick_freq
    lo, scale = ps.correction_speed_params((50.0, 100.0), period, tick_freq)
    assert lo == round(50.0 * per_mm_s)

    def mcu_fraction(speed):
        # As correction_shift() in src/phase_stepper.c computes it
        vel = int(round(speed * per_mm_s))
        if vel <= lo:
            return 0
        return min(0xFFFF, ((vel - lo) * scale) >> 32)

    assert mcu_fraction(40.0) == 0 and mcu_fraction(50.0) == 0
    assert mcu_fraction(75.0) == pytest.approx(32768, abs=2)
    assert mcu_fraction(100.0) >= 65534 and mcu_fraction(150.0) == 65535
    slow, fast = -20, 40
    assert slow + (((fast - slow) * mcu_fraction(75.0)) >> 16) == 10
    assert ps.correction_speed_params(None, period, tick_freq) == (0, 0)


def test_correction_table_follows_the_harmonics():
    table = ps.correction_table({2: 0.02 + 0j})
    scale = 4096.0 / (2.0 * math.pi)
    for i in (0, 64, 128, 300, 1000):
        theta = 2.0 * math.pi * i / 1024
        assert table[i] == round(0.02 * math.sin(2 * theta) * scale)
    # Corrections are limited to what a signed byte holds on the MCU
    assert max(ps.correction_table({1: 1.0 + 0j})) == ps.CORRECTION_LIMIT


def test_coil_value_applies_the_shift():
    phase = 300 << ps.MSCNT_SHIFT
    assert ps.coil_value(phase, 248, True, 4) == ps.coil_value(
        301 << ps.MSCNT_SHIFT, 248, True
    )


class _Rail:
    def __init__(self, steppers):
        self.steppers = steppers

    def get_steppers(self):
        return self.steppers


class _HomingMove:
    def __init__(self, steppers):
        self.endstop = _Rail(steppers)

    def get_mcu_endstops(self):
        return [self.endstop]


def _bare_phase_stepping(stepper, auto_enable):
    obj = ps.PhaseStepping.__new__(ps.PhaseStepping)
    obj.stepper = stepper
    obj.auto_enable = auto_enable
    obj.active = False
    obj.resume_after_homing = False

    def enable():
        obj.active = True

    def disable(wait=True):
        obj.active = False

    obj._enable = enable
    obj._disable = disable
    obj._report_disagreement = lambda diff, gcmd=None: None
    return obj


@pytest.mark.parametrize("auto_enable", [True, False])
def test_corexy_homing_leaves_both_motors_phase_stepping(auto_enable):
    a, b = object(), object()
    motors = [
        _bare_phase_stepping(a, auto_enable),
        _bare_phase_stepping(b, auto_enable),
    ]
    if not auto_enable:
        for m in motors:
            m.active = True
    # G28: X then Y; on CoreXY both motors move when homing either axis
    for rail in (_Rail([a]), _Rail([b])):
        for m in motors:
            m._handle_homing_move_begin(_HomingMove([a, b]))
        for m in motors:
            m._handle_home_rails_end(None, [rail])
    assert all(m.active for m in motors)


def test_hold_settings_mirror_the_driver_standstill_reduction():
    # hold_current unset: the driver module sets IHOLD to IRUN
    assert ps.hold_settings(31, 31, 8, 10, 12e6, 40000, 248)[0] == 248
    hold_amp, delay, ramp = ps.hold_settings(31, 15, 8, 10, 12e6, 40000, 248)
    assert hold_amp == 124
    # 2^20 clocks to standstill plus TPOWERDOWN * 2^18
    assert delay == round((2**20 + 10 * 2**18) / 12e6 * 40000)
    # 16 IHOLDDELAY steps of 2^18 clocks spread over 124 amplitude steps
    assert ramp == round(16 * 8 * 2**18 / 12e6 * 40000 / 124)
    assert ps.hold_settings(31, 15, 0, 0, 12e6, 40000, 248)[2] == 0


class _DriverModule:
    def __init__(self):
        from klippy.extras import tmc, tmc5160

        self.fields = tmc.FieldHelper(tmc5160.Fields, tmc5160.SignedFields)
        self.fields.set_field("irun", 20)
        self.fields.set_field("ihold", 10)
        self.fields.set_field("en_pwm_mode", 1)
        self.writes = []

    def get_fields(self):
        return self.fields

    def set_register(self, reg_name, val, print_time=None):
        self.writes.append((reg_name, val))


def _guarded():
    obj = ps.PhaseStepping.__new__(ps.PhaseStepping)
    obj.mcu_tmc = _DriverModule()
    obj.active = obj.direct = False
    obj.holds = 0

    def send_hold():
        obj.holds += 1

    obj._send_hold = send_hold
    obj._guard_driver_writes()
    return obj


def test_driver_writes_keep_direct_mode_and_its_current():
    obj = _guarded()
    tmc_mod = obj.mcu_tmc
    fields = tmc_mod.fields
    obj.direct = obj.active = True
    # SET_TMC_CURRENT while phase stepping: the driver module writes its
    # usual IHOLD_IRUN and GCONF values
    value = fields.set_field("irun", 16)
    tmc_mod.set_register("IHOLD_IRUN", value)
    tmc_mod.set_register("GCONF", fields.registers["GCONF"])
    (_, ihold_irun), (_, gconf) = tmc_mod.writes
    assert fields.get_field("ihold", ihold_irun) == 16
    assert fields.get_field("irun", ihold_irun) == 16
    assert fields.get_field("direct_mode", gconf) == 1
    assert fields.get_field("en_pwm_mode", gconf) == 0
    assert obj.holds == 1
    # The module's own values return when direct mode ends
    tmc_mod.writes.clear()
    obj.active = False
    obj._leave_direct()
    assert dict(tmc_mod.writes) == {
        "GCONF": fields.registers["GCONF"],
        "IHOLD_IRUN": fields.registers["IHOLD_IRUN"],
    }
    assert fields.get_field("ihold", fields.registers["IHOLD_IRUN"]) == 10
    assert fields.get_field("en_pwm_mode", fields.registers["GCONF"]) == 1


class _Gcode:
    def __init__(self, handler):
        self.handlers = {"FORCE_MOVE": handler}
        self.gcode_help = {"FORCE_MOVE": "Manually move a stepper"}

    def register_command(self, cmd, func, desc=None):
        if func is None:
            return self.handlers.pop(cmd, None)
        self.handlers[cmd] = func
        self.gcode_help[cmd] = desc


class _GCmd:
    def __init__(self, stepper):
        self.stepper = stepper

    def get(self, name, default=None):
        return self.stepper if name == "STEPPER" else default


def test_force_move_runs_with_phase_stepping_off():
    calls = []
    obj = _bare_phase_stepping(object(), False)
    obj.stepper_name = "stepper_x"
    obj.gcode = _Gcode(lambda gcmd: calls.append(("move", obj.active)))
    obj._motor_enabled = lambda: True
    obj._wrap_step_command("FORCE_MOVE")
    obj._wrap_step_command("STEPPER_BUZZ")
    assert "STEPPER_BUZZ" not in obj.gcode.handlers
    run = obj.gcode.handlers["FORCE_MOVE"]
    assert obj.gcode.gcode_help["FORCE_MOVE"] == "Manually move a stepper"
    obj.active = True
    run(_GCmd("stepper_x"))
    assert calls == [("move", False)] and obj.active
    # Other steppers move with phase stepping left alone
    run(_GCmd("stepper_y"))
    assert calls[-1] == ("move", True)
    # A motor turned off by the command stays off
    obj._motor_enabled = lambda: False
    run(_GCmd("stepper_x"))
    assert calls[-1] == ("move", False) and not obj.active


def test_stats_report_lead_and_mcu_counters():
    obj = ps.PhaseStepping.__new__(ps.PhaseStepping)
    obj.stepper_name = "stepper_x"
    obj.active = False
    assert obj.stats(0.0) == (False, "phase_stepping_stepper_x: off")
    obj.active = True
    obj.lead_min, obj.generate_max = 0.2312, 0.0042
    obj.mcu = type("M", (), {"get_constants": lambda s: {"CLOCK_FREQ": 1e6}})()
    obj.mcu_stats = {
        "lost_ticks": 2,
        "locked_ticks": 7,
        "replays": 0,
        "max_cycles": 3,
        "avg_cycles": 1,
    }
    active, msg = obj.stats(0.0)
    assert active
    assert msg == (
        "phase_stepping_stepper_x: lead_min=0.231 generate_max=0.0042"
        " lost_ticks=2 locked_ticks=7 replays=0 update_max_us=3.0"
        " update_avg_us=1.0"
    )
    # Each interval starts over
    assert obj.stats(0.0)[1] == (
        "phase_stepping_stepper_x: lead_min=- generate_max=0.0000"
    )
