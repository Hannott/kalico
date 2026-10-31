# Drive stepper driver coil currents directly ("phase stepping")
#
# Copyright (C) 2026  Åsmund Collin <aakjaergaard@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import math

from . import motor_kinematics

# One electrical period (4 full steps) is a full 32 bit phase wrap, and the
# drivers' microstep counter (MSCNT) has 1024 positions per period.
PHASE_PER_PERIOD = 1 << 32
MSCNT_SHIFT = 22
# Registers that take direct coil currents in Trinamic drivers (the
# TMC5160/2160 register map names it after its motion controller use).
DIRECT_REGISTER_NAMES = ("XDIRECT", "XTARGET", "DIRECT_MODE")
REQUIRED_FIELDS = ("direct_mode", "en_pwm_mode", "mscnt", "cur_a", "cur_b")
# Matches MCU flag PC_SWAP_COILS
SWAP_COILS = 1
START_DELAY = 0.250
REVERSE_DWELL = 0.100
QUERY_INTERVAL = 0.050
# Allowed difference between MSCURACT and the expected sine table value, as
# a fraction of the current amplitude.
MAPPING_TOL = 0.08

SINE_QUARTER = [round(32767 * math.sin(i * math.pi / 512)) for i in range(257)]


def scaled_sine(idx, amp):
    """Signed sine of a 1024 per period index, as computed by the MCU."""
    idx &= 1023
    off = idx & 255
    mag = SINE_QUARTER[256 - off if idx & 256 else off]
    val = (mag * amp + (1 << 14)) >> 15
    return -val if idx & 512 else val


def coil_value(phase, amplitude, swap_coils):
    idx = (phase & 0xFFFFFFFF) >> MSCNT_SHIFT
    sin_val = scaled_sine(idx, amplitude)
    cos_val = scaled_sine(idx + 256, amplitude)
    if swap_coils:
        return (cos_val & 0x1FF) | ((sin_val & 0x1FF) << 16)
    return (sin_val & 0x1FF) | ((cos_val & 0x1FF) << 16)


def check_coil_mapping(mscnt, cur_a, cur_b):
    """Return None if MSCURACT matches CUR_A = sine and CUR_B = cosine of
    the microstep counter, else a description of the mismatch."""
    amp = math.hypot(cur_a, cur_b)
    if amp < 16:
        return "coil currents too small to compare (%d, %d)" % (cur_a, cur_b)
    angle = 2.0 * math.pi * mscnt / 1024.0
    exp_a, exp_b = amp * math.sin(angle), amp * math.cos(angle)
    tol = MAPPING_TOL * amp + 2.0
    if abs(cur_a - exp_a) > tol or abs(cur_b - exp_b) > tol:
        return (
            "MSCNT %d gives CUR_A %d, CUR_B %d; expected about %.0f, %.0f"
            % (
                mscnt,
                cur_a,
                cur_b,
                exp_a,
                exp_b,
            )
        )
    return None


def plan_trapezoid(distance, velocity, accel):
    """Plan a move from rest to rest in MCU ticks and phase units.

    Returns (moves, travel) where moves are (count, start_vel, half_accel)
    segments and travel is the exact phase distance they cover. The MCU
    integrates them exactly, so negating start_vel and half_accel retraces
    the path back to the start phase.
    """
    half_accel = max(1, int(round(accel / 2.0)))
    accel = 2 * half_accel
    ramp = max(1, int(velocity // accel))
    while ramp > 1 and 2 * half_accel * ramp * ramp > distance:
        ramp -= 1
    if 2 * half_accel * ramp * ramp > distance:
        raise ValueError("distance too short for the acceleration")
    vel = ramp * accel
    if vel >= 1 << 31:
        raise ValueError("velocity too high")
    cruise = int((distance - 2 * half_accel * ramp * ramp) // vel)
    moves = [(ramp, 0, half_accel)]
    if cruise:
        moves.append((cruise, vel, 0))
    moves.append((ramp, vel, -half_accel))
    return moves, 2 * half_accel * ramp * ramp + cruise * vel


def reverse_moves(moves):
    return [(count, -vel, -half) for count, vel, half in moves]


def sign9(value):
    value &= 0x1FF
    return value - 0x200 if value & 0x100 else value


def decode_coils(value, swap_coils):
    """Return (sine, cosine) coil currents from a direct mode register."""
    low, high = sign9(value), sign9(value >> 16)
    return (high, low) if swap_coils else (low, high)


def coil_angle(sin_val, cos_val):
    """Electrical angle in microstep counter units (0-1023)."""
    angle = math.atan2(sin_val, cos_val) * 1024.0 / (2.0 * math.pi)
    return int(round(angle)) % 1024


def summarize_samples(samples, fields, swap_coils):
    """Describe driver register samples taken during a test move.

    samples are dicts of raw register values with keys "direct", "GCONF",
    "DRV_STATUS" and "MSCURACT". Returns (lines, warnings).
    """
    lines, warnings = [], []
    count = len(samples)
    if not count:
        return lines, warnings
    coils = [decode_coils(smp["direct"], swap_coils) for smp in samples]
    amps = [math.hypot(a, b) for a, b in coils]
    angles = []
    for a, b in coils:
        angle = coil_angle(a, b)
        if not angles or angles[-1] != angle:
            angles.append(angle)
    lines.append(
        "Coil register read back: %d different values in %d reads,"
        " amplitude %.0f-%.0f, angles %s"
        % (
            len(set(smp["direct"] for smp in samples)),
            count,
            min(amps),
            max(amps),
            " ".join(str(a) for a in angles[:12]),
        )
    )
    direct_on = sum(
        1 for smp in samples if fields.get_field("direct_mode", smp["GCONF"])
    )

    def drv(field):
        return [fields.get_field(field, smp["DRV_STATUS"]) for smp in samples]

    cs_actual = drv("cs_actual")
    lines.append(
        "Driver: direct_mode on in %d/%d samples, cs_actual %d-%d,"
        " standstill %d/%d, open load A/B %d/%d"
        % (
            direct_on,
            count,
            min(cs_actual),
            max(cs_actual),
            sum(drv("stst")),
            count,
            sum(drv("ola")),
            sum(drv("olb")),
        )
    )
    act = [
        (
            fields.get_field("cur_a", smp["MSCURACT"]),
            fields.get_field("cur_b", smp["MSCURACT"]),
        )
        for smp in samples
    ]
    lines.append(
        "MSCURACT: %d different values, angles %s"
        % (
            len(set(act)),
            " ".join(str(coil_angle(a, b)) for a, b in sorted(set(act))[:6]),
        )
    )
    if direct_on < count:
        warnings.append("the driver was not in direct mode for the whole test")
    errors = [
        f
        for f in ("s2ga", "s2gb", "s2vsa", "s2vsb", "ot", "otpw")
        if fields.lookup_register(f) and sum(drv(f))
    ]
    if errors:
        warnings.append("driver reported %s" % (", ".join(errors),))
    return lines, warnings


def find_tmc_section(config, stepper_name):
    for section in config.get_prefix_sections("tmc"):
        name = section.get_name()
        if name.split()[-1] == stepper_name:
            return name
    return None


class PhaseStepping:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.stepper_name = config.get_name().split()[-1]
        self.amplitude = config.getint("amplitude", 248, minval=16, maxval=255)
        self.swap_coils = config.getboolean("swap_coils", True)
        tmc_section = find_tmc_section(config, self.stepper_name)
        if tmc_section is None:
            raise config.error(
                "[%s] needs a TMC driver section for '%s'"
                % (config.get_name(), self.stepper_name)
            )
        tmc = self.printer.load_object(config, tmc_section)
        self.mcu_tmc = getattr(tmc, "mcu_tmc", None)
        tmc_spi = getattr(self.mcu_tmc, "tmc_spi", None)
        if tmc_spi is None:
            raise config.error(
                "Phase stepping needs a driver on SPI; [%s] is not"
                % (tmc_section,)
            )
        if tmc_spi.chain_len != 1:
            raise config.error("Phase stepping does not support SPI chains")
        if getattr(tmc_spi.spi, "config_fmt_ticks", None) is not None:
            raise config.error(
                "Phase stepping needs hardware SPI; [%s] uses software SPI"
                % (tmc_section,)
            )
        fields = self.mcu_tmc.get_fields()
        missing = [f for f in REQUIRED_FIELDS if not fields.lookup_register(f)]
        direct_regs = [
            n for n in DIRECT_REGISTER_NAMES if n in self.mcu_tmc.name_to_reg
        ]
        if missing or not direct_regs:
            raise config.error(
                "The driver in [%s] does not support direct coil current"
                " control" % (tmc_section,)
            )
        self.direct_reg_name = direct_regs[0]
        self.direct_reg = self.mcu_tmc.name_to_reg[self.direct_reg_name]
        self.tmc_section = tmc_section
        self.mcu = self.mcu_tmc.get_mcu()
        self.oid = self.mcu.create_oid()
        self.cmd_queue = self.mcu.alloc_command_queue()
        self.tick_freq = None
        self.queue_move_cmd = self.start_cmd = None
        self.stop_cmd = self.query_cmd = None
        self.mcu.register_config_callback(self._build_config)
        self.gcode = self.printer.lookup_object("gcode")
        self.gcode.register_mux_command(
            "PHASE_STEPPING_TEST",
            "STEPPER",
            self.stepper_name,
            self.cmd_PHASE_STEPPING_TEST,
            desc=self.cmd_PHASE_STEPPING_TEST_help,
        )

    def _build_config(self):
        self.tick_freq = self.mcu.get_constants().get("PHASE_STEPPER_TICK_FREQ")
        if self.tick_freq is None:
            raise self.printer.config_error(
                "MCU '%s' firmware lacks phase stepping; rebuild it with"
                " 'Phase stepping support' enabled" % (self.mcu.get_name(),)
            )
        self.mcu.add_config_cmd(
            "config_phase_stepper oid=%d spi_oid=%d reg=%d coil_flags=%d"
            % (
                self.oid,
                self.mcu_tmc.tmc_spi.spi.get_oid(),
                self.direct_reg,
                SWAP_COILS if self.swap_coils else 0,
            )
        )
        cq = self.cmd_queue
        self.queue_move_cmd = self.mcu.lookup_command(
            "phase_stepper_queue_move oid=%c count=%u start_vel=%i"
            " half_accel=%i",
            cq=cq,
        )
        self.start_cmd = self.mcu.lookup_command(
            "phase_stepper_start oid=%c clock=%u phase=%u amplitude=%hu",
            cq=cq,
        )
        self.stop_cmd = self.mcu.lookup_command(
            "phase_stepper_stop oid=%c", cq=cq
        )
        self.query_cmd = self.mcu.lookup_query_command(
            "phase_stepper_query oid=%c",
            "phase_stepper_state oid=%c phase=%u flags=%c pending=%c"
            " ticks=%u lost_ticks=%u locked_ticks=%u xfers=%u"
            " max_cycles=%u avg_cycles=%u",
            oid=self.oid,
            cq=cq,
        )

    def _lookup_stepper(self, gcmd):
        kin = self.printer.lookup_object("toolhead").get_kinematics()
        for stepper in kin.get_steppers():
            if stepper.get_name() == self.stepper_name:
                return kin, stepper
        raise gcmd.error(
            "'%s' is not a kinematic stepper" % (self.stepper_name,)
        )

    def _check_travel(self, gcmd, kin, stepper, distance):
        toolhead = self.printer.lookup_object("toolhead")
        status = toolhead.get_status(self.printer.get_reactor().monotonic())
        if "x" not in status["homed_axes"] or "y" not in status["homed_axes"]:
            raise gcmd.error("Must home X and Y first")
        point = toolhead.get_position()[:3]
        try:
            direction = motor_kinematics.stepper_move_direction(
                kin, stepper, point
            )
        except ValueError as e:
            raise gcmd.error("PHASE_STEPPING_TEST: %s" % (e,))
        for axis in (0, 1):
            pos = point[axis] + direction[axis] * distance
            if not (
                status["axis_minimum"][axis]
                <= pos
                <= status["axis_maximum"][axis]
            ):
                raise gcmd.error(
                    "The test move would leave the axis limits; move the"
                    " toolhead or use a shorter DISTANCE"
                )

    def _read_field(self, reg_name, field):
        fields = self.mcu_tmc.get_fields()
        return fields.get_field(field, self.mcu_tmc.get_register(reg_name))

    def _sample_driver(self):
        get = self.mcu_tmc.get_register
        return {
            "direct": get(self.direct_reg_name),
            "GCONF": get("GCONF"),
            "DRV_STATUS": get("DRV_STATUS"),
            "MSCURACT": get("MSCURACT"),
        }

    def _wait_done(self, start_time, end_time, bus_load):
        """Wait for the MCU to finish its moves.

        Meanwhile driver registers are sampled, which shows whether the
        MCU's coil writes arrive and take effect, and makes the MCU share
        the SPI bus. Returns the final MCU state and the samples taken
        while the motor was moving.
        """
        reactor = self.printer.get_reactor()
        samples = []
        while True:
            eventtime = reactor.monotonic()
            est = self.mcu.estimated_print_time(eventtime)
            if est > end_time:
                state = self.query_cmd.send([self.oid])
                if not state["pending"]:
                    return state, samples
                if est > end_time + 2.0:
                    raise self.printer.command_error(
                        "Phase stepper did not finish its moves"
                    )
            elif est > start_time:
                samples.append(self._sample_driver())
                if bus_load:
                    # The register reads themselves wait on the MCU
                    continue
            reactor.pause(eventtime + QUERY_INTERVAL)

    cmd_PHASE_STEPPING_TEST_help = (
        "Move a stepper forward and back by setting its coil currents directly"
    )

    def cmd_PHASE_STEPPING_TEST(self, gcmd):
        distance = gcmd.get_float("DISTANCE", 10.0, above=0.0, maxval=100.0)
        velocity = gcmd.get_float("VELOCITY", 20.0, above=0.0, maxval=500.0)
        accel = gcmd.get_float("ACCEL", 500.0, above=0.0)
        amplitude = gcmd.get_int(
            "AMPLITUDE", self.amplitude, minval=16, maxval=255
        )
        bus_load = gcmd.get_int("BUS_LOAD", 0, minval=0, maxval=1)
        toolhead = self.printer.lookup_object("toolhead")
        toolhead.wait_moves()
        kin, stepper = self._lookup_stepper(gcmd)
        stepper_enable = self.printer.lookup_object("stepper_enable")
        if not stepper_enable.lookup_enable(
            self.stepper_name
        ).is_motor_enabled():
            raise gcmd.error("'%s' is not enabled" % (self.stepper_name,))
        self._check_travel(gcmd, kin, stepper, distance)
        period = motor_kinematics.stepper_electrical_period(
            self.printer, stepper
        )
        if period is None:
            raise gcmd.error(
                "Cannot find microsteps for '%s'" % (self.stepper_name,)
            )

        mcu_tmc = self.mcu_tmc
        fields = mcu_tmc.get_fields()
        gconf = mcu_tmc.get_register("GCONF")
        if fields.get_field("direct_mode", gconf):
            raise gcmd.error("Driver is already in direct mode")
        mscnt = self._read_field("MSCNT", "mscnt")
        mscuract = mcu_tmc.get_register("MSCURACT")
        mismatch = check_coil_mapping(
            mscnt,
            fields.get_field("cur_a", mscuract),
            fields.get_field("cur_b", mscuract),
        )
        if mismatch is not None:
            raise gcmd.error(
                "Driver microstep table is not a plain sine: %s" % (mismatch,)
            )

        phase_per_mm = PHASE_PER_PERIOD / period
        try:
            moves, travel = plan_trapezoid(
                distance * phase_per_mm,
                velocity * phase_per_mm / self.tick_freq,
                accel * phase_per_mm / self.tick_freq**2,
            )
        except ValueError as e:
            raise gcmd.error("PHASE_STEPPING_TEST: %s" % (e,))
        dwell = int(REVERSE_DWELL * self.tick_freq)
        moves = moves + [(dwell, 0, 0)] + reverse_moves(moves)
        ticks = sum(m[0] for m in moves)
        start_phase = mscnt << MSCNT_SHIFT

        ihold_irun = mcu_tmc.get_register("IHOLD_IRUN")
        # Drivers scale direct mode currents by the hold current
        run_current = fields.set_field(
            "ihold", fields.get_field("irun", ihold_irun), ihold_irun
        )
        self.query_cmd.send([self.oid])
        completed = False
        try:
            mcu_tmc.set_register("IHOLD_IRUN", run_current)
            mcu_tmc.tmc_spi.reg_write(
                self.direct_reg,
                coil_value(start_phase, amplitude, self.swap_coils),
                mcu_tmc.chain_pos,
            )
            # Without step pulses the driver sees standstill, where an
            # enabled stealthChop would take over from the coil currents.
            direct_gconf = fields.set_field("en_pwm_mode", 0, gconf)
            direct_gconf = fields.set_field("direct_mode", 1, direct_gconf)
            mcu_tmc.set_register("GCONF", direct_gconf)
            for count, vel, half in moves:
                self.queue_move_cmd.send([self.oid, count, vel, half])
            est = self.mcu.estimated_print_time(
                self.printer.get_reactor().monotonic()
            )
            start_time = est + START_DELAY
            self.start_cmd.send(
                [
                    self.oid,
                    self.mcu.print_time_to_clock(start_time),
                    start_phase,
                    amplitude,
                ]
            )
            state, samples = self._wait_done(
                start_time, start_time + float(ticks) / self.tick_freq, bus_load
            )
            completed = True
        finally:
            self.stop_cmd.send([self.oid])
            mcu_tmc.set_register("GCONF", gconf)
            mcu_tmc.set_register("IHOLD_IRUN", ihold_irun)
            if not completed:
                # The motor may have stopped anywhere along the test move
                stepper_enable.motor_off()

        end_mscnt = self._read_field("MSCNT", "mscnt")
        cycle_us = 1000000.0 / self.mcu.get_constants()["CLOCK_FREQ"]
        tick_us = 1000000.0 / self.tick_freq
        msg = [
            "Phase stepping test on %s: %.1f mm at %.0f mm/s, %d ticks"
            % (self.stepper_name, travel / phase_per_mm, velocity, ticks),
            "Update: avg %.2f us, max %.2f us (%.0f%% of a %.0f us tick)"
            % (
                state["avg_cycles"] * cycle_us,
                state["max_cycles"] * cycle_us,
                100.0 * state["avg_cycles"] * cycle_us / tick_us,
                tick_us,
            ),
            "Ticks: %d, lost %d, skipped for other SPI traffic %d"
            % (state["ticks"], state["lost_ticks"], state["locked_ticks"]),
            "Coil writes: %d, amplitude %d" % (state["xfers"], amplitude),
        ]
        lines, warnings = summarize_samples(
            samples, mcu_tmc.get_fields(), self.swap_coils
        )
        msg.extend(lines)
        if state["xfers"] < 2 or len(set(s["direct"] for s in samples)) < 2:
            warnings.append(
                "the driver did not receive changing coil currents, so the"
                " motor did not move"
            )
        msg.extend("WARNING: %s" % (w,) for w in warnings)
        if state["phase"] != start_phase & 0xFFFFFFFF or end_mscnt != mscnt:
            msg.append(
                "WARNING: did not return to the start phase (MCU %d, MSCNT"
                " %d -> %d)" % (state["phase"] >> MSCNT_SHIFT, mscnt, end_mscnt)
            )
        else:
            msg.append("Returned to the start phase (MSCNT %d)" % (mscnt,))
        gcmd.respond_info("\n".join(msg))


def load_config_prefix(config):
    return PhaseStepping(config)
