# Drive stepper driver coil currents directly ("phase stepping")
#
# Copyright (C) 2026  Åsmund Collin <aakjaergaard@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import collections
import logging
import math

from klippy import chelper

from . import motor_kinematics, tmc

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
MASK32 = 0xFFFFFFFF
# Half the resolution of the MCU's sine table (1024 entries per period)
TOLERANCE = float(1 << 21)
SEG_BUFFER = 4096
# Matches PG_ERR_FULL in klippy/chelper/phasegen.h
PG_ERR_FULL = -1
DRAIN_TIMEOUT = 2.0
START_DELAY = 0.250
REVERSE_DWELL = 0.100
QUERY_INTERVAL = 0.050
STEP_TEST_SPEED = 10.0
# Allowed difference between MSCURACT and the expected sine table value, as
# a fraction of the current amplitude.
MAPPING_TOL = 0.08

SINE_QUARTER = [
    round(32767 * math.sin(i * math.pi / 2048)) for i in range(1025)
]
# The MCU computes the current angle in 1/4096 of an electrical period and
# corrects it by a table of 1024 signed bytes per direction.
ANGLE_SHIFT = 20
CORRECTION_SIZE = 1024
CORRECTION_LIMIT = 127
CORRECTION_CHUNK = 40
# Largest correction magnitude (radians) that fits the MCU's signed bytes
CORRECTION_MAX = CORRECTION_LIMIT * 2.0 * math.pi / 4096.0
DIRECTIONS = ("forward", "backward")


def scaled_sine(angle, amp):
    """Signed sine of a 4096 per period angle, as computed by the MCU."""
    angle &= 4095
    off = angle & 1023
    mag = SINE_QUARTER[1024 - off if angle & 1024 else off]
    val = (mag * amp + (1 << 14)) >> 15
    return -val if angle & 2048 else val


def coil_value(phase, amplitude, swap_coils, shift=0):
    angle = ((phase & 0xFFFFFFFF) >> ANGLE_SHIFT) + shift
    sin_val = scaled_sine(angle, amplitude)
    cos_val = scaled_sine(angle + 1024, amplitude)
    if swap_coils:
        return (cos_val & 0x1FF) | ((sin_val & 0x1FF) << 16)
    return (sin_val & 0x1FF) | ((cos_val & 0x1FF) << 16)


def parse_correction(text):
    """Parse "harmonic:magnitude:phase" entries (radians of electrical
    angle) into {harmonic: (magnitude, phase)}."""
    corr = {}
    for item in text.replace(",", " ").split():
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError("expected harmonic:magnitude:phase, got %r" % item)
        harmonic = int(parts[0])
        if harmonic < 1:
            raise ValueError("harmonic must be at least 1")
        corr[harmonic] = (float(parts[1]), float(parts[2]))
    return corr


def format_correction(corr):
    return " ".join(
        "%d:%.5f:%.4f" % (h, mag, pha)
        for h, (mag, pha) in sorted(corr.items())
        if mag
    )


def correction_table(corr):
    """Angle shift per microstep counter position, in 1/4096 period."""
    scale = 4096.0 / (2.0 * math.pi)
    table = []
    for i in range(CORRECTION_SIZE):
        theta = 2.0 * math.pi * i / CORRECTION_SIZE
        shift = sum(
            mag * math.sin(h * theta + pha) for h, (mag, pha) in corr.items()
        )
        units = int(round(shift * scale))
        table.append(max(-CORRECTION_LIMIT, min(CORRECTION_LIMIT, units)))
    return table


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


def with_field(fields, field_name, value, reg_value):
    """Return reg_value with one field replaced.

    Unlike FieldHelper.set_field() this leaves the driver module's cached
    register values alone, so temporary settings cannot leak into later
    register writes made by the driver module.
    """
    reg_name = fields.lookup_register(field_name)
    mask = fields.all_fields[reg_name][field_name]
    return (reg_value & ~mask) | ((value << tmc.ffs(mask)) & mask)


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
    if fields.lookup_register("sg_result"):
        # StallGuard load readings in time order, to see whether they
        # stay usable for slip detection while the coils are driven directly
        sg = drv("sg_result")
        lines.append(
            "StallGuard sg_result %d-%d, median %d, in order %s"
            % (
                min(sg),
                max(sg),
                sorted(sg)[len(sg) // 2],
                " ".join(str(v) for v in sg[:16]),
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


class FlowControl:
    """Place queued entries on the MCU's tick grid.

    The MCU queue holds queue_size entries and an entry leaves it when the
    MCU starts on it, so entry i may only be sent once entry i - queue_size
    has started.
    """

    def __init__(self, queue_size, first_tick):
        self.queue_size = queue_size
        self.cursor = first_tick
        self.starts = collections.deque()

    def place(self, count, wait_tick):
        """Return (start tick, tick to send after or None) for an entry."""
        if not count:
            self.cursor = wait_tick
            start = wait_tick
        else:
            start = self.cursor
            self.cursor += count
        send_after = None
        if len(self.starts) >= self.queue_size:
            send_after = self.starts.popleft()
        self.starts.append(start)
        return start, send_after


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
        self.corrections = {}
        for direction in DIRECTIONS:
            option = "correction_" + direction
            try:
                self.corrections[direction] = parse_correction(
                    config.get(option, "")
                )
            except ValueError as e:
                raise config.error(
                    "Option '%s' in section '%s': %s"
                    % (option, config.get_name(), e)
                )
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
        self.queue_wait_cmd = self.set_correction_cmd = None
        self.queue_size = None
        self.auto_enable = config.getboolean("auto_enable", False)
        self.ffi_main, self.ffi_lib = chelper.get_ffi()
        self.pg = self.ffi_main.gc(
            self.ffi_lib.phasegen_alloc(), self.ffi_lib.phasegen_free
        )
        self.seg_buf = self.ffi_main.new("struct phasegen_seg[]", SEG_BUFFER)
        self.stepper = None
        self.active = False
        self.resume_after_homing = False
        self.run = None
        for event, handler in (
            ("klippy:connect", self._handle_connect),
            ("klippy:shutdown", self._handle_shutdown),
            ("homing:homing_move_begin", self._handle_homing_move_begin),
            ("homing:home_rails_end", self._handle_home_rails_end),
            ("stepper_enable:motor_off", self._handle_motor_off),
            ("toolhead:set_position", self._handle_set_position),
            ("stepper:set_dir_inverted", self._handle_dir_inverted),
        ):
            self.printer.register_event_handler(event, handler)
        self.gcode = self.printer.lookup_object("gcode")
        self.gcode.register_mux_command(
            "PHASE_STEPPING_SET_CORRECTION",
            "STEPPER",
            self.stepper_name,
            self.cmd_PHASE_STEPPING_SET_CORRECTION,
            desc=self.cmd_PHASE_STEPPING_SET_CORRECTION_help,
        )
        self.gcode.register_mux_command(
            "PHASE_STEPPING_ENABLE",
            "STEPPER",
            self.stepper_name,
            self.cmd_PHASE_STEPPING_ENABLE,
            desc=self.cmd_PHASE_STEPPING_ENABLE_help,
        )
        self.gcode.register_mux_command(
            "PHASE_STEPPING_TEST",
            "STEPPER",
            self.stepper_name,
            self.cmd_PHASE_STEPPING_TEST,
            desc=self.cmd_PHASE_STEPPING_TEST_help,
        )
        self.gcode.register_mux_command(
            "PHASE_STEPPING_STEP_TEST",
            "STEPPER",
            self.stepper_name,
            self.cmd_PHASE_STEPPING_STEP_TEST,
            desc=self.cmd_PHASE_STEPPING_STEP_TEST_help,
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
        self.queue_wait_cmd = self.mcu.lookup_command(
            "phase_stepper_queue_wait oid=%c clock=%u", cq=cq
        )
        self.set_correction_cmd = self.mcu.lookup_command(
            "phase_stepper_set_correction oid=%c dir=%c offset=%hu data=%*s",
            cq=cq,
        )
        self.queue_size = self.mcu.get_constants()["PHASE_STEPPER_QUEUE_SIZE"]
        self.stop_cmd = self.mcu.lookup_command(
            "phase_stepper_stop oid=%c", cq=cq
        )
        self.query_cmd = self.mcu.lookup_query_command(
            "phase_stepper_query oid=%c",
            "phase_stepper_state oid=%c phase=%u flags=%c pending=%c"
            " ticks=%u lost_ticks=%u locked_ticks=%u pair_timeouts=%u xfers=%u"
            " max_cycles=%u avg_cycles=%u tick_clock=%u",
            oid=self.oid,
            cq=cq,
        )

    def _handle_connect(self):
        toolhead = self.printer.lookup_object("toolhead")
        for stepper in toolhead.get_kinematics().get_steppers():
            if stepper.get_name() == self.stepper_name:
                self.stepper = stepper
        toolhead.register_step_generator(self._generate)

    def get_status(self, eventtime):
        return {
            "enabled": self.active,
            "correction_forward": format_correction(
                self.corrections["forward"]
            ),
            "correction_backward": format_correction(
                self.corrections["backward"]
            ),
        }

    def get_harmonic(self, direction, harmonic):
        mag, pha = self.corrections[direction].get(harmonic, (0.0, 0.0))
        return complex(mag * math.cos(pha), mag * math.sin(pha))

    def set_harmonic(self, direction, harmonic, value):
        mag = min(abs(value), CORRECTION_MAX)
        if mag:
            self.corrections[direction][harmonic] = (
                mag,
                math.atan2(value.imag, value.real),
            )
        else:
            self.corrections[direction].pop(harmonic, None)

    def apply_corrections(self):
        """Send changed corrections to the MCU (the motor must be at rest)."""
        if self.active:
            self.printer.lookup_object("toolhead").wait_moves()
            self._upload_corrections()

    def save_corrections(self):
        configfile = self.printer.lookup_object("configfile")
        section = "phase_stepping %s" % (self.stepper_name,)
        for d in DIRECTIONS:
            configfile.set(
                section,
                "correction_" + d,
                format_correction(self.corrections[d]),
            )

    def _upload_corrections(self):
        for index, direction in enumerate(DIRECTIONS):
            table = correction_table(self.corrections[direction])
            data = bytes(v & 0xFF for v in table)
            for offset in range(0, CORRECTION_SIZE, CORRECTION_CHUNK):
                self.set_correction_cmd.send(
                    [
                        self.oid,
                        index,
                        offset,
                        data[offset : offset + CORRECTION_CHUNK],
                    ]
                )

    # Switching the driver in and out of direct mode

    def _enter_direct(self, start_phase, amplitude):
        """Hand the coils to direct mode at start_phase; returns the driver
        settings to restore."""
        mcu_tmc = self.mcu_tmc
        fields = mcu_tmc.get_fields()
        # Use the values the driver module last wrote: IHOLD_IRUN is write
        # only, so reading it back from the driver returns zero.
        gconf = fields.registers.get("GCONF", 0)
        ihold_irun = fields.registers.get("IHOLD_IRUN")
        if ihold_irun is None or not fields.get_field("irun", ihold_irun):
            raise self.printer.command_error(
                "No run current known for '%s'" % (self.stepper_name,)
            )
        # Drivers scale direct mode currents by the hold current
        run_current = fields.get_field("irun", ihold_irun)
        mcu_tmc.set_register(
            "IHOLD_IRUN", with_field(fields, "ihold", run_current, ihold_irun)
        )
        mcu_tmc.tmc_spi.reg_write(
            self.direct_reg,
            coil_value(start_phase, amplitude, self.swap_coils),
            mcu_tmc.chain_pos,
        )
        # Without step pulses the driver sees standstill, where an enabled
        # stealthChop would take over from the coil currents.
        direct_gconf = with_field(fields, "en_pwm_mode", 0, gconf)
        direct_gconf = with_field(fields, "direct_mode", 1, direct_gconf)
        mcu_tmc.set_register("GCONF", direct_gconf)
        return gconf, ihold_irun

    def _leave_direct(self, saved):
        gconf, ihold_irun = saved
        self.mcu_tmc.set_register("GCONF", gconf)
        self.mcu_tmc.set_register("IHOLD_IRUN", ihold_irun)

    def _check_driver_ready(self):
        error = self.printer.command_error
        stepper_enable = self.printer.lookup_object("stepper_enable")
        enable_line = stepper_enable.lookup_enable(self.stepper_name)
        if not enable_line.is_motor_enabled():
            raise error("'%s' is not enabled" % (self.stepper_name,))
        mcu_tmc = self.mcu_tmc
        fields = mcu_tmc.get_fields()
        if fields.get_field("direct_mode", mcu_tmc.get_register("GCONF")):
            raise error("Driver is already in direct mode")
        mscnt = self._read_field("MSCNT", "mscnt")
        mscuract = mcu_tmc.get_register("MSCURACT")
        mismatch = check_coil_mapping(
            mscnt,
            fields.get_field("cur_a", mscuract),
            fields.get_field("cur_b", mscuract),
        )
        if mismatch is not None:
            raise error(
                "Driver microstep table is not a plain sine: %s" % (mismatch,)
            )
        return mscnt

    # Phase stepping during normal motion

    def _enable(self):
        if self.active:
            return
        toolhead = self.printer.lookup_object("toolhead")
        toolhead.wait_moves()
        stepper = self.stepper
        mscnt = self._check_driver_ready()
        period = motor_kinematics.stepper_electrical_period(
            self.printer, stepper
        )
        if period is None:
            raise self.printer.command_error(
                "Cannot find microsteps for '%s'" % (self.stepper_name,)
            )
        # The driver's microstep counter keeps counting the step pulses,
        # which run on unchanged; it counts up with the stepper position
        # when the direction pin is inverted (as tmc.py also assumes).
        sign = 1.0 if stepper.get_dir_inverted()[0] else -1.0
        start_phase = mscnt << MSCNT_SHIFT
        self._upload_corrections()
        saved = self._enter_direct(start_phase, self.amplitude)
        mcu = self.mcu
        try:
            est = mcu.estimated_print_time(
                self.printer.get_reactor().monotonic()
            )
            start_clock = mcu.print_time_to_clock(est + START_DELAY)
            self.start_cmd.send(
                [self.oid, start_clock & MASK32, start_phase, self.amplitude]
            )
            state = self.query_cmd.send([self.oid])
        except self.printer.command_error:
            self._leave_direct(saved)
            raise
        grid_clock = mcu.clock32_to_clock64(state["tick_clock"])
        tick_cycles = mcu.get_constants()["CLOCK_FREQ"] // self.tick_freq
        first_tick = -(-(start_clock - grid_clock) // tick_cycles)
        grid_time = mcu.clock_to_print_time(grid_clock)
        tick_time = (
            mcu.clock_to_print_time(grid_clock + tick_cycles) - grid_time
        )
        self.run = {
            "saved": saved,
            "grid_clock": grid_clock,
            "tick_cycles": tick_cycles,
            "tick_time": tick_time,
            "phase_per_mm": sign * PHASE_PER_PERIOD / period,
            "sk": None,
            "flow": FlowControl(self.queue_size, first_tick),
        }
        self._update_sk()
        self.ffi_lib.phasegen_start(
            self.pg, grid_time, first_tick - 1, start_phase, self._rest_pos()
        )
        self.active = True
        # Keep new moves from starting before the MCU takes over
        toolhead.dwell(START_DELAY + 0.050)

    def _update_sk(self):
        sk = self.stepper.get_stepper_kinematics()
        run = self.run
        if sk != run["sk"]:
            self.ffi_lib.phasegen_setup(
                self.pg, sk, run["tick_time"], run["phase_per_mm"], TOLERANCE
            )
            run["sk"] = sk

    def _generate(self, flush_time):
        if not self.active:
            return
        self._update_sk()
        num = self.ffi_lib.phasegen_generate(
            self.pg, flush_time, self.seg_buf, SEG_BUFFER
        )
        if num < 0:
            self.active = False
            self.printer.invoke_shutdown(
                "Phase stepping could not follow '%s': %s"
                % (self.stepper_name, self._describe_failure(num))
            )
            return
        run = self.run
        grid_clock, tick_cycles = run["grid_clock"], run["tick_cycles"]
        for i in range(num):
            entry = self.seg_buf[i]
            start, send_after = run["flow"].place(entry.count, entry.wait_tick)
            reqclock = grid_clock + start * tick_cycles
            minclock = 0
            if send_after is not None:
                minclock = grid_clock + (send_after + 1) * tick_cycles
            if entry.count:
                self.queue_move_cmd.send(
                    [self.oid, entry.count, entry.start_vel, entry.half_accel],
                    minclock=minclock,
                    reqclock=reqclock,
                )
            else:
                self.queue_wait_cmd.send(
                    [self.oid, reqclock & MASK32],
                    minclock=minclock,
                    reqclock=reqclock,
                )

    def _describe_failure(self, code):
        if code == PG_ERR_FULL:
            return "more than %d segments in one flush" % (SEG_BUFFER,)
        tick = self.ffi_lib.phasegen_get_last_tick(self.pg) + 1
        target = self.ffi_lib.phasegen_target(self.pg, tick)
        phase = self.ffi_lib.phasegen_get_phase(self.pg)
        jump = (target - phase) / self.run["phase_per_mm"]
        return "target %.3f mm from the motor near tick %d" % (jump, tick)

    def _disable(self, wait=True):
        """Return the driver to step mode. Returns how far (in microstep
        counter units) the driver's counter ended from the phase, or None
        when not checked."""
        if not self.active:
            return None
        if wait:
            self.printer.lookup_object("toolhead").wait_moves()
            reactor = self.printer.get_reactor()
            end = reactor.monotonic() + DRAIN_TIMEOUT
            while self.query_cmd.send([self.oid])["pending"]:
                if reactor.monotonic() > end:
                    break
                reactor.pause(reactor.monotonic() + QUERY_INTERVAL)
        self.active = False
        self.stop_cmd.send([self.oid])
        self._leave_direct(self.run["saved"])
        if not wait:
            return None
        phase = self.ffi_lib.phasegen_get_phase(self.pg)
        phase_index = (phase >> MSCNT_SHIFT) & 1023
        mscnt = self._read_field("MSCNT", "mscnt")
        return (mscnt - phase_index + 512) % 1024 - 512

    def _report_disagreement(self, diff, gcmd=None):
        if diff is None:
            return
        # Step pulses stop on the nearest microstep, the phase does not
        microsteps = motor_kinematics.stepper_microsteps(
            self.printer, self.stepper
        )
        allowed = 1024 // (4 * microsteps) // 2 + 1 if microsteps else 2
        if abs(diff) <= allowed:
            return
        msg = (
            "Phase stepping on '%s' ended %d microstep counter units away"
            " from the driver's step position" % (self.stepper_name, diff)
        )
        logging.warning(msg)
        if gcmd is not None:
            gcmd.respond_info("WARNING: " + msg)

    def _handle_shutdown(self):
        self.active = False

    def _handle_homing_move_begin(self, hmove):
        if not self.active:
            return
        for mcu_endstop in hmove.get_mcu_endstops():
            if self.stepper in mcu_endstop.get_steppers():
                self._report_disagreement(self._disable())
                self.resume_after_homing = True
                return

    def _handle_home_rails_end(self, homing_state, rails):
        if self.active:
            return
        if not (self.resume_after_homing or self.auto_enable):
            return
        for rail in rails:
            if self.stepper in rail.get_steppers():
                self.resume_after_homing = False
                self._enable()
                return

    def _handle_motor_off(self, print_time):
        self.resume_after_homing = False
        self._disable(wait=False)

    def _rest_pos(self):
        """Stepper position (in the frame of its kinematics, including any
        input shaper wrapper) at the toolhead's current resting position."""
        toolhead = self.printer.lookup_object("toolhead")
        return self.stepper.calc_position_from_coord(
            toolhead.get_position()[:3]
        )

    def _handle_set_position(self):
        if self.active:
            self.ffi_lib.phasegen_rebase(self.pg, self._rest_pos())

    def _handle_dir_inverted(self, stepper):
        if self.active and stepper is self.stepper:
            self._report_disagreement(self._disable())

    cmd_PHASE_STEPPING_SET_CORRECTION_help = (
        "Set the angle correction of one harmonic of a phase stepped motor"
    )

    def cmd_PHASE_STEPPING_SET_CORRECTION(self, gcmd):
        harmonic = gcmd.get_int("HARMONIC", None, minval=1, maxval=16)
        direction = gcmd.get("DIRECTION", "both").lower()
        if direction not in DIRECTIONS + ("both",):
            raise gcmd.error("DIRECTION must be forward, backward or both")
        if harmonic is not None:
            mag = gcmd.get_float("MAG", minval=0.0, maxval=0.19)
            pha = gcmd.get_float("PHASE", 0.0)
            for d in DIRECTIONS:
                if direction in (d, "both"):
                    if mag:
                        self.corrections[d][harmonic] = (mag, pha)
                    else:
                        self.corrections[d].pop(harmonic, None)
            if self.active:
                self.printer.lookup_object("toolhead").wait_moves()
                self._upload_corrections()
        if gcmd.get_int("SAVE", 0, minval=0, maxval=1):
            self.save_corrections()
        gcmd.respond_info(
            "%s correction forward: %s\nbackward: %s"
            % (
                self.stepper_name,
                format_correction(self.corrections["forward"]) or "none",
                format_correction(self.corrections["backward"]) or "none",
            )
        )

    cmd_PHASE_STEPPING_ENABLE_help = (
        "Enable or disable phase stepping for a stepper"
    )

    def cmd_PHASE_STEPPING_ENABLE(self, gcmd):
        if gcmd.get_int("ENABLE", 1, minval=0, maxval=1):
            self._enable()
            gcmd.respond_info(
                "Phase stepping enabled on %s" % (self.stepper_name,)
            )
            return
        self.resume_after_homing = False
        self._report_disagreement(self._disable(), gcmd)
        gcmd.respond_info(
            "Phase stepping disabled on %s" % (self.stepper_name,)
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
        if self.active:
            raise gcmd.error("Disable phase stepping first")
        distance = gcmd.get_float("DISTANCE", 10.0, above=0.0, maxval=100.0)
        velocity = gcmd.get_float("VELOCITY", 20.0, above=0.0, maxval=1000.0)
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
        # Use the values the driver module last wrote: IHOLD_IRUN is write
        # only, so reading it back from the driver returns zero.
        gconf = fields.registers.get("GCONF", 0)
        ihold_irun = fields.registers.get("IHOLD_IRUN")
        if ihold_irun is None or not fields.get_field("irun", ihold_irun):
            raise gcmd.error(
                "No run current known for '%s'" % (self.stepper_name,)
            )
        if fields.get_field("direct_mode", mcu_tmc.get_register("GCONF")):
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

        # Drivers scale direct mode currents by the hold current
        run_current = with_field(
            fields, "ihold", fields.get_field("irun", ihold_irun), ihold_irun
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
            direct_gconf = with_field(fields, "en_pwm_mode", 0, gconf)
            direct_gconf = with_field(fields, "direct_mode", 1, direct_gconf)
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
            "Ticks: %d, lost %d, skipped for other SPI traffic %d,"
            " interrupted driver reads %d"
            % (
                state["ticks"],
                state["lost_ticks"],
                state["locked_ticks"],
                state["pair_timeouts"],
            ),
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
        if state["pair_timeouts"]:
            warnings.append(
                "%d driver reads were interrupted by coil writes, so driver"
                " values read during the test may be wrong"
                % (state["pair_timeouts"],)
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

    cmd_PHASE_STEPPING_STEP_TEST_help = (
        "Check how the driver's microstep counter follows step pulses in"
        " normal and in direct mode"
    )

    def cmd_PHASE_STEPPING_STEP_TEST(self, gcmd):
        if self.active:
            raise gcmd.error("Disable phase stepping first")
        toolhead = self.printer.lookup_object("toolhead")
        toolhead.wait_moves()
        kin, stepper = self._lookup_stepper(gcmd)
        stepper_enable = self.printer.lookup_object("stepper_enable")
        if not stepper_enable.lookup_enable(
            self.stepper_name
        ).is_motor_enabled():
            raise gcmd.error("'%s' is not enabled" % (self.stepper_name,))
        period = motor_kinematics.stepper_electrical_period(
            self.printer, stepper
        )
        if period is None:
            raise gcmd.error(
                "Cannot find microsteps for '%s'" % (self.stepper_name,)
            )
        # A whole number of periods plus a quarter: big enough to see, and
        # the counter (1024 per period) ends 256 away from where it began.
        distance = gcmd.get_float(
            "DISTANCE", 6.25 * period, above=0.0, maxval=20.0
        )
        expected = 1024.0 * distance / period
        force_move = self.printer.lookup_object("force_move")
        mcu_tmc = self.mcu_tmc
        fields = mcu_tmc.get_fields()

        def mscnt():
            return self._read_field("MSCNT", "mscnt")

        def move(dist):
            force_move.manual_move(stepper, dist, STEP_TEST_SPEED)
            toolhead.wait_moves()

        def wrap(delta):
            return (delta + 512) % 1024 - 512

        start = mscnt()
        move(distance)
        after_step = mscnt()
        move(-distance)
        back_step = mscnt()

        gconf = fields.registers.get("GCONF", 0)
        ihold_irun = fields.registers.get("IHOLD_IRUN", 0)
        run_current = with_field(
            fields, "ihold", fields.get_field("irun", ihold_irun), ihold_irun
        )
        direct_gconf = with_field(fields, "en_pwm_mode", 0, gconf)
        direct_gconf = with_field(fields, "direct_mode", 1, direct_gconf)
        completed = False
        try:
            mcu_tmc.set_register("IHOLD_IRUN", run_current)
            mcu_tmc.tmc_spi.reg_write(
                self.direct_reg,
                coil_value(
                    back_step << MSCNT_SHIFT, self.amplitude, self.swap_coils
                ),
                mcu_tmc.chain_pos,
            )
            mcu_tmc.set_register("GCONF", direct_gconf)
            move(distance)
            after_direct = mscnt()
            move(-distance)
            back_direct = mscnt()
            completed = True
        finally:
            mcu_tmc.set_register("GCONF", gconf)
            mcu_tmc.set_register("IHOLD_IRUN", ihold_irun)
            if not completed:
                stepper_enable.motor_off()

        expected_mod = wrap(int(round(expected)))
        gcmd.respond_info(
            "Step test on %s: %.3f mm forward and back (%.0f counter units)\n"
            "Normal mode: counter %d -> %d -> %d (forward %+d, expected %+d"
            " for this direction)\n"
            "Direct mode: counter %d -> %d -> %d (forward %+d)\n"
            "The motor should move in normal mode and stay still in direct"
            " mode; tell which you saw."
            % (
                self.stepper_name,
                distance,
                expected,
                start,
                after_step,
                back_step,
                wrap(after_step - start),
                expected_mod,
                back_step,
                after_direct,
                back_direct,
                wrap(after_direct - back_step),
            )
        )


def load_config_prefix(config):
    return PhaseStepping(config)
