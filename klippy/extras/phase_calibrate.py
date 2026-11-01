# Measure stepper motor harmonics with an accelerometer
#
# Copyright (C) 2026  Åsmund Collin <aakjaergaard@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import cmath
import contextlib
import logging
import math
import os
import subprocess
import sys
import tempfile
import time

from . import motor_kinematics, resonance_tester

REPOSITION_DWELL = 0.3
# Speed resolution of a calibration sweep
SWEEP_STEP = 2.5
# Speed step of a calibration at constant speeds (RAMP=0)
PASS_STEP = 5.0
GRAPH_SCRIPT = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "scripts",
        "graph_motor_harmonics.py",
    )
)
# Matches phase_stepping.CORRECTION_MAX
CORRECTION_LIMIT_RAD = 127 * 2.0 * math.pi / 4096.0
DIRECTIONS = {1: "forward", -1: "backward"}
HARMONIC_NAMES = {
    1: "1 per electrical period",
    2: "1 per 2 full steps",
    4: "1 per full step",
}
# A harmonic is calibrated when it reaches this displacement (mm) and this
# many times its noise level somewhere in the speed range
MIN_CAL_DISP = 1e-4
MIN_CAL_SNR = 5.0


class PhaseCalibrate:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.accel_chip_name = config.get("accel_chip").strip()
        self.point = config.getfloatlist("point", None, count=3)
        self.speed_min = config.getfloat("speed_min", 30.0, above=0.0)
        self.speed_max = config.getfloat(
            "speed_max", 160.0, minval=self.speed_min
        )
        self.speed_step = config.getfloat("speed_step", 5.0, above=0.0)
        self.accel = config.getfloat("accel", 2000.0, above=0.0)
        self.ramp_accel = config.getfloat("ramp_accel", 50.0, above=0.0)
        self.settle_time = config.getfloat("settle_time", 0.1, minval=0.0)
        self.measure_time = config.getfloat("measure_time", 0.5, minval=0.1)
        self.max_distance = config.getfloat("max_distance", 150.0, above=0.0)
        self.move_speed = config.getfloat("move_speed", 50.0, above=0.0)
        self.belt_pitch = config.getfloat("belt_pitch", 2.0, minval=0.0)
        self.harmonics = config.getintlist("harmonics", [1, 2, 3, 4])
        if not self.harmonics or min(self.harmonics) < 1:
            raise config.error(
                "Option 'harmonics' in section '%s' must list positive"
                " integers" % (config.get_name(),)
            )
        self.graph = config.getboolean("graph", False)
        self.output_dir = os.path.expanduser(
            config.get("output_dir", tempfile.gettempdir())
        )
        if not os.path.isdir(self.output_dir):
            raise config.error(
                "Option 'output_dir' in section '%s' is not a directory"
                % (config.get_name(),)
            )
        self.accel_chip = None
        self.printer.register_event_handler(
            "klippy:connect", self._handle_connect
        )
        self.gcode = self.printer.lookup_object("gcode")
        self.gcode.register_command(
            "PHASE_STEPPING_CALIBRATE",
            self.cmd_PHASE_STEPPING_CALIBRATE,
            desc=self.cmd_PHASE_STEPPING_CALIBRATE_help,
            params=self.cmd_PHASE_STEPPING_CALIBRATE_params,
        )
        self.gcode.register_command(
            "MEASURE_MOTOR_HARMONICS",
            self.cmd_MEASURE_MOTOR_HARMONICS,
            desc=self.cmd_MEASURE_MOTOR_HARMONICS_help,
            params=self.cmd_MEASURE_MOTOR_HARMONICS_params,
        )

    def _handle_connect(self):
        chip = self.printer.lookup_object(self.accel_chip_name, None)
        if chip is None or not hasattr(chip, "start_internal_client"):
            raise self.printer.config_error(
                "'%s' is not an accelerometer" % (self.accel_chip_name,)
            )
        self.accel_chip = chip

    def _lookup_stepper(self, gcmd, kin):
        steppers = {s.get_name(): s for s in kin.get_steppers()}
        name = gcmd.get("STEPPER")
        if name not in steppers:
            raise gcmd.error(
                "Unknown STEPPER '%s', expected one of: %s"
                % (name, ", ".join(sorted(steppers)))
            )
        return steppers[name]

    def _get_point(self, gcmd, toolhead):
        raw = gcmd.get("POINT", None)
        if raw is not None:
            try:
                point = [float(p.strip()) for p in raw.split(",")]
            except ValueError:
                point = []
            if len(point) != 3:
                raise gcmd.error("POINT must be in the form x,y,z")
            return point
        if self.point is not None:
            return list(self.point)
        # The middle of the bed leaves the most room for the test moves
        status = toolhead.get_status(self.printer.get_reactor().monotonic())
        lo, hi = status["axis_minimum"], status["axis_maximum"]
        return [
            0.5 * (lo[0] + hi[0]),
            0.5 * (lo[1] + hi[1]),
            toolhead.get_position()[2],
        ]

    def _get_harmonics(self, gcmd):
        raw = gcmd.get("HARMONICS", None)
        if raw is None:
            return list(self.harmonics)
        try:
            harmonics = [int(h.strip()) for h in raw.split(",")]
        except ValueError:
            harmonics = []
        if not harmonics or min(harmonics) < 1:
            raise gcmd.error("HARMONICS must list positive integers")
        return harmonics

    def _check_travel(self, gcmd, toolhead, ends):
        status = toolhead.get_status(self.printer.get_reactor().monotonic())
        homed = status["homed_axes"]
        if "x" not in homed or "y" not in homed:
            raise gcmd.error("Must home X and Y first")
        axis_min, axis_max = status["axis_minimum"], status["axis_maximum"]
        for end in ends:
            for axis in (0, 1):
                if not axis_min[axis] <= end[axis] <= axis_max[axis]:
                    raise gcmd.error(
                        "Test move to (%.1f, %.1f) is outside the axis"
                        " limits; choose another POINT or lower SPEED_MAX"
                        % (end[0], end[1])
                    )

    def _suspend_limits(self, test):
        """Test limits and no input shaping for a whole command, so that
        its many test runs don't each announce the change."""
        scale = test["scale"]
        return resonance_tester.suspend_limits(
            self.printer,
            test["accel"] * scale,
            test["max_speed"] * scale,
            False,
        )

    @contextlib.contextmanager
    def _uncorrected(self, phase_stepping, harmonics, speeds):
        """Clear the corrections of the given harmonics and set the speed
        range of the corrections, and restore both if calibration fails."""
        previous = {
            (label, h): phase_stepping.get_harmonic(label, h)
            for label in DIRECTIONS.values()
            for h in harmonics
        }
        previous_speeds = phase_stepping.get_correction_speeds()
        for label, h in previous:
            phase_stepping.set_harmonic(label, h, 0j)
        phase_stepping.set_correction_speeds(speeds)
        phase_stepping.apply_corrections()
        try:
            yield
        except BaseException:
            phase_stepping.set_correction_speeds(previous_speeds)
            for (label, h), value in previous.items():
                phase_stepping.set_harmonic(label, h, value)
            try:
                phase_stepping.apply_corrections()
            except Exception:
                logging.exception(
                    "Unable to restore phase stepping corrections"
                )
            raise

    def _motor_geometry(self, gcmd, kin, stepper, point):
        """(toolhead move per mm of motor travel, electrical period)"""
        try:
            direction = motor_kinematics.stepper_move_direction(
                kin, stepper, point
            )
        except ValueError as e:
            raise gcmd.error("%s: %s" % (gcmd.get_command(), e))
        period = motor_kinematics.stepper_electrical_period(
            self.printer, stepper
        )
        if period is None:
            raise gcmd.error(
                "Cannot find microsteps for '%s'" % (stepper.get_name(),)
            )
        return direction, period

    def _ends(self, point, direction, half):
        return {
            sign: [
                point[0] + sign * half * direction[0],
                point[1] + sign * half * direction[1],
                point[2],
            ]
            for sign in (-1, 1)
        }

    def _setup_test(self, gcmd, kin, stepper, point, speeds, accel):
        """Plan constant speed passes of one motor around point."""
        from . import motor_harmonics

        direction, period = self._motor_geometry(gcmd, kin, stepper, point)
        cruise_time = self.settle_time + self.measure_time
        travel = motor_harmonics.motor_travel(max(speeds), accel, cruise_time)
        if travel > self.max_distance:
            raise gcmd.error(
                "The test needs %.1f mm of motor travel at %.0f mm/s, more"
                " than max_distance (%.1f mm); lower the speed or raise"
                " ACCEL" % (travel, max(speeds), self.max_distance)
            )
        plan = []
        for speed in speeds:
            half = 0.5 * motor_harmonics.motor_travel(speed, accel, cruise_time)
            ends = self._ends(point, direction, half)
            plan.append((speed, half, ends[-1], ends[1]))
        toolhead = self.printer.lookup_object("toolhead")
        self._check_travel(gcmd, toolhead, max(plan)[2:])
        return {
            "kind": "passes",
            "point": point,
            "direction": direction,
            "scale": math.hypot(*direction),
            "period": period,
            "belt": motor_harmonics.belt_order(period, self.belt_pitch),
            "plan": plan,
            "speeds": list(speeds),
            "speed_step": 0.0,
            "accel": accel,
            "max_speed": max(speeds),
        }

    def _setup_ramps(self, gcmd, kin, stepper, point, speeds, step, accels):
        """Plan speed ramps of one motor around point covering speeds, which
        are then measured every step; accels are (to reach and leave a
        ramp, along the ramp)."""
        from . import motor_harmonics

        direction, period = self._motor_geometry(gcmd, kin, stepper, point)
        accel = accels[0]
        try:
            segments = motor_harmonics.plan_ramps(
                speeds[0],
                speeds[-1],
                period,
                accels,
                self.max_distance,
                self.settle_time,
            )
        except ValueError as e:
            raise gcmd.error("%s: %s" % (gcmd.get_command(), e))
        travel = max(
            motor_harmonics.ramp_travel(v1, v2, ramp_accel, accel)
            for v1, v2, ramp_accel in segments
        )
        toolhead = self.printer.lookup_object("toolhead")
        self._check_travel(
            gcmd, toolhead, self._ends(point, direction, 0.5 * travel).values()
        )
        return {
            "kind": "ramps",
            "point": point,
            "direction": direction,
            "scale": math.hypot(*direction),
            "period": period,
            "belt": motor_harmonics.belt_order(period, self.belt_pitch),
            "segments": segments,
            "speeds": list(speeds),
            "speed_step": step,
            "accel": accel,
            "max_speed": segments[-1][1],
        }

    def _run_passes(self, test):
        """Move at each planned speed in both directions; returns the
        accelerometer client and the measuring window of each pass."""
        from . import motor_harmonics

        toolhead = self.printer.lookup_object("toolhead")
        point, scale, accel = test["point"], test["scale"], test["accel"]
        settle = self.settle_time
        windows = []
        toolhead.manual_move(point, self.move_speed)
        toolhead.wait_moves()
        aclient = self.accel_chip.start_internal_client()
        try:
            for speed, half, start, end in test["plan"]:
                toolhead.manual_move(start, self.move_speed)
                toolhead.dwell(REPOSITION_DWELL)
                # Every test move starts from rest (a dwell or a full
                # reversal precedes it), so the cruise phase begins exactly
                # speed/accel after the move start.
                cruise = speed * speed / (2.0 * accel) + speed * settle
                for motor_dir, target in ((1, end), (-1, start)):
                    start_time = toolhead.get_last_move_time()
                    toolhead.manual_move(target, speed * scale)
                    windows.append(
                        motor_harmonics.Ramp(
                            motor_dir,
                            start_time + speed / accel + settle,
                            speed,
                            0.0,
                            self.measure_time,
                            -motor_dir * (half - cruise),
                        )
                    )
            toolhead.manual_move(point, self.move_speed)
        finally:
            aclient.finish_measurements()
        return aclient, windows

    def _run_ramps(self, test):
        """Ramp up and down through each segment in both directions;
        returns the accelerometer client and the ramps."""
        from . import motor_harmonics

        toolhead = self.printer.lookup_object("toolhead")
        point, direction = test["point"], test["direction"]
        scale, accel = test["scale"], test["accel"]
        settle = self.settle_time

        def set_accel(value):
            # A move keeps the acceleration limit it was queued with
            toolhead.set_accel(value * scale)

        ramps = []
        toolhead.manual_move(point, self.move_speed)
        toolhead.wait_moves()
        aclient = self.accel_chip.start_internal_client()
        try:
            for v1, v2, ramp_accel in test["segments"]:
                ramp_d = abs(v2 * v2 - v1 * v1) / (2.0 * ramp_accel)
                travel = motor_harmonics.ramp_travel(v1, v2, ramp_accel, accel)
                ends = self._ends(point, direction, 0.5 * travel)
                toolhead.manual_move(ends[-1], self.move_speed)
                for motor_dir, up in ((1, 1), (-1, 1), (1, 0), (-1, 0)):
                    first, last = (v1, v2) if up else (v2, v1)
                    # Reach the ramp's first speed, ramp, then stop; the
                    # move junctions keep the speed between the parts.
                    legs = (
                        (first * first / (2.0 * accel), first, accel),
                        (ramp_d, v2, ramp_accel),
                        (last * last / (2.0 * accel), last, accel),
                    )
                    toolhead.dwell(REPOSITION_DWELL)
                    start_time = toolhead.get_last_move_time()
                    start = ends[-motor_dir]
                    done = 0.0
                    for length, speed, leg_accel in legs:
                        if length < 1e-6:
                            continue
                        done += length
                        set_accel(leg_accel)
                        toolhead.manual_move(
                            [
                                start[i] + motor_dir * done * direction[i]
                                for i in (0, 1)
                            ]
                            + [point[2]],
                            speed * scale,
                        )
                    set_accel(accel)
                    # Skip the ringing from reaching the ramp's first speed
                    sign = 1.0 if up else -1.0
                    duration = (v2 - v1) / ramp_accel - settle
                    if duration <= 0.0:
                        continue
                    covered = (
                        first * first / (2.0 * accel)
                        + first * settle
                        + 0.5 * sign * ramp_accel * settle * settle
                    )
                    ramps.append(
                        motor_harmonics.Ramp(
                            motor_dir,
                            start_time + first / accel + settle,
                            first + sign * ramp_accel * settle,
                            sign * ramp_accel,
                            duration,
                            motor_dir * (covered - 0.5 * travel),
                        )
                    )
            toolhead.manual_move(point, self.move_speed)
        finally:
            aclient.finish_measurements()
        return aclient, ramps

    def _run(self, gcmd, test):
        """Run the test moves once; returns (sample times, acceleration
        along the motion, ramps)."""
        from . import motor_harmonics

        if test["kind"] == "ramps":
            aclient, ramps = self._run_ramps(test)
        else:
            aclient, ramps = self._run_passes(test)
        if not aclient.has_valid_samples():
            raise gcmd.error(
                "accelerometer '%s' measured no data" % (self.accel_chip_name,)
            )
        direction, scale = test["direction"], test["scale"]
        times, values = motor_harmonics.project_samples(
            aclient.get_samples(), direction[0] / scale, direction[1] / scale
        )
        return times, values, ramps

    def _measure(self, gcmd, test, harmonics, max_freq_ratio, passes=1):
        """Run the test moves passes times and measure the harmonics at
        every planned speed; returns sorted (speed, direction, result)
        rows, averaged over ramps and passes."""
        from . import motor_harmonics

        rows = []
        for _ in range(passes):
            times, values, ramps = self._run(gcmd, test)
            for ramp in ramps:
                rows.extend(
                    motor_harmonics.analyze_ramp(
                        times,
                        values,
                        ramp,
                        test["period"],
                        harmonics,
                        test["speeds"],
                        test["speed_step"],
                        test["belt"],
                        max_freq_ratio,
                    )
                )
        rows = motor_harmonics.merge_rows(rows)
        rows.sort(key=lambda row: (row[0], -row[1], row[2].harmonic))
        return rows

    def _sweep_table(self, rows):
        """{(direction, harmonic): ({speed: phasor}, {speed: noise})} of
        the measured rows."""
        table = {}
        for speed, motor_dir, res in rows:
            if res.phasor is None:
                continue
            phasors, noises = table.setdefault(
                (motor_dir, res.harmonic), ({}, {})
            )
            phasors[speed] = res.phasor
            noises[speed] = res.noise_disp
        return table

    def _write_csv(self, filename, stepper_name, period, rows):
        with open(filename, "w") as f:
            f.write(
                "# stepper=%s electrical_period=%.6f chip=%s\n"
                % (stepper_name, period, self.accel_chip_name)
            )
            f.write(
                "speed,direction,harmonic,freq,accel,disp_um,noise_um,phase\n"
            )
            for speed, motor_dir, res in rows:
                if res.disp is None:
                    continue
                noise = (
                    ""
                    if res.noise_disp is None
                    else "%.5f" % (res.noise_disp * 1000.0,)
                )
                phase = (
                    ""
                    if res.phasor is None
                    else "%.4f" % cmath.phase(res.phasor)
                )
                f.write(
                    "%.3f,%s,%d,%.3f,%.5f,%.5f,%s,%s\n"
                    % (
                        speed,
                        "fwd" if motor_dir > 0 else "bwd",
                        res.harmonic,
                        res.freq,
                        res.accel,
                        res.disp * 1000.0,
                        noise,
                        phase,
                    )
                )

    def _get_name(self, gcmd):
        name = gcmd.get("NAME", time.strftime("%Y%m%d_%H%M%S"))
        if not name.replace("-", "").replace("_", "").isalnum():
            raise gcmd.error("Invalid NAME parameter")
        return name

    def _csv_path(self, stepper_name, name):
        return os.path.join(
            self.output_dir,
            "motor_harmonics_%s_%s.csv" % (stepper_name, name),
        )

    def _graph(self, csv_files):
        """Plot each CSV to a PNG beside it with the standalone graph
        script, in a child process so matplotlib neither loads into nor
        blocks klippy. Returns the lines to report."""
        reactor = self.printer.get_reactor()
        lines = []
        for csv_file in csv_files:
            png = os.path.splitext(csv_file)[0] + ".png"
            proc = subprocess.Popen(
                [sys.executable, GRAPH_SCRIPT, csv_file, "-o", png],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            eventtime = reactor.monotonic()
            while proc.poll() is None:
                eventtime = reactor.pause(eventtime + 0.1)
            err = proc.stderr.read().decode(errors="replace").strip()
            proc.stderr.close()
            if proc.returncode:
                lines.append(
                    "Graph of %s failed: %s"
                    % (csv_file, err.splitlines()[-1] if err else "unknown")
                )
            else:
                lines.append("Graph written to %s" % (png,))
        return lines

    def _format_peak(self, row):
        if row is None:
            return "not above noise"
        speed, _, res = row
        text = "%.3g um at %.0f mm/s (%.0f Hz" % (
            res.disp * 1000.0,
            speed,
            res.freq,
        )
        if res.noise_disp:
            text += ", %.1fx noise" % (res.disp / res.noise_disp,)
        return text + ")"

    cmd_MEASURE_MOTOR_HARMONICS_help = (
        "Measure the vibration of one stepper motor at harmonics of its"
        " electrical frequency"
    )
    cmd_MEASURE_MOTOR_HARMONICS_params = {
        "STEPPER": {"type": "string", "required": True},
        "SPEED_MIN": {"type": "float", "required": False},
        "SPEED_MAX": {"type": "float", "required": False},
        "SPEED_STEP": {"type": "float", "required": False},
        "ACCEL": {"type": "float", "required": False},
        "RAMP": {"type": "int", "default": 1},
        "RAMP_ACCEL": {"type": "float", "required": False},
        "GRAPH": {"type": "int", "required": False},
        # Read by the _get_*() helpers, not directly in this method
        "HARMONICS": {"type": "string", "required": False},
        "NAME": {"type": "string", "required": False},
        "POINT": {"type": "string", "required": False},
    }

    def cmd_MEASURE_MOTOR_HARMONICS(self, gcmd):
        try:
            from . import motor_harmonics
        except ImportError:
            raise gcmd.error(
                "MEASURE_MOTOR_HARMONICS requires numpy; install it in the"
                " klippy environment"
            )
        toolhead = self.printer.lookup_object("toolhead")
        kin = toolhead.get_kinematics()
        stepper = self._lookup_stepper(gcmd, kin)
        speed_min = gcmd.get_float("SPEED_MIN", self.speed_min, above=0.0)
        speed_max = gcmd.get_float(
            "SPEED_MAX", max(self.speed_max, speed_min), minval=speed_min
        )
        speed_step = gcmd.get_float("SPEED_STEP", self.speed_step, above=0.0)
        accel = gcmd.get_float("ACCEL", self.accel, above=0.0)
        ramp = gcmd.get_int("RAMP", 1, minval=0, maxval=1)
        ramp_accel = gcmd.get_float("RAMP_ACCEL", self.ramp_accel, above=0.0)
        harmonics = self._get_harmonics(gcmd)
        name = self._get_name(gcmd)
        graph = gcmd.get_int("GRAPH", int(self.graph), minval=0, maxval=1)
        point = self._get_point(gcmd, toolhead)
        speeds = motor_harmonics.plan_speeds(speed_min, speed_max, speed_step)
        if ramp:
            test = self._setup_ramps(
                gcmd,
                kin,
                stepper,
                point,
                speeds,
                speed_step,
                (accel, ramp_accel),
            )
            how = "%d speed ramps" % (len(test["segments"]),)
        else:
            test = self._setup_test(gcmd, kin, stepper, point, speeds, accel)
            how = "%d speeds" % (len(speeds),)
        period = test["period"]
        gcmd.respond_info(
            "Measuring %s: electrical period %.4f mm, %s from %.0f to"
            " %.0f mm/s"
            % (stepper.get_name(), period, how, speeds[0], speeds[-1])
        )
        with self._suspend_limits(test):
            rows = self._measure(
                gcmd, test, harmonics, motor_harmonics.MAX_FREQ_RATIO
            )

        filename = self._csv_path(stepper.get_name(), name)
        self._write_csv(filename, stepper.get_name(), period, rows)
        peaks = motor_harmonics.peak_per_harmonic(rows)
        lines = ["Peak vibration per harmonic of %s:" % (stepper.get_name(),)]
        for harmonic in harmonics:
            label = HARMONIC_NAMES.get(harmonic)
            lines.append(
                "H%d%s: fwd %s, bwd %s"
                % (
                    harmonic,
                    " (%s)" % (label,) if label else "",
                    self._format_peak(peaks.get((harmonic, 1))),
                    self._format_peak(peaks.get((harmonic, -1))),
                )
            )
        lines.append("Data written to %s" % (filename,))
        if graph:
            lines.extend(self._graph([filename]))
        gcmd.respond_info("\n".join(lines))

    cmd_PHASE_STEPPING_CALIBRATE_help = (
        "Find the angle corrections that cancel a phase stepped motor's"
        " harmonics"
    )
    cmd_PHASE_STEPPING_CALIBRATE_params = {
        "STEPPER": {"type": "string", "required": True},
        "HARMONICS": {"type": "string", "default": "2,4,1"},
        "ACCEL": {"type": "float", "required": False},
        "PASSES": {"type": "int", "default": 2},
        "PROBE": {"type": "float", "default": 0.02},
        "SPEED_MIN": {"type": "float", "required": False},
        "SPEED_MAX": {"type": "float", "required": False},
        "SPEED_COUNT": {"type": "int", "default": 5},
        "RAMP": {"type": "int", "default": 1},
        "RAMP_ACCEL": {"type": "float", "required": False},
        "GRAPH": {"type": "int", "required": False},
        "SAVE": {"type": "int", "default": 1},
        # Read by the _get_*() helpers, not directly in this method
        "NAME": {"type": "string", "required": False},
        "POINT": {"type": "string", "required": False},
    }

    def cmd_PHASE_STEPPING_CALIBRATE(self, gcmd):
        try:
            from . import motor_harmonics, phase_fit
        except ImportError:
            raise gcmd.error("PHASE_STEPPING_CALIBRATE requires numpy")
        toolhead = self.printer.lookup_object("toolhead")
        kin = toolhead.get_kinematics()
        stepper = self._lookup_stepper(gcmd, kin)
        name = stepper.get_name()
        phase_stepping = self.printer.lookup_object(
            "phase_stepping %s" % (name,), None
        )
        if phase_stepping is None or not phase_stepping.active:
            raise gcmd.error(
                "Enable phase stepping on '%s' first (PHASE_STEPPING_ENABLE)"
                % (name,)
            )
        harmonics = self._get_harmonics(gcmd)
        accel = gcmd.get_float("ACCEL", self.accel, above=0.0)
        passes = gcmd.get_int("PASSES", 1, minval=1, maxval=5)
        probe = gcmd.get_float("PROBE", 0.0, minval=0.0, maxval=0.1)
        iterations = gcmd.get_int("ITERATIONS", 2, minval=0, maxval=5)
        speed_dependent = gcmd.get_int("SPEED_DEPENDENT", 1, minval=0, maxval=1)
        speed_min = gcmd.get_float("SPEED_MIN", self.speed_min, above=0.0)
        speed_max = gcmd.get_float(
            "SPEED_MAX", max(self.speed_max, speed_min), minval=speed_min
        )
        ramp = gcmd.get_int("RAMP", 1, minval=0, maxval=1)
        ramp_accel = gcmd.get_float("RAMP_ACCEL", self.ramp_accel, above=0.0)
        graph = gcmd.get_int("GRAPH", int(self.graph), minval=0, maxval=1)
        graph_name = self._get_name(gcmd)
        point = self._get_point(gcmd, toolhead)
        limit = CORRECTION_LIMIT_RAD

        if ramp:
            speeds = motor_harmonics.plan_speeds(
                speed_min, speed_max, SWEEP_STEP
            )
            test = self._setup_ramps(
                gcmd,
                kin,
                stepper,
                point,
                speeds,
                SWEEP_STEP,
                (accel, ramp_accel),
            )
            how = "%d speed ramps" % (len(test["segments"]),)
        else:
            speeds = motor_harmonics.plan_speeds(
                speed_min, speed_max, PASS_STEP
            )
            test = self._setup_test(gcmd, kin, stepper, point, speeds, accel)
            how = "%d constant speeds" % (len(speeds),)
        period = test["period"]
        reactor = self.printer.get_reactor()
        started = reactor.monotonic()
        sweeps = []

        def sweep(what):
            gcmd.respond_info(
                "Sweep %d of about %d: %s"
                % (len(sweeps) + 1, 6 + iterations, what)
            )
            rows = self._measure(
                gcmd,
                test,
                harmonics,
                motor_harmonics.CALIBRATION_FREQ_RATIO,
                passes,
            )
            sweeps.append(rows)
            return self._sweep_table(rows)

        def apply(corrections):
            for (motor_dir, harmonic), c in corrections.items():
                phase_stepping.set_harmonic(DIRECTIONS[motor_dir], harmonic, c)
            phase_stepping.apply_corrections()

        def describe(key):
            return "H%d %s" % (key[1], DIRECTIONS[key[0]])

        def describe_correction(correction):
            slow, fast = phase_fit.as_pair(correction)
            text = "%.4f rad at %.2f rad" % (abs(slow), cmath.phase(slow))
            if fast != slow:
                text += " at %.0f mm/s, %.4f rad at %.2f rad at %.0f mm/s" % (
                    speed_min,
                    abs(fast),
                    cmath.phase(fast),
                    speed_max,
                )
            return text

        gcmd.respond_info(
            "Calibrating %s over %.0f-%.0f mm/s with %s%s"
            % (
                name,
                speed_min,
                speed_max,
                how,
                ", %d passes each" % (passes,) if passes > 1 else "",
            )
        )
        # Calibrate from no correction so that the measurements show the
        # motor's own error; the probe and refinement sweeps then determine
        # the response to each harmonic's correction at every speed.
        speeds = (speed_min, speed_max) if speed_max > speed_min else None
        uncorrected = self._uncorrected(phase_stepping, harmonics, speeds)
        with uncorrected, self._suspend_limits(test):
            base = sweep("no correction")
            models, skipped = {}, {}
            for harmonic in harmonics:
                for motor_dir in DIRECTIONS:
                    key = (motor_dir, harmonic)
                    phasors, noises = base.get(key, ({}, {}))
                    strong = any(
                        abs(p)
                        >= max(
                            MIN_CAL_DISP, MIN_CAL_SNR * (noises.get(s) or 0.0)
                        )
                        for s, p in phasors.items()
                        if speed_min <= s <= speed_max
                    )
                    if not phasors:
                        skipped[key] = "not measurable in this speed range"
                    elif not strong:
                        skipped[key] = "below noise, left uncorrected"
                    else:
                        model = phase_fit.HarmonicResponse(
                            speed_min, speed_max, limit
                        )
                        model.add(0j, phasors, noises)
                        models[key] = model
            if not models:
                raise gcmd.error(
                    "No harmonic of %s is measurably above noise between"
                    " %.0f and %.0f mm/s" % (name, speed_min, speed_max)
                )
            probes = {
                key: probe or phase_fit.probe_size(base[key][0], period, limit)
                for key in models
            }
            for setting in phase_fit.probe_corrections(probes, limit):
                apply(setting)
                table = sweep(
                    "probing "
                    + ", ".join(
                        "%s %+.3f%s"
                        % (
                            describe(key),
                            c.imag or c.real,
                            "j" if c.imag else "",
                        )
                        for key, c in setting.items()
                    )
                )
                for key, model in models.items():
                    model.add(setting[key], *table.get(key, ({}, {})))
            applied = dict.fromkeys(models, 0j)
            apply(applied)
            for it in range(iterations + 1):
                proposal = {}
                for key, model in models.items():
                    if not model.fit():
                        continue
                    candidate = model.solve(
                        speed_dependent=bool(speed_dependent)
                    )
                    if model.improves(candidate, applied[key]):
                        proposal[key] = candidate
                if not proposal:
                    break
                applied.update(proposal)
                apply(applied)
                table = sweep(
                    ("verifying " if it == iterations else "refining ")
                    + ", ".join(describe(key) for key in proposal)
                )
                for key, model in models.items():
                    model.add(applied[key], *table.get(key, ({}, {})))
            # Keep a correction only where it measurably beats none
            final = {}
            lines = ["Phase stepping calibration of %s:" % (name,)]
            for key in sorted(models, key=lambda k: (k[1], -k[0])):
                model = models[key]
                before = model.measured(0j)
                after = model.measured(applied[key])
                noise = dict(zip(model.speeds, model.noise))
                worst_b, worst_a, avg_b, avg_a, worse = phase_fit.compare(
                    before, after, noise
                )
                if not any(abs(c) for c in phase_fit.as_pair(applied[key])):
                    final[key] = 0j
                    lines.append(
                        "%s over %.0f-%.0f mm/s: worst %.3f um, noise %.3f um;"
                        " no correction should help, left uncorrected"
                        % (
                            describe(key),
                            speed_min,
                            speed_max,
                            worst_b * 1000.0,
                            model.noise_floor() * 1000.0,
                        )
                    )
                    continue
                if worst_a >= worst_b:
                    final[key] = 0j
                    lines.append(
                        "%s over %.0f-%.0f mm/s: the correction found left"
                        " %.3f um against %.3f um without, left uncorrected"
                        % (
                            describe(key),
                            speed_min,
                            speed_max,
                            worst_a * 1000.0,
                            worst_b * 1000.0,
                        )
                    )
                    continue
                final[key] = applied[key]
                line = (
                    "%s over %.0f-%.0f mm/s: worst %.3f -> %.3f um, average"
                    " %.3f -> %.3f um, noise %.3f um (correction %s)"
                    % (
                        describe(key),
                        speed_min,
                        speed_max,
                        worst_b * 1000.0,
                        worst_a * 1000.0,
                        avg_b * 1000.0,
                        avg_a * 1000.0,
                        model.noise_floor() * 1000.0,
                        describe_correction(applied[key]),
                    )
                )
                if worse:
                    line += "; worse at %s mm/s" % (
                        ", ".join("%.1f" % s for s in worse),
                    )
                lines.append(line)
            for key in sorted(skipped, key=lambda k: (k[1], -k[0])):
                lines.append("%s: %s" % (describe(key), skipped[key]))
            if final != applied:
                apply(final)
        lines.append(
            "%d sweeps in %.0f s" % (len(sweeps), reactor.monotonic() - started)
        )
        csv_files = [
            self._csv_path(name, graph_name),
            self._csv_path(name, graph_name + "_calibrated"),
        ]
        self._write_csv(csv_files[0], name, period, sweeps[0])
        self._write_csv(csv_files[1], name, period, sweeps[-1])
        lines.append("Sweeps written to %s and %s" % tuple(csv_files))
        if gcmd.get_int("SAVE", 1, minval=0, maxval=1):
            phase_stepping.save_corrections()
            lines.append(
                "Corrections stored; run SAVE_CONFIG to keep them across"
                " restarts"
            )
        if graph:
            lines.extend(self._graph(csv_files))
        gcmd.respond_info("\n".join(lines))


def load_config(config):
    return PhaseCalibrate(config)
