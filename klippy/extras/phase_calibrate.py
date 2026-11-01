# Measure stepper motor harmonics with an accelerometer
#
# Copyright (C) 2026  Åsmund Collin <aakjaergaard@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import math
import os
import tempfile
import time

from . import motor_kinematics, resonance_tester

REPOSITION_DWELL = 0.3
# Matches phase_stepping.CORRECTION_MAX
CORRECTION_LIMIT_RAD = 127 * 2.0 * math.pi / 4096.0
HARMONIC_NAMES = {
    1: "1 per electrical period",
    2: "1 per 2 full steps",
    4: "1 per full step",
}


class PhaseCalibrate:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.accel_chip_name = config.get("accel_chip").strip()
        self.point = config.getfloatlist("point", None, count=3)
        self.speed_min = config.getfloat("speed_min", 10.0, above=0.0)
        self.speed_max = config.getfloat(
            "speed_max", 160.0, minval=self.speed_min
        )
        self.speed_step = config.getfloat("speed_step", 5.0, above=0.0)
        self.accel = config.getfloat("accel", 2000.0, above=0.0)
        self.settle_time = config.getfloat("settle_time", 0.1, minval=0.0)
        self.measure_time = config.getfloat("measure_time", 0.5, minval=0.1)
        self.max_distance = config.getfloat("max_distance", 150.0, above=0.0)
        self.move_speed = config.getfloat("move_speed", 50.0, above=0.0)
        self.harmonics = config.getintlist("harmonics", (1, 2, 3, 4))
        if not self.harmonics or min(self.harmonics) < 1:
            raise config.error(
                "Option 'harmonics' in section '%s' must list positive"
                " integers" % (config.get_name(),)
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
        )
        self.gcode.register_command(
            "MEASURE_MOTOR_HARMONICS",
            self.cmd_MEASURE_MOTOR_HARMONICS,
            desc=self.cmd_MEASURE_MOTOR_HARMONICS_help,
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
        return toolhead.get_position()[:3]

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
            max(p[0] for p in test["plan"]) * scale,
            False,
        )

    def _run_moves(self, toolhead, point, plan, scale, accel):
        windows = []
        toolhead.manual_move(point, self.move_speed)
        toolhead.wait_moves()
        aclient = self.accel_chip.start_internal_client()
        try:
            for speed, start, end in plan:
                toolhead.manual_move(start, self.move_speed)
                toolhead.dwell(REPOSITION_DWELL)
                for motor_dir, target in ((1, end), (-1, start)):
                    # Every test move starts from rest (a dwell or a full
                    # reversal precedes it), so the cruise phase begins
                    # exactly speed/accel after the move start.
                    start_time = toolhead.get_last_move_time()
                    toolhead.manual_move(target, speed * scale)
                    begin = start_time + speed / accel + self.settle_time
                    windows.append(
                        (speed, motor_dir, begin, begin + self.measure_time)
                    )
            toolhead.manual_move(point, self.move_speed)
        finally:
            aclient.finish_measurements()
        return aclient, windows

    def _setup_test(self, gcmd, kin, stepper, point, speeds, accel):
        """Plan constant speed passes of one motor around point."""
        from . import motor_harmonics

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
            start, end = [
                [
                    point[0] + sign * half * direction[0],
                    point[1] + sign * half * direction[1],
                    point[2],
                ]
                for sign in (-1.0, 1.0)
            ]
            plan.append((speed, start, end))
        toolhead = self.printer.lookup_object("toolhead")
        self._check_travel(gcmd, toolhead, max(plan)[1:])
        return {
            "point": point,
            "direction": direction,
            "scale": math.hypot(*direction),
            "period": period,
            "plan": plan,
            "accel": accel,
        }

    def _run_test(self, gcmd, test, plan=None):
        """Run passes and return (times, accel along the motion, windows)."""
        from . import motor_harmonics

        toolhead = self.printer.lookup_object("toolhead")
        aclient, windows = self._run_moves(
            toolhead,
            test["point"],
            plan or test["plan"],
            test["scale"],
            test["accel"],
        )
        if not aclient.has_valid_samples():
            raise gcmd.error(
                "accelerometer '%s' measured no data" % (self.accel_chip_name,)
            )
        direction, scale = test["direction"], test["scale"]
        times, values = motor_harmonics.project_samples(
            aclient.get_samples(), direction[0] / scale, direction[1] / scale
        )
        return times, values, windows

    def _measure_rows(self, gcmd, test, harmonics):
        from . import motor_harmonics

        times, values, windows = self._run_test(gcmd, test)
        rows = []
        for speed, motor_dir, begin, end in windows:
            sl = motor_harmonics.window_slice(times, begin, end)
            for res in motor_harmonics.analyze_window(
                times[sl], values[sl], speed, test["period"], harmonics
            ):
                rows.append((speed, motor_dir, res))
        return rows

    def _measure_coherent(self, gcmd, test, speeds, harmonic, passes):
        """Average complex displacement (mm) of one harmonic per direction
        and speed, in phase with the motor's electrical angle; returns
        {direction: {speed: displacement}}."""
        from . import motor_harmonics

        plan = [p for p in test["plan"] if p[0] in speeds] * passes
        times, values, windows = self._run_test(gcmd, test, plan)
        sums = {}
        for speed, motor_dir, begin, end in windows:
            omega = 2.0 * math.pi * harmonic * speed / test["period"]
            sl = motor_harmonics.window_slice(times, begin, end)
            # Every pass at a speed starts from the same point, so the
            # motor's angle at the window start is the same and the phases
            # are comparable.
            amp = motor_harmonics.harmonic_amplitude(
                times[sl],
                values[sl],
                motor_dir * omega / (2.0 * math.pi),
                t_ref=begin,
            )
            sums.setdefault(motor_dir, {}).setdefault(speed, []).append(
                amp / omega**2
            )
        return {
            d: {s: sum(v) / len(v) for s, v in by_speed.items()}
            for d, by_speed in sums.items()
        }

    def _write_csv(self, filename, stepper_name, period, rows):
        with open(filename, "w") as f:
            f.write(
                "# stepper=%s electrical_period=%.6f chip=%s\n"
                % (stepper_name, period, self.accel_chip_name)
            )
            f.write("speed,direction,harmonic,freq,accel,disp_um,noise_um\n")
            for speed, motor_dir, res in rows:
                if res.disp is None:
                    continue
                noise = (
                    ""
                    if res.noise_disp is None
                    else "%.5f" % (res.noise_disp * 1000.0,)
                )
                f.write(
                    "%.3f,%s,%d,%.3f,%.5f,%.5f,%s\n"
                    % (
                        speed,
                        "fwd" if motor_dir > 0 else "bwd",
                        res.harmonic,
                        res.freq,
                        res.accel,
                        res.disp * 1000.0,
                        noise,
                    )
                )

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
        harmonics = self._get_harmonics(gcmd)
        name = gcmd.get("NAME", time.strftime("%Y%m%d_%H%M%S"))
        if not name.replace("-", "").replace("_", "").isalnum():
            raise gcmd.error("Invalid NAME parameter")
        point = self._get_point(gcmd, toolhead)
        speeds = motor_harmonics.plan_speeds(speed_min, speed_max, speed_step)
        test = self._setup_test(gcmd, kin, stepper, point, speeds, accel)
        period = test["period"]
        gcmd.respond_info(
            "Measuring %s: electrical period %.4f mm, %d speeds from %.0f to"
            " %.0f mm/s"
            % (stepper.get_name(), period, len(speeds), speeds[0], speeds[-1])
        )
        with self._suspend_limits(test):
            rows = self._measure_rows(gcmd, test, harmonics)

        filename = os.path.join(
            tempfile.gettempdir(),
            "motor_harmonics_%s_%s.csv" % (stepper.get_name(), name),
        )
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
        gcmd.respond_info("\n".join(lines))

    cmd_PHASE_STEPPING_CALIBRATE_help = (
        "Find the angle corrections that cancel a phase stepped motor's"
        " harmonics"
    )

    def cmd_PHASE_STEPPING_CALIBRATE(self, gcmd):
        try:
            from . import motor_harmonics
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
        harmonics = [
            int(h) for h in gcmd.get("HARMONICS", "2,4,1").split(",") if h
        ]
        accel = gcmd.get_float("ACCEL", self.accel, above=0.0)
        passes = gcmd.get_int("PASSES", 2, minval=1, maxval=10)
        probe = gcmd.get_float("PROBE", 0.02, above=0.0, maxval=0.1)
        speed_min = gcmd.get_float("SPEED_MIN", self.speed_min, above=0.0)
        speed_max = gcmd.get_float(
            "SPEED_MAX", max(self.speed_max, speed_min), minval=speed_min
        )
        speed_count = gcmd.get_int("SPEED_COUNT", 5, minval=1, maxval=10)
        point = self._get_point(gcmd, toolhead)

        # A motor's error changes with speed (its load angle grows), so each
        # harmonic is measured at several speeds where it is clearly seen
        # and gets the one correction that does best across all of them.
        speeds = motor_harmonics.plan_speeds(speed_min, speed_max, 5.0)
        test = self._setup_test(gcmd, kin, stepper, point, speeds, accel)
        with self._suspend_limits(test):
            gcmd.respond_info("Finding calibration speeds for %s" % (name,))
            rows = self._measure_rows(gcmd, test, harmonics)
            period = test["period"]
            directions = {1: "forward", -1: "backward"}
            lines = ["Phase stepping calibration of %s:" % (name,)]
            for harmonic in harmonics:
                cal_speeds = motor_harmonics.pick_calibration_speeds(
                    rows, harmonic, speed_count
                )
                if not cal_speeds:
                    lines.append(
                        "H%d: not measurably above noise" % (harmonic,)
                    )
                    continue

                def measure(corr):
                    for d, label in directions.items():
                        phase_stepping.set_harmonic(label, harmonic, corr[d])
                    phase_stepping.apply_corrections()
                    return self._measure_coherent(
                        gcmd, test, cal_speeds, harmonic, passes
                    )

                gcmd.respond_info(
                    "Calibrating H%d at %s mm/s"
                    % (harmonic, ", ".join("%.1f" % s for s in cal_speeds))
                )
                c0 = {
                    d: phase_stepping.get_harmonic(label, harmonic)
                    for d, label in directions.items()
                }
                d0 = measure(c0)
                d_real = measure({d: c0[d] + probe for d in directions})
                d_imag = measure({d: c0[d] + 1j * probe for d in directions})
                new, gains, weights, best, best_vib, best_cost = (
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                )
                for d in directions:
                    new[d], gains[d], weights[d] = (
                        motor_harmonics.multi_speed_correction(
                            d0[d], d_real[d], d_imag[d], c0[d], probe
                        )
                    )
                    best[d], best_vib[d] = c0[d], d0[d]
                    best_cost[d] = motor_harmonics.weighted_cost(
                        d0[d], weights[d]
                    )
                for _ in range(2):
                    vib = measure(new)
                    for d in directions:
                        cost = motor_harmonics.weighted_cost(vib[d], weights[d])
                        if cost < best_cost[d]:
                            best[d], best_vib[d], best_cost[d] = (
                                new[d],
                                vib[d],
                                cost,
                            )
                        new[d] = motor_harmonics.refine_correction(
                            best[d], best_vib[d], gains[d], weights[d]
                        )
                for d, label in directions.items():
                    phase_stepping.set_harmonic(label, harmonic, best[d])
                phase_stepping.apply_corrections()
                for d, label in directions.items():
                    before = [abs(v) * 1000.0 for v in d0[d].values()]
                    after = [abs(v) * 1000.0 for v in best_vib[d].values()]
                    lines.append(
                        "H%d %s over %.0f-%.0f mm/s: worst %.3f -> %.3f um,"
                        " average %.3f -> %.3f um (correction %.4f rad at %.2f"
                        " rad)"
                        % (
                            harmonic,
                            label,
                            cal_speeds[0],
                            cal_speeds[-1],
                            max(before),
                            max(after),
                            sum(before) / len(before),
                            sum(after) / len(after),
                            min(abs(best[d]), CORRECTION_LIMIT_RAD),
                            math.atan2(best[d].imag, best[d].real),
                        )
                    )
        if gcmd.get_int("SAVE", 1, minval=0, maxval=1):
            phase_stepping.save_corrections()
            lines.append(
                "Corrections stored; run SAVE_CONFIG to keep them across"
                " restarts"
            )
        gcmd.respond_info("\n".join(lines))


def load_config(config):
    return PhaseCalibrate(config)
