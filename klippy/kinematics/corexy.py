# Code for handling the kinematics of corexy robots
#
# Copyright (C) 2017-2021  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging
import math

from klippy import stepper
from klippy.extras import endstop_phase, homing


class CoreXYKinematics:
    def __init__(self, toolhead, config):
        # Setup axis rails
        self.rails = [
            stepper.LookupMultiRail(config.getsection("stepper_" + n))
            for n in "xyz"
        ]
        for s in self.rails[1].get_steppers():
            self.rails[0].get_endstops()[0][0].add_stepper(s)
        for s in self.rails[0].get_steppers():
            self.rails[1].get_endstops()[0][0].add_stepper(s)
        self.rails[0].setup_itersolve("corexy_stepper_alloc", b"+")
        self.rails[1].setup_itersolve("corexy_stepper_alloc", b"-")
        self.rails[2].setup_itersolve("cartesian_stepper_alloc", b"z")
        for s in self.get_steppers():
            s.set_trapq(toolhead.get_trapq())
        # Setup boundary checks
        max_velocity, max_accel = toolhead.get_max_velocity()
        self.max_z_velocity = config.getfloat(
            "max_z_velocity", max_velocity, above=0.0, maxval=max_velocity
        )
        self.max_z_accel = config.getfloat(
            "max_z_accel", max_accel, above=0.0, maxval=max_accel
        )
        self.limits = [(1.0, -1.0)] * 3
        ranges = [r.get_range() for r in self.rails]
        self.axes_min = toolhead.Coord(*[r[0] for r in ranges], e=0.0)
        self.axes_max = toolhead.Coord(*[r[1] for r in ranges], e=0.0)
        self.supports_dual_carriage = False
        self.precise_homing = None
        if config.has_section("corexy_precise_homing"):
            self.precise_homing = CoreXYPreciseHoming(
                config.getsection("corexy_precise_homing"), toolhead, self
            )

    def get_steppers(self):
        return [s for rail in self.rails for s in rail.get_steppers()]

    def calc_position(self, stepper_positions):
        pos = [stepper_positions[rail.get_name()] for rail in self.rails]
        return [0.5 * (pos[0] + pos[1]), 0.5 * (pos[0] - pos[1]), pos[2]]

    def set_position(self, newpos, homing_axes):
        for i, rail in enumerate(self.rails):
            rail.set_position(newpos)
            if "xyz"[i] in homing_axes:
                self.limits[i] = rail.get_range()

    def note_z_not_homed(self):
        self.clear_homing_state("z")

    def clear_homing_state(self, clear_axes):
        for axis, axis_name in enumerate("xyz"):
            if axis_name in clear_axes:
                self.limits[axis] = (1.0, -1.0)

    def home(self, homing_state):
        # Each axis is homed independently and in order
        for axis in homing_state.get_axes():
            rail = self.rails[axis]
            # Determine movement
            position_min, position_max = rail.get_range()
            hi = rail.get_homing_info()
            homepos = [None, None, None, None]
            homepos[axis] = hi.position_endstop
            forcepos = list(homepos)
            if hi.positive_dir:
                forcepos[axis] -= 1.5 * (hi.position_endstop - position_min)
            else:
                forcepos[axis] += 1.5 * (position_max - hi.position_endstop)
            # Perform homing
            homing_state.home_rails([rail], forcepos, homepos)
        if self.precise_homing is not None:
            self.precise_homing.refine(homing_state)

    def _check_endstops(self, move):
        end_pos = move.end_pos
        for i in (0, 1, 2):
            if move.axes_d[i] and (
                end_pos[i] < self.limits[i][0] or end_pos[i] > self.limits[i][1]
            ):
                if self.limits[i][0] > self.limits[i][1]:
                    raise move.move_error("Must home axis first")
                raise move.move_error()

    def check_move(self, move):
        limits = self.limits
        xpos, ypos = move.end_pos[:2]
        if (
            xpos < limits[0][0]
            or xpos > limits[0][1]
            or ypos < limits[1][0]
            or ypos > limits[1][1]
        ):
            self._check_endstops(move)
        if not move.axes_d[2]:
            # Normal XY move - use defaults
            return
        # Move with Z - update velocity and accel for slower Z axis
        self._check_endstops(move)
        z_ratio = move.move_d / abs(move.axes_d[2])
        move.limit_speed(
            self.max_z_velocity * z_ratio, self.max_z_accel * z_ratio
        )

    def get_status(self, eventtime):
        axes = [a for a, (l, h) in zip("xyz", self.limits) if l <= h]
        return {
            "homed_axes": "".join(axes),
            "axis_minimum": self.axes_min,
            "axis_maximum": self.axes_max,
        }


# Refine a sensorless XY home by bumping the X and Y hard stops with a
# single motor. One motor alone moves the toolhead along a diagonal, so
# it reaches the two stops separately. A motor stalled against a hard
# stop rests on a detent of its electrical cycle, so the distance from
# a cycle-aligned start point to each stop is an integer number of
# cycles plus a per-machine constant (the "origin"). Rounding the
# measured distances onto that grid removes the trigger jitter of the
# regular sensorless homing. The method follows the CoreXY precise
# homing of the Prusa Buddy firmware.
UNSTABLE_THRESHOLD = 0.25


class CoreXYPreciseHoming:
    def __init__(self, config, toolhead, kin):
        self.printer = config.get_printer()
        self.toolhead = toolhead
        self.kin = kin
        self.gcode = self.printer.lookup_object("gcode")
        self.rails = kin.rails[:2]
        self.steppers = [rail.get_steppers()[0] for rail in self.rails]
        for rail in self.rails:
            name = rail.get_name()
            if not rail.get_homing_info().use_sensorless_homing:
                raise config.error(
                    "corexy_precise_homing requires sensorless homing on %s"
                    % (name,)
                )
            if config.has_section("endstop_phase %s" % (name,)):
                raise config.error(
                    "corexy_precise_homing can not be combined with"
                    " [endstop_phase %s]" % (name,)
                )
        hi_x, hi_y = [rail.get_homing_info() for rail in self.rails]
        self.home_dir = [[-1.0, 1.0][hi.positive_dir] for hi in (hi_x, hi_y)]
        # Motor A moves the toolhead along x=y, motor B along x=-y. Pick
        # the one whose diagonal meets the X stop and the Y stop at two
        # different places instead of running into the home corner.
        self.moved_rail = 1 if hi_x.positive_dir == hi_y.positive_dir else 0
        self.moved_hi = self.rails[self.moved_rail].get_homing_info()
        self.speed = config.getfloat("speed", self.moved_hi.speed, above=0.0)
        self.backoff = config.getfloat("backoff_distance", 5.0, above=0.0)
        for rail in self.rails:
            position_min, position_max = rail.get_range()
            if position_max - position_min < 4.0 * self.backoff:
                raise config.error(
                    "corexy_precise_homing backoff_distance is too large"
                    " for the travel of %s" % (rail.get_name(),)
                )
        self.samples = config.getint("samples", 2, minval=1)
        self.retries = config.getint("retries", 3, minval=0)
        self.phases = [
            config.getsection(s.get_name()).getint(
                "microsteps", note_valid=False
            )
            * 4
            for s in self.steppers
        ]
        self.step_dists = [s.get_step_dist() for s in self.steppers]
        self.cycle_dists = [
            p * sd for p, sd in zip(self.phases, self.step_dists)
        ]
        full_step = self.cycle_dists[self.moved_rail] / 4.0
        self.tolerance = config.getfloat("tolerance", full_step, minval=0.0)
        self.origin = config.getfloatlist("origin", None, count=2)
        self.phase_calcs = [
            endstop_phase.PhaseCalc(self.printer, s.get_name(), p)
            for s, p in zip(self.steppers, self.phases)
        ]
        self.is_simulated = (
            self.printer.get_start_args().get("debugoutput") is not None
        )
        self.printer.register_event_handler(
            "klippy:connect", self._handle_connect
        )
        self.gcode.register_command(
            "COREXY_PRECISE_HOMING_CALIBRATE",
            self.cmd_COREXY_PRECISE_HOMING_CALIBRATE,
            desc=self.cmd_COREXY_PRECISE_HOMING_CALIBRATE_help,
            params=self.cmd_COREXY_PRECISE_HOMING_CALIBRATE_params,
        )

    def _handle_connect(self):
        for phase_calc in self.phase_calcs:
            phase_calc.lookup_tmc()
            if phase_calc.tmc_module is None:
                raise self.printer.config_error(
                    "corexy_precise_homing: no TMC driver found for %s"
                    % (phase_calc.name,)
                )

    def refine(self, homing_state):
        if not any(axis in (0, 1) for axis in homing_state.get_axes()):
            return
        if any(low > high for low, high in self.kin.limits[:2]):
            return
        grid, ref = self._measure(homing_state, self.origin)
        if self.origin is None:
            self.origin = [g - round(g) for g in grid]
            self.gcode.respond_info(
                "corexy_precise_homing: using origin %.3f, %.3f from this"
                " homing until restart. Run COREXY_PRECISE_HOMING_CALIBRATE"
                " to store a calibrated origin." % tuple(self.origin)
            )
        self._apply(grid, ref)

    def _start_position(self):
        corners = [
            rail.get_homing_info().position_endstop for rail in self.rails
        ]
        start = [c - d * self.backoff for c, d in zip(corners, self.home_dir)]
        return corners, start

    def _bump(self, target):
        rail = self.rails[self.moved_rail]
        hmove = homing.HomingMove(self.printer, rail.get_endstops())
        hmove.homing_move(target, self.speed, probe_pos=True)
        name = self.steppers[self.moved_rail].get_name()
        for sp in hmove.stepper_positions:
            if sp.stepper_name == name:
                if sp.trig_pos == sp.start_pos:
                    raise self.printer.command_error(
                        "corexy_precise_homing: %s stalled before moving"
                        % (name,)
                    )
                return sp.trig_pos - sp.start_pos
        raise self.printer.command_error(
            "corexy_precise_homing: no trigger position for %s" % (name,)
        )

    def _measure(self, homing_state, origin):
        # Returns the corner position in electrical cycles of each
        # motor ("grid"), counted from the last cycle boundary before
        # the start point, and the start point's own distance from
        # that boundary ("ref"), also in cycles.
        toolhead = self.toolhead
        dx, dy = self.home_dir
        travel = 2.0 * self.backoff
        _, start = self._start_position()
        toolhead.manual_move(
            [start[0], start[1], None], self.moved_hi.retract_speed
        )
        # Shift the coordinate frame away from the stops for the
        # duration of the measurement, so that the bump targets beyond
        # the endstops pass the kinematic limit checks.
        pos = toolhead.get_position()
        fake_start = [
            start[0] - dx * self.backoff,
            start[1] - dy * self.backoff,
        ]
        toolhead.set_position(fake_start + pos[2:])
        ref = [
            pc.calc_phase(s, s.get_mcu_position()) * sd / cycle
            for pc, s, sd, cycle in zip(
                self.phase_calcs,
                self.steppers,
                self.step_dists,
                self.cycle_dists,
            )
        ]
        targets = [
            [fake_start[0] + dx * travel, fake_start[1] - dy * travel, pos[2]],
            [fake_start[0] - dx * travel, fake_start[1] + dy * travel, pos[2]],
        ]
        hi = self.moved_hi
        try:
            homing_state.set_homing_accel(hi.accel, pre_homing=True)
            homing_state.set_homing_current("xy", pre_homing=True)
            try:
                grid = self._collect(
                    homing_state, targets, fake_start, ref, origin
                )
            finally:
                homing_state.set_homing_current("xy", pre_homing=False)
                homing_state.set_homing_accel(hi.accel, pre_homing=False)
        except Exception:
            # The toolhead position is unknown, so force a new homing
            self.kin.clear_homing_state("xy")
            raise
        toolhead.set_position(start + pos[2:])
        return grid, ref

    def _collect(self, homing_state, targets, fake_start, ref, origin):
        toolhead = self.toolhead
        endstops = self.rails[self.moved_rail].get_endstops()
        step_dist = self.step_dists[self.moved_rail]
        retract_speed = self.moved_hi.retract_speed
        for attempt in range(self.retries + 1):
            steps = [[], []]
            for i in range(self.samples):
                for stop in (0, 1):
                    homing_state.reset_endstop_states(endstops)
                    steps[stop].append(self._bump(targets[stop]))
                    toolhead.manual_move(
                        [fake_start[0], fake_start[1], None], retract_speed
                    )
            spreads = [(max(s) - min(s)) * step_dist for s in steps]
            if max(spreads) > self.tolerance:
                self.gcode.respond_info(
                    "corexy_precise_homing: sample spread %.4f, %.4f mm"
                    " exceeds tolerance, retrying" % tuple(spreads)
                )
                continue
            means = [sum(s) / len(s) * step_dist for s in steps]
            grid = self._to_grid(means, ref)
            if origin is not None and self._is_unstable(grid, origin):
                self.gcode.respond_info(
                    "corexy_precise_homing: measurement %.3f, %.3f lies"
                    " between grid nodes, retrying" % tuple(grid)
                )
                continue
            logging.info(
                "corexy_precise_homing: steps %s grid %.4f, %.4f"
                " ref %.4f, %.4f",
                steps,
                grid[0],
                grid[1],
                ref[0],
                ref[1],
            )
            return grid
        raise self.printer.command_error(
            "corexy_precise_homing failed after %d attempts"
            % (self.retries + 1,)
        )

    def _to_grid(self, means, ref):
        # Motor A moves the toolhead by (dA/2, dA/2) and motor B by
        # (dB/2, -dB/2); the moved motor travelled means[0] to the X
        # stop and means[1] to the Y stop.
        ex = 0.5 * means[0]
        ey = 0.5 * means[1]
        if self.moved_rail == 1:
            ey = -ey
        corner_ab = [ex + ey, ex - ey]
        return [
            c / cycle + r
            for c, cycle, r in zip(corner_ab, self.cycle_dists, ref)
        ]

    def _is_unstable(self, grid, origin):
        if self.is_simulated:
            return False
        for g, o in zip(grid, origin):
            deviation = g - o
            deviation -= round(deviation)
            if abs(deviation) > UNSTABLE_THRESHOLD:
                return True
        return False

    def _apply(self, grid, ref):
        nodes = [round(g - o) for g, o in zip(grid, self.origin)]
        corner_ab = [
            (n - r) * cycle for n, r, cycle in zip(nodes, ref, self.cycle_dists)
        ]
        ex = 0.5 * (corner_ab[0] + corner_ab[1])
        ey = 0.5 * (corner_ab[0] - corner_ab[1])
        corners, start = self._start_position()
        newpos = [corners[0] - ex, corners[1] - ey]
        pos = self.toolhead.get_position()
        self.toolhead.set_position(newpos + pos[2:])
        self.gcode.respond_info(
            "corexy_precise_homing: adjusted X by %+.4f mm, Y by %+.4f mm"
            % (newpos[0] - start[0], newpos[1] - start[1])
        )

    def _mean_origin(self, grids):
        origin = []
        spread = 0.0
        for axis in range(2):
            angles = [2.0 * math.pi * grid[axis] for grid in grids]
            mean_cos = sum(math.cos(a) for a in angles) / len(angles)
            mean_sin = sum(math.sin(a) for a in angles) / len(angles)
            mean = math.atan2(mean_sin, mean_cos) / (2.0 * math.pi)
            for grid in grids:
                deviation = grid[axis] - mean
                deviation -= round(deviation)
                spread = max(spread, abs(deviation))
            origin.append(mean)
        return origin, spread

    cmd_COREXY_PRECISE_HOMING_CALIBRATE_help = (
        "Measure the grid origin of the corexy precise homing"
    )
    cmd_COREXY_PRECISE_HOMING_CALIBRATE_params = {
        "SAMPLES": {"type": "int", "default": 3},
    }

    def cmd_COREXY_PRECISE_HOMING_CALIBRATE(self, gcmd):
        samples = gcmd.get_int("SAMPLES", 3, minval=1)
        if any(low > high for low, high in self.kin.limits[:2]):
            raise gcmd.error("Must home X and Y first")
        homing_state = homing.Homing(self.printer)
        grids = []
        for i in range(samples):
            grid, ref = self._measure(homing_state, None)
            grids.append(grid)
        origin, spread = self._mean_origin(grids)
        if spread > UNSTABLE_THRESHOLD:
            raise gcmd.error(
                "corexy_precise_homing: measurements differ by %.3f cycles,"
                " check the stallguard tuning and try again" % (spread,)
            )
        self.origin = origin
        self._apply(grids[-1], ref)
        configfile = self.printer.lookup_object("configfile")
        configfile.set(
            "corexy_precise_homing", "origin", "%.4f, %.4f" % tuple(origin)
        )
        gcmd.respond_info(
            "corexy_precise_homing: origin %.4f, %.4f (spread %.3f cycles)\n"
            "The SAVE_CONFIG command will update the printer config file\n"
            "with this value and restart the printer."
            % (origin[0], origin[1], spread)
        )


def load_kinematics(toolhead, config):
    return CoreXYKinematics(toolhead, config)
