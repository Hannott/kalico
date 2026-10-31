# Helpers relating stepper motor positions to toolhead moves
#
# Copyright (C) 2026  Åsmund Collin <aakjaergaard@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import math

FULL_STEPS_PER_ELECTRICAL_PERIOD = 4
# Toolhead offset (mm) used to probe how a motor's position depends on X/Y.
JACOBIAN_PROBE = 1.0
LINEARITY_TOL = 1e-6
# Relative tolerance when deciding whether two motor axes are parallel or a
# solved direction moves another motor.
DIRECTION_TOL = 1e-6


def electrical_period(rotation_distance, full_steps):
    """Motor travel (mm) of one electrical period.

    full_steps is the number of motor full steps per rotation_distance,
    including any gear ratio.
    """
    return FULL_STEPS_PER_ELECTRICAL_PERIOD * rotation_distance / full_steps


def stepper_electrical_period(printer, stepper):
    """Electrical period (mm) of a configured stepper, or None if its
    microsteps setting cannot be found."""
    rotation_dist, steps_per_rotation = stepper.get_rotation_distance()
    configfile = printer.lookup_object("configfile")
    settings = configfile.get_status(None)["settings"]
    microsteps = settings.get(stepper.get_name(), {}).get("microsteps")
    if not microsteps:
        return None
    return electrical_period(rotation_dist, steps_per_rotation / microsteps)


def motor_jacobian(stepper, point):
    """Return (d_motor/dx, d_motor/dy) at point, or None if the motor
    position is not linear in X and Y."""

    def pos(dx, dy):
        return stepper.calc_position_from_coord(
            (point[0] + dx, point[1] + dy, point[2])
        )

    p0 = pos(0.0, 0.0)
    jx = pos(JACOBIAN_PROBE, 0.0) - p0
    jy = pos(0.0, JACOBIAN_PROBE) - p0
    if (
        abs(pos(2.0 * JACOBIAN_PROBE, 0.0) - p0 - 2.0 * jx) > LINEARITY_TOL
        or abs(pos(0.0, 2.0 * JACOBIAN_PROBE) - p0 - 2.0 * jy) > LINEARITY_TOL
    ):
        return None
    return jx / JACOBIAN_PROBE, jy / JACOBIAN_PROBE


def single_motor_direction(target, others):
    """Return the toolhead XY move per 1 mm of target motor travel that
    keeps every other XY motor still, or None if there is no such move.

    target and others are (d_motor/dx, d_motor/dy) pairs. Motors parallel to
    the target (for example a second motor on the same axis) necessarily
    move with it and are ignored, as are motors that do not move in XY.
    """
    tx, ty = target
    t_norm = math.hypot(tx, ty)
    if t_norm == 0.0:
        return None
    constraints = []
    for ox, oy in others:
        o_norm = math.hypot(ox, oy)
        if o_norm == 0.0:
            continue
        if abs(tx * oy - ty * ox) <= DIRECTION_TOL * t_norm * o_norm:
            continue
        constraints.append((ox, oy, o_norm))
    if not constraints:
        return (tx / t_norm**2, ty / t_norm**2)
    cx, cy, _ = constraints[0]
    det = tx * cy - ty * cx
    direction = (cy / det, -cx / det)
    d_norm = math.hypot(*direction)
    for ox, oy, o_norm in constraints[1:]:
        if abs(ox * direction[0] + oy * direction[1]) > (
            DIRECTION_TOL * o_norm * d_norm
        ):
            return None
    return direction


def stepper_move_direction(kin, stepper, point):
    """Toolhead XY move per 1 mm of travel of stepper alone at point.

    Raises ValueError with a user facing reason when there is none.
    """
    target = motor_jacobian(stepper, point)
    others = [motor_jacobian(s, point) for s in kin.get_steppers()]
    if target is None or None in others:
        raise ValueError(
            "the motor position must be linear in X and Y for these kinematics"
        )
    others = [j for s, j in zip(kin.get_steppers(), others) if s is not stepper]
    direction = single_motor_direction(target, others)
    if direction is None:
        raise ValueError(
            "'%s' cannot move without moving another motor"
            % (stepper.get_name(),)
        )
    return direction
