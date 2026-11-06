import math
from unittest import mock

import pytest

from klippy import gcode as gcode_module
from klippy.extras import tuning_tower
from klippy.kinematics import extruder

MODELS = [extruder.PATanhModel, extruder.PAReciprModel]


def _model(model_class, linear_advance, nonlinear_offset, velocity):
    model = model_class()
    model.linear_advance = linear_advance
    model.nonlinear_offset = nonlinear_offset
    model.linearization_velocity = velocity
    return model


@pytest.mark.parametrize("model_class", MODELS)
def test_solve_recovers_the_parameters_the_measurements_came_from(
    model_class,
):
    expected = _model(model_class, 0.017, 0.12, 1.0)
    slow_velocity, fast_velocity = 20.0, 150.0
    linear_advance, nonlinear_offset = extruder.solve_nonlinear_pa(
        model_class,
        slow_velocity,
        expected.advance(slow_velocity) / slow_velocity,
        fast_velocity,
        expected.advance(fast_velocity) / fast_velocity,
        1.0,
    )
    assert linear_advance == pytest.approx(0.017)
    assert nonlinear_offset == pytest.approx(0.12)


@pytest.mark.parametrize("model_class", MODELS)
def test_solution_matches_both_linear_measurements(model_class):
    linear_advance, nonlinear_offset = extruder.solve_nonlinear_pa(
        model_class, 20.0, 0.06, 150.0, 0.02, 2.0
    )
    assert linear_advance >= 0.0
    assert nonlinear_offset >= 0.0
    model = _model(model_class, linear_advance, nonlinear_offset, 2.0)
    assert model.advance(20.0) == pytest.approx(0.06 * 20.0)
    assert model.advance(150.0) == pytest.approx(0.02 * 150.0)


@pytest.mark.parametrize("model_class", MODELS)
def test_equal_measurements_need_no_offset(model_class):
    linear_advance, nonlinear_offset = extruder.solve_nonlinear_pa(
        model_class, 20.0, 0.04, 150.0, 0.04, 1.0
    )
    assert linear_advance == pytest.approx(0.04)
    assert nonlinear_offset == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("model_class", MODELS)
def test_advance_grows_with_velocity(model_class):
    model = _model(model_class, 0.017, 0.12, 1.0)
    velocities = [0.0, 1.0, 5.0, 20.0, 50.0, 150.0, 300.0]
    advances = [model.advance(v) for v in velocities]
    assert advances[0] == 0.0
    assert advances == sorted(advances)


def test_linear_model_advance():
    model = extruder.PALinearModel()
    model.pressure_advance = 0.05
    assert model.advance(100.0) == pytest.approx(5.0)


def test_rejects_fast_face_wanting_more_than_slow_face():
    with pytest.raises(ValueError, match="linear model"):
        extruder.solve_nonlinear_pa(
            extruder.PAReciprModel, 20.0, 0.02, 150.0, 0.03, 1.0
        )


def test_rejects_more_total_advance_at_the_slow_point():
    with pytest.raises(ValueError, match="total advance"):
        extruder.solve_nonlinear_pa(
            extruder.PAReciprModel, 20.0, 0.2, 150.0, 0.02, 1.0
        )


@pytest.mark.parametrize("model_class", MODELS)
def test_rejects_linearization_velocity_far_above_test_range(model_class):
    with pytest.raises(ValueError, match="VELOCITY"):
        extruder.solve_nonlinear_pa(model_class, 20.0, 0.06, 150.0, 0.02, 1.0e6)


def test_rejects_fast_velocity_not_above_slow_velocity():
    with pytest.raises(ValueError, match="FAST_VELOCITY"):
        extruder.solve_nonlinear_pa(
            extruder.PAReciprModel, 50.0, 0.06, 50.0, 0.02, 1.0
        )


def test_filament_velocity_scales_with_line_cross_section():
    filament_area = math.pi * (1.75 * 0.5) ** 2
    # A 0.4 x 0.2 line at 150 mm/s pushes about 5 mm/s of 1.75 mm filament
    assert extruder.filament_velocity(
        150.0, 0.4, 0.2, filament_area
    ) == pytest.approx(4.989, abs=1e-3)
    assert extruder.filament_velocity(
        150.0, 0.8, 0.2, filament_area
    ) == pytest.approx(2 * 4.989, abs=2e-3)


def _tower(step_delta=0.005, step_height=5.0, start=0.0, skip=0.0):
    tower = tuning_tower.TuningTower.__new__(tuning_tower.TuningTower)
    tower.command, tower.parameter = "SET_PRESSURE_ADVANCE", "ADVANCE"
    tower.start, tower.factor, tower.band = start, 0.0, 0.0
    tower.step_delta, tower.step_height, tower.skip = (
        step_delta,
        step_height,
        skip,
    )
    return tower


class _Printer:
    def __init__(self, **objects):
        self.objects = objects

    def lookup_object(self, name, default=None):
        return self.objects.get(name, default)


def _stepper(**objects):
    stepper = extruder.ExtruderStepper.__new__(extruder.ExtruderStepper)
    stepper.printer = _Printer(**objects)
    return stepper


def _gcmd(**params):
    gcode = mock.Mock()
    return gcode_module.GCodeCommand(
        gcode, "PA_SOLVE", "", {k: str(v) for k, v in params.items()}, False
    )


def test_band_value_in_step_mode():
    tower = _tower()
    assert tower.get_band_value(
        "SET_PRESSURE_ADVANCE", "ADVANCE", 6
    ) == pytest.approx(0.030)
    assert tower.get_band_value(
        "SET_PRESSURE_ADVANCE", "ADVANCE", 0
    ) == pytest.approx(0.0)


def test_band_value_counts_bands_after_skip():
    tower = _tower(start=0.02, skip=3.0)
    assert tower.get_band_value(
        "SET_PRESSURE_ADVANCE", "ADVANCE", 2
    ) == pytest.approx(0.03)


def test_band_value_in_factor_mode_uses_the_band_center():
    tower = _tower(step_delta=0.0, step_height=0.0)
    tower.factor, tower.band = 0.001, 5.0
    assert tower.get_band_value(
        "SET_PRESSURE_ADVANCE", "ADVANCE", 1
    ) == pytest.approx(0.0075)


def test_band_value_needs_the_same_command_and_bands():
    tower = _tower()
    assert tower.get_band_value("SET_PRESSURE_ADVANCE", "OFFSET", 1) is None
    tower.step_delta = tower.step_height = 0.0
    tower.factor = 0.001
    assert tower.get_band_value("SET_PRESSURE_ADVANCE", "ADVANCE", 1) is None
    tower.command = None
    assert tower.get_band_value("SET_PRESSURE_ADVANCE", "ADVANCE", 1) is None


def test_band_reads_the_value_from_the_last_tower():
    stepper = _stepper(tuning_tower=_tower(), pa_test=mock.Mock(height=50.0))
    assert stepper._get_tower_advance(
        _gcmd(SLOW_BAND=6), "SLOW"
    ) == pytest.approx((0.030, 6))


def test_advance_value_is_used_as_given():
    stepper = _stepper()
    assert stepper._get_tower_advance(_gcmd(FAST_ADVANCE=0.025), "FAST") == (
        0.025,
        None,
    )


@pytest.mark.parametrize("params", [{}, {"SLOW_BAND": 6, "SLOW_ADVANCE": 0.03}])
def test_needs_exactly_one_of_band_and_advance(params):
    stepper = _stepper(tuning_tower=_tower())
    with pytest.raises(gcode_module.CommandError, match="either"):
        stepper._get_tower_advance(_gcmd(**params), "SLOW")


def test_band_without_a_tower_since_restart_is_rejected():
    tower = _tower()
    tower.command = None
    stepper = _stepper(tuning_tower=tower)
    with pytest.raises(gcode_module.CommandError, match="SLOW_ADVANCE"):
        stepper._get_tower_advance(_gcmd(SLOW_BAND=6), "SLOW")


def test_band_above_the_tower_is_rejected():
    stepper = _stepper(tuning_tower=_tower(), pa_test=mock.Mock(height=50.0))
    with pytest.raises(gcode_module.CommandError, match="above the top"):
        stepper._get_tower_advance(_gcmd(SLOW_BAND=10), "SLOW")
