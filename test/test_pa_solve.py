import math

import pytest

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
