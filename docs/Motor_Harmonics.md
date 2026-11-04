# Motor harmonics

A stepper motor's rotor does not follow the commanded microstep angle
exactly. Detent torque and small differences between its coils and
magnets make it run slightly ahead and behind within every electrical
period, which is 4 full steps. At a constant speed this becomes
vibration at whole multiples (harmonics) of the electrical frequency,
and on printed walls it shows up as vertical fine artifacts (VFA).

The `MEASURE_MOTOR_HARMONICS` command measures that vibration with an
accelerometer, so you can see how large it is on your printer and at
which speeds it is worst.

## Setup

Enable the module with the accelerometer you use for input shaper
calibration:

```
[phase_calibrate]
accel_chip: lis2dw
```

numpy must be installed in the Kalico environment, as for
[resonance measurements](Measuring_Resonances.md).

## Running the measurement

Home the printer, then measure each X/Y motor:

```
G28
MEASURE_MOTOR_HARMONICS STEPPER=stepper_x
MEASURE_MOTOR_HARMONICS STEPPER=stepper_y
```

The test moves only the requested motor. On a CoreXY printer that is a
diagonal move. The motor ramps smoothly up and down through the speeds
between `speed_min` and `speed_max` in both directions, like the sweep of
a resonance test, which takes about a minute per motor. Ramping up and
down both counts, so that the slight lag of the vibration behind the
changing speed cancels out. `RAMP=0` measures at constant speeds in
steps of `speed_step` instead, which takes longer but holds each speed
until the vibration is steady.
Turn off fans that are not needed, as their vibration raises the noise
level.

The electrical frequency is `motor speed / electrical period`, where the
electrical period is `4 * rotation_distance / full_steps_per_rotation`.
With a 40 mm rotation distance and 200 full steps, it is 0.8 mm, so a
motor speed of 100 mm/s gives 125 Hz and harmonic 4 is at 500 Hz.

## Reading the results

The command reports, per harmonic and direction, the largest toolhead
displacement found and the speed it occurred at, for example:

```
H2 (1 per 2 full steps): fwd 1.2 um at 85 mm/s (212 Hz, 14.0x noise), ...
H4 (1 per full step): fwd 0.45 um at 45 mm/s (225 Hz, 9.3x noise), ...
```

Harmonics 2 and 4 usually dominate. A peak is often largest where its
frequency matches a mechanical resonance of the printer. Measurements
less than 3 times above the noise level are not reported as peaks.

Harmonics are only measured up to 40% of the accelerometer's sample
rate (640 Hz for a LIS2DW at 1600 Hz), because accelerometers filter
out the band close to half their sample rate.

All measurements are written to a CSV file. Graph it with:

```
~/klippy-env/bin/python ~/klipper/scripts/graph_motor_harmonics.py /tmp/motor_harmonics_stepper_x_<name>.csv -o /tmp/motor_harmonics.png
```

To have the graph made automatically next to the CSV, add `GRAPH=1` to
the command or set `graph: True` in
[phase_calibrate](Config_Reference.md#phase_calibrate). Matplotlib must
be installed in the klippy environment. Setting `output_dir` to a
directory the web interface shows, such as `~/printer_data/config`,
makes the files easy to open.
