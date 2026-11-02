# Phase stepping

**This feature is experimental.**

Phase stepping drives a stepper motor by setting its two coil currents
directly, many thousand times a second, instead of sending step pulses
to the driver. This is the basis for correcting the motor's own
position errors, which cause vertical fine artifacts (VFA). See
[motor harmonics](Motor_Harmonics.md) for how to measure them.

The technique comes from Prusa Research, whose
[Prusa-Firmware-Buddy](https://github.com/prusa3d/Prusa-Firmware-Buddy)
introduced phase stepping and its accelerometer based correction for
their printers. This implementation is written for Kalico's split
between the host, which plans the motion, and the micro-controller,
which drives the coils.

## Requirements

- A Trinamic driver on SPI that supports direct mode: TMC2130, TMC2160,
  TMC5160 or TMC2240. UART drivers such as the TMC2209 are not
  supported. Only the TMC5160 has been tested so far.
- The driver must use a hardware SPI bus (`spi_bus`, not
  `spi_software_*_pin`) and must not be in a daisy chain.
- Micro-controller firmware built with phase stepping support. In
  `make menuconfig`, enable "Enable extra low-level configuration
  options" and then "Phase stepping support (experimental)".

While the coils are driven directly, the driver runs in spreadCycle even
if `stealthchop_threshold` is set; its own setting returns afterwards.
The driver's `run_current` applies while the motor moves, and changes
made with `SET_TMC_CURRENT` or by homing currents take effect as usual.
If a `hold_current` lower than the run current is configured, the
micro-controller lowers the current at standstill the way the driver
would, after the delay and in the steps set by its `driver_TPOWERDOWN`
and `driver_IHOLDDELAY`, and returns to the run current when the motor
moves again.

The micro-controller updates the coil currents 40000 times a second on
STM32F405, F407, F427, F429 and F446 (using a dedicated timer and DMA)
and on STM32H723, H743 and H750 (using a dedicated timer and the SPI
FIFO; not yet tested on hardware). Other micro-controllers with hardware
SPI use 10000 updates a second, which is too coarse for fast travel
moves.

## Configuration

Add a section for each stepper to drive this way:

```
[phase_stepping stepper_x]
```

## Enabling it

```
PHASE_STEPPING_ENABLE STEPPER=stepper_x
```

The step pulses keep running as before, so homing and position tracking
work unchanged; the driver ignores them for its coils while phase stepping
is enabled but keeps counting them. Phase stepping turns off by itself for
homing moves of the stepper and returns afterwards, and turns off when the
motors are disabled. Set `auto_enable: True` to turn it on after each
homing. `FORCE_MOVE` and `STEPPER_BUZZ` turn it off for their stepper
while they run and back on afterwards if the motor is still enabled.

When it is disabled, the command warns if the driver's step position and
the phase ended apart by more than half a microstep, which would point to
a wrong direction or lost steps.

## Calibration

Phase stepping pays off with the corrections that cancel the motor's
own position errors. Finding them needs the accelerometer used for input
shaper calibration on the toolhead, and numpy in the Kalico environment
(as [resonance measurements](Measuring_Resonances.md) do).

### Setting up

Name the accelerometer, and let phase stepping start by itself after
homing:

```
[phase_calibrate]
accel_chip: lis2dw

[phase_stepping stepper_x]
auto_enable: True

[phase_stepping stepper_y]
auto_enable: True
```

The defaults suit a belt driven printer: the test moves run around the
middle of the bed, the belt is GT2, and the corrections are calibrated
for motor speeds from 30 to 160 mm/s. On a CoreXY printer a move along X
or Y at a toolhead speed of v runs each motor at 1.4 v, so that range
covers toolhead speeds of about 20 to 115 mm/s; on a Cartesian printer
motor and toolhead speed are the same. If you print walls at other
speeds, set `speed_min` and `speed_max` in `[phase_calibrate]` to the
motor speeds you use: the corrections are best inside the range they
were calibrated for, and the calibration spends its effort there. Set
`belt_pitch: 0` for a motor without a belt, and `graph: True` for a PNG
graph of each sweep (needs matplotlib). The other options are described
in the [config reference](Config_Reference.md#phase_calibrate).

### Calibrating

Have the printer as it is when it prints: motors warm from a few
minutes of moves or a short print, fans that are not needed off (their
vibration raises the noise level). Then run, for each motor:

```
G28
PHASE_STEPPING_CALIBRATE STEPPER=stepper_x
PHASE_STEPPING_CALIBRATE STEPPER=stepper_y
SAVE_CONFIG
```

A calibration sweeps the motor up and down through the speed range
about eight times, a minute each: once without correction, four times
with small probe corrections that show how each harmonic answers, then
with the correction those sweeps predict to be best, refined until it
stops improving. The last sweep verifies the result. On a CoreXY printer
the moves are diagonal, since only one motor runs.

### Reading the result

For each harmonic and direction the command reports the worst and the
average vibration over the speed range before and after, the noise
level, and the correction found, for example:

```
H2 forward over 30-160 mm/s: worst 4.520 -> 0.170 um, average 1.812 -> 0.094 um,
noise 0.052 um (correction 0.0312 rad at 2.21 rad at 30 mm/s, 0.0251 rad at 2.41 rad at 160 mm/s)
H4 forward over 30-160 mm/s: worst 0.160 -> 0.041 um, ... (correction 0.0011 rad at -0.40 rad)
H3 forward: below noise, left uncorrected
```

- A correction given at two speeds changes with the speed between them:
  a loaded motor's error moves as its speed grows, and the
  micro-controller follows the speed linearly between the two ends and
  holds the nearer end outside them. A correction given once is the same
  at every speed; the calibration keeps it that way unless following the
  speed clearly helps.
- A result close to the noise level is as good as the accelerometer can
  tell. Repeated calibrations differ by about the noise.
- "left uncorrected" means the harmonic was not measurable above the
  noise, or that no correction beat having none. Both are normal for
  harmonics 1 and 3; harmonics 2 and 4 are the ones that matter on most
  motors.
- "worse at ..." lists speeds where the harmonic rose above its
  uncorrected level by more than the noise. A few such speeds at a low
  level are the price of one correction for a whole range; many, or
  large ones, mean the range should be narrowed to the speeds you print.

The sweeps before and after are written to `output_dir` as CSV files
(see [motor harmonics](Motor_Harmonics.md)), with graphs when `graph` is
set.

### Checking

`MEASURE_MOTOR_HARMONICS STEPPER=stepper_x` measures a motor without
changing anything, so running it before and after shows the difference
across all speeds; `PHASE_STEPPING_ENABLE STEPPER=stepper_x ENABLE=0`
switches back to step pulses for a comparison. A tall, thin print with
walls at your usual speed shows the vertical fine artifacts directly.

### Calibrating again

The corrections belong to the motor, its driver settings and its load.
Calibrate again after changing `run_current`, `microsteps`, the motor or
the driver, the belt tension or the toolhead mass, and after a change of
`speed_min` or `speed_max`. A calibration starts from no correction, so
it never depends on the one before it.

## Test move

Home the printer, move the toolhead away from the edges, then run:

```
PHASE_STEPPING_TEST STEPPER=stepper_x
```

The motor moves 10 mm forward and back using direct coil currents, and
the driver then returns to normal stepping at exactly the position it
started from. The command reports:

- The average and longest time the micro-controller spends per update,
  and how much of each update period that is.
- Lost updates (the timer was delayed by more than one period) and
  updates skipped while other devices used the SPI bus.
- How many coil current writes the micro-controller made, and how many
  different values the driver's coil register held when read back during
  the move. If the read back value does not change, the writes are not
  reaching the driver and the motor does not move.
- Whether the motor returned to its start phase.

Use `BUS_LOAD=1` to read driver registers during the move. This checks
that other SPI traffic on the same bus still works while the coils are
being updated.

If the test is interrupted, the motors are turned off, because the motor
may have stopped anywhere along the move. Home again before continuing.

## Diagnostics

While phase stepping is on, the statistics lines in the log include, for
each stepper, `lead_min` (the shortest time between queueing the start of
a motion and that start; the micro-controller shuts down with "Phase
stepper move too late" if it reaches zero) and `generate_max` (the longest
time spent computing coil updates in one batch). One stepper per
micro-controller also reports its lost and bus locked updates, driver
messages sent again after a coil write (`replays`) and update times. A
warning is logged when a motion is queued less than 50 ms before it
starts.

The driver module's periodic checks keep running in direct mode: a driver
error or reset shuts the printer down as it would without phase stepping.

See the [development findings](Phase_Stepping_Findings.md) for what was
learned while developing phase stepping and what is still open.
