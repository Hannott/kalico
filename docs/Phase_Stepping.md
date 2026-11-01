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

With phase stepping enabled and the accelerometer from
[phase_calibrate](Config_Reference.md#phase_calibrate) on the toolhead,
run for each motor:

```
PHASE_STEPPING_CALIBRATE STEPPER=stepper_x
PHASE_STEPPING_CALIBRATE STEPPER=stepper_y
SAVE_CONFIG
```

The correction shifts the coil current angle so that the rotor lands
where it should; it is measured and stored separately for each motor and
direction. A motor's error changes somewhat with speed, so each harmonic
is calibrated over several speeds around its largest peaks; use
`SPEED_MIN` and `SPEED_MAX` to limit it to the speeds you print walls at.
Compare with `MEASURE_MOTOR_HARMONICS` before and after.

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
micro-controller also reports its lost and bus locked updates, unfinished
driver register reads (`pair_timeouts`) and update times. A warning is
logged when a motion is queued less than 50 ms before it starts.

The driver module's periodic checks keep running in direct mode: a driver
error or reset shuts the printer down as it would without phase stepping.

See the [development findings](Phase_Stepping_Findings.md) for what was
learned while developing phase stepping and what is still open.
