# Phase stepping

**This feature is experimental.** It follows normal motion but does not
yet correct the motor's own position errors.

Phase stepping drives a stepper motor by setting its two coil currents
directly, many thousand times a second, instead of sending step pulses
to the driver. This is the basis for correcting the motor's own
position errors, which cause vertical fine artifacts (VFA). See
[motor harmonics](Motor_Harmonics.md) for how to measure them.

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

The micro-controller updates the coil currents 40000 times a second on
STM32F405, F407, F427, F429 and F446, using a dedicated timer and DMA.
Other micro-controllers with hardware SPI use 10000 updates a second.

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
homing. `FORCE_MOVE` and `STEPPER_BUZZ` do not move a phase stepped motor:
disable phase stepping first with `ENABLE=0`.

When it is disabled, the command warns if the driver's step position and
the phase ended apart by more than half a microstep, which would point to
a wrong direction or lost steps.

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
