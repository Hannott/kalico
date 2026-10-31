# Phase stepping

**This feature is experimental and incomplete.** It can currently only
run a test move; printing with it is not supported yet.

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
- spreadCycle: set `stealthchop_threshold: 0` in the driver section.
- Micro-controller firmware built with phase stepping support. In
  `make menuconfig`, enable "Enable extra low-level configuration
  options" and then "Phase stepping support (experimental)".

The micro-controller updates the coil currents 40000 times a second on
STM32F405, F407, F427, F429 and F446, using a dedicated timer and DMA.
Other micro-controllers with hardware SPI use 10000 updates a second.

## Configuration

Add a section for each stepper to drive this way:

```
[phase_stepping stepper_x]
```

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
- Whether the motor returned to its start phase.

Use `BUS_LOAD=1` to read driver registers during the move. This checks
that other SPI traffic on the same bus still works while the coils are
being updated.

If the test is interrupted, the motors are turned off, because the motor
may have stopped anywhere along the move. Home again before continuing.
