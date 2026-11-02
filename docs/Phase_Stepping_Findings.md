# Phase stepping development findings

Notes from developing [phase stepping](Phase_Stepping.md) and the
[motor harmonics](Motor_Harmonics.md) measurements: what was learned on
hardware, why parts of the design are the way they are, and what is
still open. The hardware referred to is an STM32F446 board with TMC5160
drivers on hardware SPI (64 microsteps), a CoreXY printer and a USB
LIS2DW accelerometer on the toolhead, with travel moves at 600 mm/s.

## Design overview

- The micro-controller updates the coil currents on a fixed grid of
  ticks. It executes queued constant acceleration segments in integer
  phase units, where a 32 bit wrap is one electrical period (4 full
  steps) and the driver's microstep counter is `phase >> 22`. A segment
  with negated velocity and acceleration retraces the integer path
  exactly, so a test move returns to its start phase.
- Wait entries start a burst of segments on the tick at a given clock.
  Running out of segments while moving, reaching a wait while moving, or
  reaching a wait after its clock shuts down instead of drifting from the
  planned timeline.
- The host converter (`phasegen.c`) samples the stepper position,
  including kinematics and input shaping, at the time each tick's
  currents take effect, and fits integer segments to it while simulating
  the micro-controller's integer state, so rounding never accumulates. In
  tests with CoreXY moves up to 600 mm/s and MZV/EI shaping every update
  stays within 0.4 um of the planned position at 20-160 segments per
  second.
- Step pulses keep running alongside. A TMC5160 in direct mode ignores
  them for its coils but its microstep counter (MSCNT) keeps counting
  them, so homing and position tracking need no change. This was
  confirmed with `PHASE_STEPPING_STEP_TEST`.
- Phase stepping registers as a step generator on the toolhead (Kalico
  bleeding-edge-v2). Klipper master and Kalico main moved step
  generation to `motion_queuing` (`register_flush_callback`); a port
  there would need that hook. Kalico is not expected to bring
  `motion_queuing` to bleeding-edge-v2, so this stays a bleeding-edge
  feature.

## Micro-controller

- `enable_pclock()` derives the clock enable bit from the peripheral
  address, which is wrong for the stm32f4 DMA controllers: the stream
  ignored every write and no currents reached the driver. The DMA clock
  is enabled explicitly, and the firmware shuts down if the stream does
  not take its configuration or a transfer does not start.
- STM32H7 keeps Klipper's ram in DTCM, which the DMA controllers cannot
  reach. The H7 SPI FIFOs hold at least 8 bytes, so a whole 5 byte coil
  write goes into the FIFO from the timer interrupt and needs no DMA.
  This backend is built for H723 and H743 but not yet tested on
  hardware.
- Other micro-controllers fall back to 10 kHz updates from the scheduler
  with blocking transfers, which is too coarse for fast travel.
- At 40 kHz an update takes about 2 us on an STM32F446 with DMA.
- Trinamic drivers answer each message with the register that the
  message before it addressed, so the host reads a register with two
  messages. A coil write between the two made the second answer return
  the coil register, which decoded as a driver error (a corrupted GSTAT
  read).
- Locking the bus from the first to the second message did not hold up.
  Waiting 2 ms for the second message stopped coil updates long enough
  for the motor to slip at 200 mm/s under heavy SPI traffic, so the lock
  timed out after 250 us. The host sends the two messages one at a time,
  however, and they can arrive further apart than that: a print was shut
  down by "GSTAT: 010a01df reset=1 drv_err=1 uv_cp=1", which is a coil
  value (two 9 bit currents of length 248), not a driver error.
- The MCU now keeps the host's last message to each phase stepped driver
  and, when a coil write went to the driver after it, sends it again
  before a host transfer that reads an answer. The answer is then right
  however far apart the host's messages are, and the bus is no longer
  locked between them. Such resends are counted as `replays`.
- Verified on hardware: test moves from 5 to 850 mm/s and back with no
  jump or slip, also under heavy SPI load; `swap_coils: True` is right
  for the TMC5160.

## Driver registers

- IHOLD_IRUN is write only, so reading it from the driver returns zero;
  an early version ran (and restored) zero current. Values come from the
  driver module's register cache instead.
- `FieldHelper.set_field()` always writes the driver module's register
  cache, so temporary direct mode values leaked into later writes such
  as sensorless homing. Temporary values are computed with `with_field()`
  instead.
- Any `stealthchop_threshold` enables stealthChop, which takes over at
  standstill, and a driver in direct mode sees no step pulses. en_pwm_mode
  is cleared while the coils are driven directly.
- The driver scales direct mode currents by IHOLD, so direct mode sets
  IHOLD to IRUN. The driver module writes IHOLD_IRUN and GCONF on its own
  (SET_TMC_CURRENT, homing currents, sensorless homing, SET_TMC_FIELD);
  those writes are adjusted while in direct mode, and its own values
  return afterwards. Restoring a snapshot instead undid such changes.
- With a hold_current below the run current, the micro-controller lowers
  the current at standstill as the driver would: 2^20 driver clocks plus
  TPOWERDOWN * 2^18, then one step per IHOLDDELAY * 2^18 clocks.
- A driver reset or error is caught by the driver module's periodic
  GSTAT and DRV_STATUS checks, which shut down as without phase stepping.
- Register transfers that bypass the driver module must hold its SPI
  mutex. The write of the starting coil currents did not, and at the end
  of homing it overlapped a transfer from another greenlet on the same
  SPI oid; the two took each other's response and Klipper stopped with
  `KeyError: ('spi_transfer_response', 0)`.

## Host integration

- An idle toolhead has no moves in the trapq and its head sentinel reads
  as position zero, so the first move after enabling made the target jump
  hundreds of millimetres ("could not follow"). The converter holds the
  last position there and early in the stationary filler move ahead of
  the next move, and takes its start position from the toolhead (a fresh
  input shaper wrapper reports positions in its own frame).
- The toolhead flushes nothing while idle, so the first flush after a
  heat-up sampled every tick of the rest. On a Raspberry Pi a few minutes
  of rest took over 100 ms per motor, and the second motor's first motion
  reached the micro-controller after it was due ("Phase stepper move too
  late"). Ticks where no move can drive the stepper (widened by the
  input shaper's activity window) are now skipped.
- After idle, the first motion is generated only about 0.25 s before it
  starts, and the serial queue sends it about 100 ms ahead. The log
  statistics report this lead (`lead_min`) and warn below 50 ms.
- On CoreXY both motors move when homing either axis, so homing Y turned
  stepper_x off, and it only returned after homing a rail it belongs to.
  Phase stepping now returns after any homing that turned it off.
- FORCE_MOVE and STEPPER_BUZZ move by step pulses only, so they run with
  phase stepping off for their stepper.
- `FIRMWARE_RESTART` and `RESTART` restart klippy within the same
  process and do not reload Python modules or `c_helper.so`. After
  updating the host code the service must be restarted.
- SET_VELOCITY_LIMIT reports the limits every time it runs; the speed
  ramps set accelerations with `toolhead.set_accel()` instead.

## Measurements and calibration

- Uncorrected, the test printer showed about 5 um of H2 (1 per 2 full
  steps), up to 21 um where H2 met a 38 Hz resonance, strong H4 on X only
  and strong H1 on Y above 90 mm/s. Calibration therefore covers H1, H2
  and H4 per motor and direction.
- A correction calibrated at the speed where a harmonic is largest made
  it worse at other speeds: the motor's error changes with speed as its
  load angle grows (stepper_x H4 rose from 0.37 to 1.13 um at
  52.5 mm/s). Each harmonic is now calibrated over several speeds with
  one correction for all of them.
- The multi-speed correction weights each speed by its uncorrected level
  (equal relative reduction). Over 10-160 mm/s this let large low speed
  peaks grow back (H2 at 15 mm/s from 0.34 to 2.06 um). Limiting the
  range to the speeds that are printed works well: with SPEED_MIN=40,
  H2 and H4 at 40-100 mm/s fell by 66-97 % from uncorrected (stepper_x
  H2 3.96 to 0.13 um, H4 0.16 to 0.055 um; stepper_y H2 4.65 to 0.16 um,
  H4 0.13 to 0.037 um). A minimax objective on absolute vibration is an
  option if wider ranges are needed. The default range now starts at
  30 mm/s.
- Calibration used to start from the stored correction and pick the
  speeds where the remaining vibration was above noise. Speeds the
  earlier correction already cancelled were then left out, and the new
  correction made them worse (stepper_x H2 at 40-50 mm/s rose 2-4 times
  while 55-100 mm/s fell). The result also depended on which
  calibration came before. Calibration now clears the harmonic's
  correction first.
- From no correction, stepper_y H2 fell from 4.52 to 0.17 um on average
  over 40-100 mm/s, close to the earlier calibrations, but 0.26-0.34 um
  was left at 50-60 mm/s where the motor's H2 peaks at about 10 um (a
  resonance near 145 Hz). Weighting each speed by its uncorrected level
  asks for the same relative reduction everywhere, so the largest peak
  keeps the largest absolute vibration, and the evenly spread
  calibration speeds missed the peak. Calibration speeds are now picked
  around the largest peaks, and the correction minimises the worst
  vibration left (Lawson's algorithm on the measured linear response).
- With that and SPEED_MIN=50, the worst H4 over 50-100 mm/s fell to
  0.058 um on stepper_x and 0.032 um on stepper_y, about half of the best
  earlier calibration, and stepper_x H2 peaked at 0.18 um. Stepper_y H2
  stayed at 0.17 um on average with a 0.32 um worst case: its
  uncorrected H2 peaks near 9.5 um at both 57.5 and 70 mm/s, and the
  residue is spread evenly over 50-95 mm/s, as minimax does. All of
  these are ramp sweeps; repeated uncorrected sweeps differ by about
  10-15 %.
- With phase stepping and corrections the motors are much quieter. Input
  shaper measurements before and after showed unchanged resonance
  frequencies; differences in peak height and damping were within run to
  run variation.
- Speed ramps: like a resonance sweep, the motor ramps up and down
  through the speed range and each harmonic is demodulated against the
  motor's electrical angle from the planned motion. Ramps through low
  speeds are slower so that a window of 8 electrical periods spans at
  most a quarter of the speed, and up and down ramps are averaged to
  cancel their lag behind the sweep. Measured back to back on hardware,
  ramps and constant speeds agree for H2 and H4 (median ratio 1.05 and
  1.10); the narrowest peaks read 20-30 % lower on ramps.
- H1 reads about 3 times too high on ramps. The likely cause is the belt:
  with a 20 tooth pulley the tooth meshing is at order 0.4 of the
  electrical frequency, its overtones at 0.8 and 1.2 sit next to H1, and
  an 8 period window cannot separate them (H2 and H4 coincide with belt
  overtones, so both methods agree there). Windows of about 20 periods
  would fix it at the cost of slower ramps.
- SAVE_CONFIG keeps a backup of printer.cfg as it was before saving, so
  restoring corrections from the backup of a calibration's save brings
  back the previous calibration's values.
- The calibrated stepper_x sweep showed two limits of calibrating at a
  handful of constant speeds. H2 was cancelled to about 0.1 um at
  100 mm/s but 1.5 um was left at 70 mm/s, where the uncorrected peak
  sat: the best correction moves with speed and a few speeds cannot pin
  the compromise down. H4 rose from about 0.07 to 0.3 um at 60-75 mm/s,
  between the calibration speeds, and nothing reported it, since the
  sweep after calibration ran only with GRAPH=1 and was not compared.
- Calibration therefore now works from whole-range sweeps. Each sweep
  measures every speed in the range every 2.5 mm/s, coherently: the
  phasors refer to the motor's electrical angle at the test point, so
  ramps up and down, passes and sweeps are comparable and average
  coherently. Every harmonic's response to its correction is identified
  from one baseline sweep and four probe sweeps in which all harmonics
  are perturbed at once (both signs of a real and of an imaginary step,
  so that the differences cancel the even orders of the response), since
  the demodulation separates the harmonics anyway. All sweeps so far,
  the refinements included, enter one weighted least squares fit of
  offset and gain per speed, and the correction minimises the worst
  predicted vibration over the range. Refinement stops when a step
  should gain less than 5 % or less than half the noise floor, the last
  sweep is the verification, a harmonic whose correction does not beat
  none is left uncorrected, and speeds that got worse are reported.
  About 8 sweeps replace the 15 batches of constant speed passes.
- Each window is now demodulated by a joint least squares fit of all
  harmonics, the belt tooth orders and two noise orders, instead of a
  DFT bin per order. In an 8 period window a Hann bin at H1 takes a
  large part of a belt order 0.2 orders away, which is the likely reason
  H1 read 3 times too high on ramps. The belt order is the electrical
  period over the belt pitch (0.4 with a 0.8 mm period and GT2), and
  multiples that coincide with a harmonic are dropped, as are noise
  orders within a cycle of another order.
- Calibration demodulates up to 0.47 of the sample rate instead of 0.4.
  The accelerometer's filter attenuates and delays the band below
  Nyquist, but a calibration only compares measurements made the same
  way, so the filter cancels; only aliasing would matter, and a
  demodulation at one order does not pick up the band above Nyquist.
  This takes H4 from 128 to about 150 mm/s with a LIS2DW at 1600 Hz.
- The probe size follows the harmonic: about the correction its largest
  uncorrected displacement implies (2 pi times the displacement over the
  period), at least 0.02 rad so that a small harmonic answers clearly
  above the noise, and at most 0.06 rad to stay linear. A fixed 0.02 rad
  was 20 times the implied correction of H4 and a quarter of that of
  stepper_y's H2.
- The gain of a harmonic's response at a speed is the printer's
  transfer function at that frequency and so smooth over speed, which
  would allow smoothing it across neighbouring speeds. This was left out:
  by the end of a calibration the offset at each speed is averaged over
  all sweeps, and a gain error only slows convergence, which the refits
  correct.
- The sweep-based calibration has not been run on hardware yet. The
  per-speed optimum of each harmonic can be read from its sweeps (minus
  offset over gain); if it drifts with speed as the stepper_x H2 residue
  suggests, a correction with a term linear in speed would be the next
  step, and the fit stays linear in its unknowns.

## Status and open items

- Tested on hardware: STM32F446 with TMC5160. Not yet tested: STM32H7
  backend, TMC2130, TMC2160 and TMC2240.
- Next: a hardware run of the sweep-based calibration against the
  earlier results, a DMA backend for STM32G0B1 (its SPI FIFO holds
  4 bytes), and a print comparison of the wall surface with and without
  phase stepping.
- Moving phase segment generation into C threads, as Klipper's
  `motion_queuing` does for steps, would remove the time one motor waits
  for the other at a flush.
