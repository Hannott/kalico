# Stepper motor harmonic analysis
#
# A stepper rotor does not follow the commanded microstep angle exactly:
# detent torque and coil and magnet asymmetries make it lead and lag within
# every electrical period (4 full steps). At a constant speed this shows up
# as vibration at integer multiples ("harmonics") of the electrical
# frequency, and on printed walls as vertical fine artifacts (VFA).
#
# Copyright (C) 2026  Åsmund Collin <aakjaergaard@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import collections
import math

import numpy as np

# Only trust harmonics well below Nyquist: accelerometers low-pass filter
# their output, which attenuates the band close to half the sample rate.
MAX_FREQ_RATIO = 0.4
# A calibration compares measurements made with the same accelerometer, so
# its filter's gain and phase cancel and only aliasing matters, which a
# demodulation at one order does not pick up from the band above Nyquist.
CALIBRATION_FREQ_RATIO = 0.47
MIN_WINDOW_CYCLES = 5
MIN_WINDOW_SAMPLES = 64
MIN_PEAK_SNR = 3.0
# Electrical periods in a ramp window: the noise level is measured half an
# order from each harmonic, which a shorter window cannot separate
RAMP_WINDOW_PERIODS = 8
# Largest speed change, relative to the speed, across one ramp window
RAMP_WINDOW_SPAN = 0.25
# Sinusoids fitted together must differ by this many cycles over the window
MIN_ORDER_SEPARATION = 1.0

# disp and noise_disp are displacement amplitudes (mm); phasor is the
# complex displacement in phase with the motor's electrical angle, so that
# measurements of the same motion can be averaged and compared coherently.
HarmonicResult = collections.namedtuple(
    "HarmonicResult",
    ("harmonic", "freq", "accel", "disp", "noise_disp", "phasor"),
    defaults=(None,),
)
# A constant acceleration part of a test move: from start_time (print time)
# the motor moves start_speed * t + accel * t^2 / 2 in motor_dir, starting
# start_pos (mm, positive forward) from the test's reference point.
Ramp = collections.namedtuple(
    "Ramp",
    (
        "motor_dir",
        "start_time",
        "start_speed",
        "accel",
        "duration",
        "start_pos",
    ),
    defaults=(0.0,),
)


def plan_speeds(speed_min, speed_max, speed_step):
    count = int(math.floor((speed_max - speed_min) / speed_step + 1e-9)) + 1
    return [speed_min + i * speed_step for i in range(count)]


def motor_travel(speed, accel, cruise_time):
    """Motor travel of one test move: accelerate, cruise, decelerate."""
    return speed * speed / accel + speed * cruise_time


def belt_order(period, belt_pitch):
    """Belt tooth meshing frequency as an order of the electrical frequency,
    or None without a belt."""
    if not belt_pitch:
        return None
    return period / belt_pitch


def project_samples(samples, dir_x, dir_y):
    """Split (time, x, y, z) accelerometer samples into times and the
    acceleration component along the unit XY vector (dir_x, dir_y)."""
    data = np.asarray(samples, dtype=float)
    return data[:, 0], data[:, 1] * dir_x + data[:, 2] * dir_y


def window_slice(times, start_time, end_time):
    lo = np.searchsorted(times, start_time, side="left")
    hi = np.searchsorted(times, end_time, side="right")
    return slice(lo, hi)


def fit_orders(times, values, angle, orders):
    """Complex amplitudes of values at several orders of a known angle
    (radians per sample), fitted together by weighted least squares.

    Fitting the orders together keeps a strong neighbour from leaking into
    a weak one when a window is too short to separate them in a plain
    DFT. Uses a Hann window over the sample times, so windows need not be
    a whole number of periods. Each sample is weighted by the time it
    covers, capped at the typical sample interval so that the samples next
    to a dropped accelerometer batch do not stand in for the whole gap.
    For values = A*cos(o*angle + phi) it returns about A*exp(1j*phi) for
    order o, also when the angle does not advance at a constant rate.
    """
    t = times - times[0]
    duration = t[-1]
    intervals = np.diff(t)
    nominal = np.median(intervals)
    covered = np.minimum(intervals, nominal)
    spacing = 0.5 * (np.append(covered, 0.0) + np.insert(covered, 0, 0.0))
    weight = (0.5 - 0.5 * np.cos(2.0 * math.pi * t / duration)) * spacing
    root = np.sqrt(weight)
    columns = [np.ones_like(t)]
    for order in orders:
        columns.append(np.cos(order * angle))
        columns.append(np.sin(order * angle))
    design = np.column_stack(columns) * root[:, None]
    coef = np.linalg.lstsq(design, values * root, rcond=None)[0]
    return {
        order: complex(coef[1 + 2 * i], -coef[2 + 2 * i])
        for i, order in enumerate(orders)
    }


def phase_amplitude(times, values, phase):
    """Complex amplitude of values along a known phase (radians per
    sample); see fit_orders()."""
    return fit_orders(times, values, phase, [1.0])[1.0]


def harmonic_amplitude(times, values, freq, t_ref=None):
    """Complex amplitude of values at freq (single-bin DFT), relative to the
    first sample time, or to t_ref when given."""
    t_phase = times - (times[0] if t_ref is None else t_ref)
    return phase_amplitude(times, values, 2.0 * math.pi * freq * t_phase)


def _accel_to_disp(accel, freq):
    return accel / (2.0 * math.pi * freq) ** 2


def _choose_orders(harmonic, harmonics, belt, periods):
    """Orders to fit together with harmonic in a window of so many
    electrical periods: all harmonics, the belt tooth orders within an
    order of harmonic that the window separates from the harmonics, and
    two noise orders near harmonic that are separated from everything.
    Returns (orders, noise orders)."""
    min_gap = MIN_ORDER_SEPARATION / periods

    def separated(order, others):
        return all(abs(order - o) >= min_gap for o in others)

    orders = sorted(set(harmonics) | {harmonic})
    if belt:
        for k in range(1, int((harmonic + 1.0) / belt) + 1):
            order = k * belt
            if abs(order - harmonic) < 1.0 and separated(order, orders):
                orders.append(order)
    noise = []
    for side in (-1.0, 1.0):
        for shift in (0.5, 0.4, 0.6, 0.3, 0.7):
            order = harmonic + side * shift
            if order > 0.0 and separated(order, orders + noise):
                noise.append(order)
                break
    return orders + noise, noise


def _unresolved(harmonics, speed, period):
    return [
        HarmonicResult(h, h * speed / period, None, None, None)
        for h in harmonics
    ]


def analyze_window(
    times,
    accel,
    speed,
    period,
    harmonics,
    belt=None,
    max_freq_ratio=MAX_FREQ_RATIO,
):
    """Measure each harmonic of one constant-speed window; the phasors are
    relative to the first sample. See analyze_ramp()."""
    if len(times) < 2:
        return _unresolved(harmonics, speed, period)
    ramp = Ramp(1, times[0], speed, 0.0, times[-1] - times[0])
    rows = analyze_ramp(
        times,
        accel,
        ramp,
        period,
        harmonics,
        [speed],
        0.0,
        belt,
        max_freq_ratio,
    )
    return [res for _, _, res in rows]


def plan_ramps(speed_min, speed_max, period, limits, max_travel, settle):
    """Split speed_min..speed_max into ramp segments (v1, v2, ramp_accel).

    limits are (accel, max_ramp_accel). Each test move reaches v1 at accel,
    ramps to v2 at ramp_accel and stops at accel within max_travel of
    motor travel. A window of RAMP_WINDOW_PERIODS electrical periods
    should span at most RAMP_WINDOW_SPAN of the speed, so slow segments
    ramp gently, like a logarithmic sweep. Segments overlap by the speed
    gained while the vibration from reaching v1 settles plus half a window,
    and extend half a window past both ends of the range.
    """
    accel, max_ramp_accel = limits

    def ramp_accel_at(speed):
        rate = RAMP_WINDOW_SPAN * speed * speed / (RAMP_WINDOW_PERIODS * period)
        return min(max_ramp_accel, rate)

    def lead(speed, ramp_accel):
        return ramp_accel * settle + 0.5 * RAMP_WINDOW_SPAN * speed

    top = speed_max * (1.0 + 0.5 * RAMP_WINDOW_SPAN)
    ramp_accel = ramp_accel_at(speed_min * (1.0 - RAMP_WINDOW_SPAN))
    v1 = max(speed_min - lead(speed_min, ramp_accel), 0.0)
    segments = []
    while True:
        room = (
            max_travel - v1 * v1 / (2.0 * accel) + v1 * v1 / (2.0 * ramp_accel)
        )
        v2 = math.sqrt(max(room, 0.0) / (0.5 / ramp_accel + 0.5 / accel))
        if v2 >= top:
            segments.append((v1, top, ramp_accel))
            return segments
        if v2 - v1 <= lead(v2, ramp_accel):
            raise ValueError(
                "a speed ramp needs more travel; raise max_distance or"
                " lower SPEED_MAX"
            )
        segments.append((v1, v2, ramp_accel))
        v1 = v2 - lead(v2, ramp_accel_at(v2))
        ramp_accel = ramp_accel_at(v1)


def ramp_travel(v1, v2, ramp_accel, accel):
    """Motor travel of a test move ramping between v1 and v2."""
    return (v1 * v1 + v2 * v2) / (2.0 * accel) + abs(v2 * v2 - v1 * v1) / (
        2.0 * ramp_accel
    )


def analyze_ramp(
    times,
    accel,
    ramp,
    period,
    harmonics,
    speeds,
    speed_step,
    belt=None,
    max_freq_ratio=MAX_FREQ_RATIO,
):
    """Measure each harmonic around each of speeds that the ramp passes.

    Demodulates against the motor's electrical angle, which the planned
    motion gives at every sample, so the frequency may change during the
    window, and the phasors of all windows of a test refer to the same
    angle. A window spans speed_step of the ramp where the ramp has room
    for it, and at least MIN_WINDOW_CYCLES cycles of the harmonic and
    RAMP_WINDOW_PERIODS electrical periods; a ramp without acceleration
    (a constant speed pass) is one window. The other harmonics, the belt
    tooth orders (belt, as an order of the electrical frequency) and the
    noise orders are fitted together with the harmonic. Harmonics whose
    window does not fit in the ramp, or whose frequency is above
    max_freq_ratio of the sample rate, get None. Returns (speed,
    motor_dir, result) rows.
    """
    a = ramp.accel
    rows = []
    for speed in speeds:
        if a:
            t_center = (speed - ramp.start_speed) / a
        elif abs(speed - ramp.start_speed) > 1e-6:
            continue
        else:
            t_center = 0.5 * ramp.duration
        if not 0.0 <= t_center <= ramp.duration:
            continue
        base_freq = speed / period
        results = []
        for harmonic in harmonics:
            freq = harmonic * base_freq
            needed = 0.5 * MIN_WINDOW_CYCLES / freq
            room = min(t_center, ramp.duration - t_center)
            if a:
                # Cover speed_step where the ramp leaves room for it
                needed = max(needed, 0.5 * RAMP_WINDOW_PERIODS / base_freq)
                half = max(needed, min(0.5 * speed_step / abs(a), room))
            else:
                half = room
            lo, hi = t_center - half, t_center + half
            sl = window_slice(times, ramp.start_time + lo, ramp.start_time + hi)
            count = sl.stop - sl.start
            if (
                half < needed
                or lo < 0.0
                or hi > ramp.duration
                or count < MIN_WINDOW_SAMPLES
            ):
                results.append(HarmonicResult(harmonic, freq, None, None, None))
                continue
            win_times = times[sl]
            sample_rate = (count - 1) / (win_times[-1] - win_times[0])
            max_freq = max_freq_ratio * sample_rate
            if freq > max_freq:
                results.append(HarmonicResult(harmonic, freq, None, None, None))
                continue
            t = win_times - ramp.start_time
            pos = ramp.start_pos + ramp.motor_dir * (
                ramp.start_speed * t + 0.5 * a * t * t
            )
            angle = 2.0 * math.pi * pos / period
            orders, noise_orders = _choose_orders(
                harmonic, harmonics, belt, 2.0 * half * base_freq
            )
            orders = [o for o in orders if o * base_freq <= max_freq]
            noise_orders = [o for o in noise_orders if o in orders]
            amps = fit_orders(win_times, accel[sl], angle, orders)
            amp = amps[harmonic]
            noise = None
            if noise_orders:
                noise_amp = sum(abs(amps[o]) for o in noise_orders) / len(
                    noise_orders
                )
                noise = _accel_to_disp(noise_amp, freq)
            # The acceleration of a displacement D*cos(h*angle) is
            # -omega^2 * D * cos(h*angle)
            results.append(
                HarmonicResult(
                    harmonic,
                    freq,
                    abs(amp),
                    _accel_to_disp(abs(amp), freq),
                    noise,
                    -_accel_to_disp(amp, freq),
                )
            )
        rows.extend((speed, ramp.motor_dir, res) for res in results)
    return rows


def merge_rows(rows):
    """Average rows of the same speed, direction and harmonic, such as from
    ramps up and down, whose lags behind the sweep cancel; the phasors are
    averaged coherently. Results that could not be measured (a window not
    fitting in one of two overlapping ramps) are left out; a speed none
    measured stays None."""
    groups = collections.OrderedDict()
    for speed, motor_dir, res in rows:
        groups.setdefault((speed, motor_dir, res.harmonic), []).append(res)
    merged = []
    for (speed, motor_dir, harmonic), results in groups.items():
        valid = [r for r in results if r.disp is not None]
        if not valid:
            merged.append((speed, motor_dir, results[0]))
            continue
        n = float(len(valid))
        phasors = [r.phasor for r in valid if r.phasor is not None]
        merged.append(
            (
                speed,
                motor_dir,
                HarmonicResult(
                    harmonic,
                    valid[0].freq,
                    sum(r.accel for r in valid) / n,
                    sum(r.disp for r in valid) / n,
                    sum(r.noise_disp for r in valid) / n,
                    sum(phasors) / len(phasors) if phasors else None,
                ),
            )
        )
    return merged


def peak_per_harmonic(rows, min_snr=MIN_PEAK_SNR):
    """Pick the strongest measurement per (harmonic, direction).

    rows are (speed, direction, HarmonicResult) tuples; the result maps
    (harmonic, direction) to the row with the largest displacement. Rows
    less than min_snr times above their noise level are skipped: at low
    frequencies accelerometer noise alone converts to a large displacement.
    """
    peaks = {}
    for row in rows:
        result = row[2]
        if result.disp is None:
            continue
        if (
            result.noise_disp is not None
            and result.disp < min_snr * result.noise_disp
        ):
            continue
        key = (result.harmonic, row[1])
        if key not in peaks or result.disp > peaks[key][2].disp:
            peaks[key] = row
    return peaks
