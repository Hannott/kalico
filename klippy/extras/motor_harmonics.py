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
MIN_WINDOW_CYCLES = 5
MIN_WINDOW_SAMPLES = 64
MIN_PEAK_SNR = 3.0
# Electrical periods in a ramp window: the noise level is measured half an
# order from each harmonic, which a shorter window cannot separate
RAMP_WINDOW_PERIODS = 8
# Largest speed change, relative to the speed, across one ramp window
RAMP_WINDOW_SPAN = 0.25

HarmonicResult = collections.namedtuple(
    "HarmonicResult", ("harmonic", "freq", "accel", "disp", "noise_disp")
)
# A constant acceleration part of a test move: from start_time (print time)
# the motor moves start_speed * t + accel * t^2 / 2 in motor_dir.
Ramp = collections.namedtuple(
    "Ramp", ("motor_dir", "start_time", "start_speed", "accel", "duration")
)


def plan_speeds(speed_min, speed_max, speed_step):
    count = int(math.floor((speed_max - speed_min) / speed_step + 1e-9)) + 1
    return [speed_min + i * speed_step for i in range(count)]


def motor_travel(speed, accel, cruise_time):
    """Motor travel of one test move: accelerate, cruise, decelerate."""
    return speed * speed / accel + speed * cruise_time


def project_samples(samples, dir_x, dir_y):
    """Split (time, x, y, z) accelerometer samples into times and the
    acceleration component along the unit XY vector (dir_x, dir_y)."""
    data = np.asarray(samples, dtype=float)
    return data[:, 0], data[:, 1] * dir_x + data[:, 2] * dir_y


def window_slice(times, start_time, end_time):
    lo = np.searchsorted(times, start_time, side="left")
    hi = np.searchsorted(times, end_time, side="right")
    return slice(lo, hi)


def phase_amplitude(times, values, phase):
    """Complex amplitude of values along a known phase (radians per sample).

    Uses a Hann window over the sample times, so windows need not be a
    whole number of periods. Each sample is weighted by the time it covers,
    capped at the typical sample interval so that the samples next to a
    dropped accelerometer batch do not stand in for the whole gap.
    For values = A*cos(phase + phi) it returns about A*exp(1j*phi), also
    when the phase does not advance at a constant rate.
    """
    t = times - times[0]
    duration = t[-1]
    intervals = np.diff(t)
    nominal = np.median(intervals)
    covered = np.minimum(intervals, nominal)
    spacing = 0.5 * (np.append(covered, 0.0) + np.insert(covered, 0, 0.0))
    window = (0.5 - 0.5 * np.cos(2.0 * math.pi * t / duration)) * spacing
    centered = values - np.sum(window * values) / np.sum(window)
    return (
        2.0 * np.sum(window * centered * np.exp(-1j * phase)) / np.sum(window)
    )


def harmonic_amplitude(times, values, freq, t_ref=None):
    """Complex amplitude of values at freq (single-bin DFT), relative to the
    first sample time, or to t_ref when given."""
    t_phase = times - (times[0] if t_ref is None else t_ref)
    return phase_amplitude(times, values, 2.0 * math.pi * freq * t_phase)


def _accel_to_disp(accel, freq):
    return accel / (2.0 * math.pi * freq) ** 2


def analyze_window(times, accel, speed, period, harmonics):
    """Measure each harmonic of one constant-speed window.

    Returns a HarmonicResult per harmonic. accel is in mm/s^2; disp and
    noise_disp are displacement amplitudes in mm. noise_disp is the mean
    level at the half orders either side of the harmonic (h-0.5, h+0.5),
    which no motor harmonic falls on. Harmonics that the window cannot
    resolve have accel, disp and noise_disp set to None.
    """
    count = len(times)
    duration = times[-1] - times[0] if count > 1 else 0.0
    sample_rate = (count - 1) / duration if duration > 0.0 else 0.0
    max_freq = MAX_FREQ_RATIO * sample_rate
    base_freq = speed / period

    def resolvable(freq):
        return (
            count >= MIN_WINDOW_SAMPLES
            and freq * duration >= MIN_WINDOW_CYCLES
            and freq <= max_freq
        )

    results = []
    for harmonic in harmonics:
        freq = harmonic * base_freq
        if not resolvable(freq):
            results.append(HarmonicResult(harmonic, freq, None, None, None))
            continue
        amp = abs(harmonic_amplitude(times, accel, freq))
        noise_freqs = [
            (harmonic + offset) * base_freq
            for offset in (-0.5, 0.5)
            if resolvable((harmonic + offset) * base_freq)
        ]
        noise = None
        if noise_freqs:
            noise_amp = sum(
                abs(harmonic_amplitude(times, accel, f)) for f in noise_freqs
            ) / len(noise_freqs)
            noise = _accel_to_disp(noise_amp, freq)
        results.append(
            HarmonicResult(
                harmonic, freq, amp, _accel_to_disp(amp, freq), noise
            )
        )
    return results


# A new ramp segment must reach at least this fraction higher than the
# previous one; plan_ramps() otherwise converges without reaching the top.
MIN_RAMP_GAIN = 0.01
# Hard limit on segments, so no input can grow the plan without bound
MAX_RAMP_SEGMENTS = 32


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
        # Near the top of what max_travel allows, each segment gains less
        # than the last and v2 converges below top, so require real progress
        # over the previous segment as well as over v1.
        stalled = len(segments) >= MAX_RAMP_SEGMENTS or (
            segments and v2 - segments[-1][1] < MIN_RAMP_GAIN * v2
        )
        if v2 - v1 <= lead(v2, ramp_accel) or stalled:
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


def analyze_ramp(times, accel, ramp, period, harmonics, speeds, speed_step):
    """Measure each harmonic around each of speeds that the ramp passes.

    Demodulates against the motor's electrical angle, which the planned
    motion gives at every sample, so the frequency may change during the
    window. A window spans speed_step of the ramp where the ramp has room
    for it, and at least MIN_WINDOW_CYCLES cycles of the harmonic and
    RAMP_WINDOW_PERIODS electrical periods; harmonics whose window does not
    fit in the ramp get None. Returns (speed, motor_dir, result) rows
    with analyze_window() results.
    """
    a = ramp.accel
    rows = []
    for speed in speeds:
        t_center = (speed - ramp.start_speed) / a
        if not 0.0 <= t_center <= ramp.duration:
            continue
        base_freq = speed / period
        results = []
        for harmonic in harmonics:
            freq = harmonic * base_freq
            # Cover speed_step where the ramp leaves room for it
            needed = 0.5 * max(
                MIN_WINDOW_CYCLES / freq, RAMP_WINDOW_PERIODS / base_freq
            )
            room = min(t_center, ramp.duration - t_center)
            half = max(needed, min(0.5 * speed_step / abs(a), room))
            lo, hi = t_center - half, t_center + half
            sl = window_slice(times, ramp.start_time + lo, ramp.start_time + hi)
            count = sl.stop - sl.start
            if lo < 0.0 or hi > ramp.duration or count < MIN_WINDOW_SAMPLES:
                results.append(HarmonicResult(harmonic, freq, None, None, None))
                continue
            win_times = times[sl]
            if freq > MAX_FREQ_RATIO * (count - 1) / (
                win_times[-1] - win_times[0]
            ):
                results.append(HarmonicResult(harmonic, freq, None, None, None))
                continue
            t = win_times - ramp.start_time
            angle = (
                2.0
                * math.pi
                * (ramp.start_speed * t + 0.5 * a * t * t)
                / period
            )
            amp = abs(phase_amplitude(win_times, accel[sl], harmonic * angle))
            noise_amp = 0.5 * sum(
                abs(phase_amplitude(win_times, accel[sl], order * angle))
                for order in (harmonic - 0.5, harmonic + 0.5)
            )
            results.append(
                HarmonicResult(
                    harmonic,
                    freq,
                    amp,
                    _accel_to_disp(amp, freq),
                    _accel_to_disp(noise_amp, freq),
                )
            )
        rows.extend((speed, ramp.motor_dir, res) for res in results)
    return rows


def merge_rows(rows):
    """Average rows of the same speed, direction and harmonic, such as from
    ramps up and down, whose lags behind the sweep cancel. Results that
    could not be measured (a window not fitting in one of two overlapping
    ramps) are left out; a speed none measured stays None."""
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


def estimate_correction(r0, r_real, r_imag, c0, delta):
    """One step towards the correction that cancels a harmonic.

    The vibration at a harmonic responds linearly to a small change of its
    correction c (complex: magnitude * exp(1j * phase)): r(c) = r0 + G*(c -
    c0). r_real and r_imag are measured at c0 + delta and c0 + 1j*delta.
    Returns (new correction, G, nonlinearity), where nonlinearity compares
    the two estimates of G.
    """
    g_real = (r_real - r0) / delta
    g_imag = (r_imag - r0) / (1j * delta)
    gain = 0.5 * (g_real + g_imag)
    if abs(gain) == 0.0:
        return c0, gain, float("inf")
    nonlinearity = abs(g_real - g_imag) / abs(gain)
    return c0 - r0 / gain, gain, nonlinearity


def pick_calibration_speeds(rows, harmonic, count, min_snr=5.0, min_disp=1e-4):
    """Up to count speeds where a harmonic vibrates most in both directions
    (above min_disp in mm), kept apart so that they cover its peaks."""
    by_speed = {}
    for speed, motor_dir, res in rows:
        if res.harmonic != harmonic or res.disp is None:
            continue
        floor = min_disp
        if res.noise_disp is not None:
            floor = max(floor, min_snr * res.noise_disp)
        if res.disp >= floor:
            by_speed.setdefault(speed, {})[motor_dir] = res.disp
    usable = {s: min(d.values()) for s, d in by_speed.items() if len(d) == 2}
    if not usable:
        return []
    gap = (max(usable) - min(usable)) / (2.0 * count)
    picked = []
    for speed in sorted(usable, key=lambda s: (-usable[s], s)):
        if len(picked) == count:
            break
        if all(abs(speed - p) >= gap for p in picked):
            picked.append(speed)
    return sorted(picked)


def _weighted_step(c, vib, gains, weights):
    num = sum(weights[s] * gains[s].conjugate() * vib[s] for s in vib)
    den = sum(weights[s] * abs(gains[s]) ** 2 for s in vib)
    return c - num / den if den else c


def _minimax_step(c, vib, gains, iterations=100):
    """Correction that leaves the least worst vibration over all speeds,
    predicted from vibration vib at correction c (Lawson's algorithm:
    least squares, reweighted towards the speeds left worst)."""
    weights = dict.fromkeys(vib, 1.0)
    step = _weighted_step(c, vib, gains, weights)
    for _ in range(iterations):
        left = {
            s: weights[s] * abs(vib[s] + gains[s] * (step - c)) for s in vib
        }
        total = sum(left.values())
        if not total:
            break
        weights = {s: w / total for s, w in left.items()}
        step = _weighted_step(c, vib, gains, weights)
    return step


def multi_speed_correction(d0, d_real, d_imag, c0, delta):
    """Correction that best cancels a harmonic over several speeds.

    Each argument d* maps speed to the complex vibration measured with the
    correction at c0, c0 + delta and c0 + 1j*delta. Every speed responds
    linearly but differently, so a single correction minimises the worst
    vibration left at any of them. Returns (correction, gains).
    """
    gains = {
        s: 0.5
        * ((d_real[s] - d0[s]) / delta + (d_imag[s] - d0[s]) / (1j * delta))
        for s in d0
    }
    return _minimax_step(c0, d0, gains), gains


def refine_correction(c, vib, gains):
    """Step from correction c, which gave vibration vib."""
    return _minimax_step(c, vib, gains)


def worst_vibration(vib):
    return max(abs(v) for v in vib.values())
