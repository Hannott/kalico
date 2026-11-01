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

HarmonicResult = collections.namedtuple(
    "HarmonicResult", ("harmonic", "freq", "accel", "disp", "noise_disp")
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


def harmonic_amplitude(times, values, freq):
    """Complex amplitude of values at freq (single-bin DFT).

    Uses a Hann window over the sample times, so windows need not be a
    whole number of periods. Each sample is weighted by the time it covers,
    capped at the typical sample interval so that the samples next to a
    dropped accelerometer batch do not stand in for the whole gap.
    For values = A*cos(2*pi*freq*t + phi) it returns about A*exp(1j*phi),
    relative to the first sample time.
    """
    t = times - times[0]
    duration = t[-1]
    intervals = np.diff(t)
    nominal = np.median(intervals)
    covered = np.minimum(intervals, nominal)
    spacing = 0.5 * (np.append(covered, 0.0) + np.insert(covered, 0, 0.0))
    window = (0.5 - 0.5 * np.cos(2.0 * math.pi * t / duration)) * spacing
    centered = values - np.sum(window * values) / np.sum(window)
    phasor = np.exp(-2j * math.pi * freq * t)
    return 2.0 * np.sum(window * centered * phasor) / np.sum(window)


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
