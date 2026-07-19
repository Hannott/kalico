# Automatic calibration of input shapers
#
# Copyright (C) 2020-2024  Dmitry Butyugin <dmbutyugin@google.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import collections
import importlib
import math
import multiprocessing
import traceback

from . import shaper_defs

MIN_FREQ = 5.0
MAX_FREQ = 200.0
WINDOW_T_SEC = 0.5
MAX_SHAPER_FREQ = 300.0

TEST_DAMPING_RATIOS = [0.075, 0.1, 0.15]

AUTOTUNE_SHAPERS = [
    "smooth_zv",
    "smooth_mzv",
    "smooth_ei",
    "smooth_2hump_ei",
    "smooth_zvd_ei",
    "smooth_si",
    "mzv",
    "ei",
    "2hump_ei",
]

# Base shapers considered for multimode auto-tuning. Restricted to the
# low-impulse-count bases: a multimode shaper convolves multiple copies of
# a base, so higher-order bases (zvd, 2hump_ei, 3hump_ei) produce so much
# smoothing that they can never win the recommendation, and only clutter
# the output and slow the search.
MULTIMODE_AUTOTUNE_BASES = ["zv", "mzv", "ei"]

# Largest number of resonance peaks multimode auto-tuning will search for.
# The search cost grows geometrically with the number of modes (one extra
# frequency ratio dimension per mode beyond the first, plus a per-base-combo
# multiplier), so this bounds it to what real printers actually show: more
# than a handful of distinct, well-separated, significant resonances is rare.
# Manually configuring more than this many modes (shaper_freq_<axis> with
# more entries) is still fully supported -- only auto-detection is capped.
MULTIMODE_MAX_PEAKS = 4

######################################################################
# Frequency response calculation and shaper auto-tuning
######################################################################


class CalibrationData:
    def __init__(self, freq_bins, psd_sum, psd_x, psd_y, psd_z):
        self.freq_bins = freq_bins
        self.psd_sum = psd_sum
        self.psd_x = psd_x
        self.psd_y = psd_y
        self.psd_z = psd_z
        self._psd_list = [self.psd_sum, self.psd_x, self.psd_y, self.psd_z]
        self._psd_map = {
            "x": self.psd_x,
            "y": self.psd_y,
            "z": self.psd_z,
            "all": self.psd_sum,
        }

    def add_data(self, other):
        np = self.numpy
        for psd, other_psd in zip(self._psd_list, other._psd_list):
            # `other` data may be defined at different frequency bins,
            # interpolating to fix that.
            other_normalized = np.interp(
                self.freq_bins, other.freq_bins, other_psd
            )
            psd[:] = np.maximum(psd, other_normalized)

    def set_numpy(self, numpy):
        self.numpy = numpy

    def normalize_to_frequencies(self):
        freq_bins = self.freq_bins
        for psd in self._psd_list:
            # Avoid division by zero errors and remove low-frequency noise
            psd *= self.numpy.tanh(0.5 / MIN_FREQ * freq_bins) / (
                freq_bins + 0.1
            )

    def get_psd(self, axis="all"):
        return self._psd_map[axis]


CalibrationResult = collections.namedtuple(
    "CalibrationResult",
    (
        "name",
        "freq",
        "vals",
        "vibrs",
        "smoothing",
        "score",
        "max_accel",
        # Only set for multimode candidates (N >= 2 peaks): the base shaper,
        # design frequency, and damping ratio used at each peak, as
        # same-length tuples (one entry per peak). `freq` above is always
        # freqs[0], the reference peak multimode's local search is centered
        # on.
        "bases",
        "freqs",
        "damping_ratios",
    ),
    defaults=(None, None, None),
)


def _trapz(np, y, dx):
    # np.trapz was removed in numpy 2.0 in favor of np.trapezoid
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, dx=dx)
    return np.trapz(y, dx=dx)


def step_response(np, t, omega, damping_ratio):
    t = np.maximum(t, 0.0)
    omega = np.swapaxes(np.array(omega, ndmin=2), 0, 1)
    damping = damping_ratio * omega
    omega_d = omega * math.sqrt(1.0 - damping_ratio**2)
    phase = math.acos(damping_ratio)
    return 1.0 - np.exp((-damping * t)) * np.sin((omega_d * t) + phase) * (
        1.0 / math.sin(phase)
    )


def step_response_velocity(np, t, omega, damping_ratio):
    # Analytic derivative of step_response:
    # v(t) = omega * exp(-zeta*omega*t) * sin(omega_d*t) / sqrt(1-zeta^2)
    t = np.maximum(t, 0.0)
    omega = np.swapaxes(np.array(omega, ndmin=2), 0, 1)
    df = math.sqrt(1.0 - damping_ratio**2)
    return (
        np.exp(-damping_ratio * omega * t)
        * np.sin(omega * df * t)
        * (omega / df)
    )


def step_response_min_velocity(damping_ratio):
    d2 = damping_ratio * damping_ratio
    d_r = damping_ratio / math.sqrt(1.0 - d2)
    # Analytical formula for the minimum was obtained using Maxima system
    t = 0.5 * math.atan2(2.0 * d2, (2.0 * d2 - 1.0) * d_r) + math.pi
    phase = math.acos(damping_ratio)
    v = math.exp(-d_r * t) * (d_r * math.sin(t + phase) - math.cos(t + phase))
    return v


def _refined_min(np, v):
    # Refine the per-row discrete minimum with a parabola through the
    # 3 points around it (near-exact for smooth extrema)
    j = np.argmin(v, axis=-1)
    rows = np.arange(v.shape[0])
    jc = np.clip(j, 1, v.shape[-1] - 2)
    v_m, v_0, v_p = v[rows, jc - 1], v[rows, jc], v[rows, jc + 1]
    denom = v_m - 2.0 * v_0 + v_p
    interior = (j == jc) & (denom > 0.0)
    adj = np.where(interior, (v_p - v_m) ** 2 / (8.0 * denom), 0.0)
    return v[rows, j] - adj


def estimate_shaper_old(np, shaper, test_damping_ratio, test_freqs):
    A, T = np.asarray(shaper[0]), np.asarray(shaper[1])
    inv_D = 1.0 / A.sum()

    omega = 2.0 * math.pi * np.asarray(test_freqs)
    damping = test_damping_ratio * omega
    omega_d = omega * math.sqrt(1.0 - test_damping_ratio**2)
    W = A * np.exp(np.outer(-damping, (T[-1] - T)))
    S = W * np.sin(np.outer(omega_d, T))
    C = W * np.cos(np.outer(omega_d, T))
    return np.sqrt(S.sum(axis=1) ** 2 + C.sum(axis=1) ** 2) * inv_D


def estimate_shaper(np, shaper, test_damping_ratio, test_freqs):
    A, T = np.asarray(shaper[0]), np.asarray(shaper[1])
    inv_D = 1.0 / A.sum()
    n = len(T)
    t_s = T[-1] - T[0]

    test_freqs = np.asarray(test_freqs)
    t_start = T[0]
    t_end = T[-1] + 2.0 * np.maximum(1.0 / test_freqs[test_freqs > 0.0], t_s)
    n_t = 1000
    unity_range = np.linspace(0.0, 1.0, n_t)
    time = (t_end[:, np.newaxis] - t_start) * unity_range + t_start

    min_v = -step_response_min_velocity(test_damping_ratio)

    omega = 2.0 * math.pi * test_freqs[test_freqs > 0.0]

    velocity = np.zeros(shape=(omega.shape[0], time.shape[-1]))
    # The velocity has kinks at the impulse times, evaluate it there exactly
    kink_velocity = np.zeros(shape=(omega.shape[0], n))
    for i in range(n):
        velocity += A[i] * step_response_velocity(
            np, time - T[i], omega, test_damping_ratio
        )
        kink_velocity += A[i] * step_response_velocity(
            np, T - T[i], omega, test_damping_ratio
        )
    # step_response_min_velocity is normalized per unit omega
    velocity *= inv_D / omega[:, np.newaxis]
    kink_velocity *= inv_D / omega[:, np.newaxis]
    velocity_min = np.minimum(
        _refined_min(np, velocity), kink_velocity.min(axis=-1)
    )
    res = np.zeros(shape=test_freqs.shape)
    res[test_freqs > 0.0] = -velocity_min / min_v
    res[test_freqs <= 0.0] = 1.0
    return res


def estimate_smoother_old(np, smoother, test_damping_ratio, test_freqs):
    C, t_sm = smoother[0], smoother[1]
    hst = t_sm * 0.5

    test_freqs = np.asarray(test_freqs)
    omega = 2.0 * math.pi * test_freqs
    damping = test_damping_ratio * omega
    omega_d = omega * math.sqrt(1.0 - test_damping_ratio**2)

    n_t = max(100, 100 * round(t_sm * np.max(test_freqs)))
    t, dt = np.linspace(0.0, t_sm, n_t, retstep=True)
    w = np.zeros(shape=t.shape)
    for c in C[::-1]:
        w = w * (t - hst) + c

    E = w * np.exp(np.outer(damping, (t - t_sm)))
    C = np.cos(np.outer(omega_d, (t - t_sm)))
    S = np.sin(np.outer(omega_d, (t - t_sm)))
    return np.sqrt(
        _trapz(np, E * C, dx=dt) ** 2 + _trapz(np, E * S, dx=dt) ** 2
    )


def estimate_smoother(np, smoother, test_damping_ratio, test_freqs):
    C, t_sm = smoother[0], smoother[1]
    hst = t_sm * 0.5

    test_freqs = np.asarray(test_freqs)

    t_start = -t_sm
    t_end = hst + np.maximum(1.5 / test_freqs[test_freqs > 0.0], 2.0 * t_sm)
    n_t = 1000
    unity_range = np.linspace(0.0, 1.0, n_t)
    time = (t_end[:, np.newaxis] - t_start) * unity_range + t_start
    dt = (time[:, -1] - time[:, 0]) / n_t
    tau = np.copy(time)
    tau[time > hst] = 0.0
    tau[time < -hst] = 0.0

    w = np.zeros(shape=tau.shape)
    for c in C[::-1]:
        w = w * tau + c
    w[time > hst] = 0.0
    w[time < -hst] = 0.0
    norms = (w * dt[:, np.newaxis]).sum(axis=-1)

    min_v = -step_response_min_velocity(test_damping_ratio)

    omega = 2.0 * math.pi * test_freqs[test_freqs > 0.0]

    wm = np.count_nonzero(time < -hst, axis=-1).min()
    wp = np.count_nonzero(time <= hst, axis=-1).max()

    def get_windows(m, wl):
        nrows = m.shape[-1] - wl + 1
        n = m.strides[-1]
        return np.lib.stride_tricks.as_strided(
            m, shape=(m.shape[0], nrows, wl), strides=(m.strides[0], n, n)
        )

    # The velocity of the smoothed response is the smoother convolved
    # with the analytic velocity of the step response
    s_v = (
        step_response_velocity(np, time, omega, test_damping_ratio)
        / omega[:, np.newaxis]
    )
    w_dt = w[:, wm:wp] * (np.reciprocal(norms) * dt)[:, np.newaxis]
    velocity = np.einsum("ijk,ik->ij", get_windows(s_v, wp - wm), w_dt[:, ::-1])
    res = np.zeros(shape=test_freqs.shape)
    # The smoothed velocity is C^1, a parabolic refinement of the discrete
    # minimum is sufficient
    res[test_freqs > 0.0] = -_refined_min(np, velocity) / min_v
    res[test_freqs <= 0.0] = 1.0
    return res


class ShaperCalibrate:
    def __init__(self, printer):
        self.printer = printer
        self.error = printer.command_error if printer else Exception
        try:
            self.numpy = importlib.import_module("numpy")
        except ImportError:
            raise self.error(
                "Failed to import `numpy` module, make sure it was "
                "installed via `~/klippy-env/bin/pip install` (refer to "
                "docs/Measuring_Resonances.md for more details)."
            )
        self._smoother_integrals_cache = {}
        self._vibr_threshold_cache = None

    def background_process_exec(self, method, args):
        if self.printer is None:
            return method(*args)
        import queuelogger

        parent_conn, child_conn = multiprocessing.Pipe()

        def wrapper():
            queuelogger.clear_bg_logging()
            try:
                res = method(*args)
            except:
                child_conn.send((True, traceback.format_exc()))
                child_conn.close()
                return
            child_conn.send((False, res))
            child_conn.close()

        # Start a process to perform the calculation
        calc_proc = multiprocessing.Process(target=wrapper)
        calc_proc.daemon = True
        calc_proc.start()
        # Wait for the process to finish
        reactor = self.printer.get_reactor()
        gcode = self.printer.lookup_object("gcode")
        eventtime = last_report_time = reactor.monotonic()
        while calc_proc.is_alive():
            if eventtime > last_report_time + 5.0:
                last_report_time = eventtime
                gcode.respond_info("Wait for calculations..", log=False)
            eventtime = reactor.pause(eventtime + 0.1)
        # Return results
        is_err, res = parent_conn.recv()
        if is_err:
            raise self.error("Error in remote calculation: %s" % (res,))
        calc_proc.join()
        parent_conn.close()
        return res

    def background_process_exec_parallel(self, method, args_list):
        # Like background_process_exec, but runs one child process per
        # args tuple with up to cpu_count() of them concurrently -- the
        # per-shaper fits are independent and CPU-bound, so running them
        # back to back leaves all but one core idle for the whole
        # calibration. Results are returned in args_list order.
        if not args_list:
            return []
        if self.printer is None:
            return [method(*args) for args in args_list]
        import queuelogger

        try:
            max_workers = multiprocessing.cpu_count()
        except NotImplementedError:
            max_workers = 2
        max_workers = max(1, min(max_workers, len(args_list)))

        def start_job(args):
            parent_conn, child_conn = multiprocessing.Pipe()

            def wrapper():
                queuelogger.clear_bg_logging()
                try:
                    res = method(*args)
                except:
                    child_conn.send((True, traceback.format_exc()))
                    child_conn.close()
                    return
                child_conn.send((False, res))
                child_conn.close()

            proc = multiprocessing.Process(target=wrapper)
            proc.daemon = True
            proc.start()
            return proc, parent_conn

        reactor = self.printer.get_reactor()
        gcode = self.printer.lookup_object("gcode")
        eventtime = last_report_time = reactor.monotonic()
        results = [None] * len(args_list)
        pending = list(enumerate(args_list))
        active = {}
        error = None
        while pending or active:
            while pending and len(active) < max_workers and error is None:
                idx, args = pending.pop(0)
                active[idx] = start_job(args)
            if error is not None:
                # An earlier job failed: stop feeding new work, just
                # drain what is already running.
                pending = []
            if not active:
                break
            if eventtime > last_report_time + 5.0:
                last_report_time = eventtime
                gcode.respond_info("Wait for calculations..", log=False)
            eventtime = reactor.pause(eventtime + 0.1)
            for idx in list(active):
                proc, conn = active[idx]
                if conn.poll():
                    is_err, res = conn.recv()
                    conn.close()
                    proc.join()
                    del active[idx]
                    if is_err:
                        error = error or res
                    else:
                        results[idx] = res
                elif not proc.is_alive():
                    # Child died without reporting a result
                    conn.close()
                    proc.join()
                    del active[idx]
                    error = error or (
                        "Background calculation process terminated unexpectedly"
                    )
        if error is not None:
            raise self.error("Error in remote calculation: %s" % (error,))
        return results

    def _split_into_windows(self, x, window_size, overlap):
        # Memory-efficient algorithm to split an input 'x' into a series
        # of overlapping windows
        step_between_windows = window_size - overlap
        n_windows = (x.shape[-1] - overlap) // step_between_windows
        shape = (window_size, n_windows)
        strides = (x.strides[-1], step_between_windows * x.strides[-1])
        return self.numpy.lib.stride_tricks.as_strided(
            x, shape=shape, strides=strides, writeable=False
        )

    def _psd(self, x, fs, nfft):
        # Calculate power spectral density (PSD) using Welch's algorithm
        np = self.numpy
        window = np.kaiser(nfft, 6.0)
        # Compensation for windowing loss
        scale = 1.0 / (window**2).sum()

        # Split into overlapping windows of size nfft
        overlap = nfft // 2
        x = self._split_into_windows(x, nfft, overlap)

        # First detrend, then apply windowing function
        x = window[:, None] * (x - np.mean(x, axis=0))

        # Calculate frequency response for each window using FFT
        result = np.fft.rfft(x, n=nfft, axis=0)
        result = np.conjugate(result) * result
        result *= scale / fs
        # For one-sided FFT output the response must be doubled, except
        # the last point for unpaired Nyquist frequency (assuming even nfft)
        # and the 'DC' term (0 Hz)
        result[1:-1, :] *= 2.0

        # Welch's algorithm: average response over windows
        psd = result.real.mean(axis=-1)

        # Calculate the frequency bins
        freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
        return freqs, psd

    def calc_freq_response(self, raw_values):
        np = self.numpy
        if raw_values is None:
            return None
        if isinstance(raw_values, np.ndarray):
            data = raw_values
        else:
            samples = raw_values.get_samples()
            if not samples:
                return None
            data = np.array(samples)

        N = data.shape[0]
        T = data[-1, 0] - data[0, 0]
        SAMPLING_FREQ = N / T
        # Round up to the nearest power of 2 for faster FFT
        M = 1 << int(SAMPLING_FREQ * WINDOW_T_SEC - 1).bit_length()
        if N <= M:
            return None

        # Calculate PSD (power spectral density) of vibrations per
        # frequency bins (the same bins for X, Y, and Z)
        fx, px = self._psd(data[:, 1], SAMPLING_FREQ, M)
        fy, py = self._psd(data[:, 2], SAMPLING_FREQ, M)
        fz, pz = self._psd(data[:, 3], SAMPLING_FREQ, M)
        return CalibrationData(fx, px + py + pz, px, py, pz)

    def process_accelerometer_data(self, data):
        calibration_data = self.background_process_exec(
            self.calc_freq_response, (data,)
        )
        if calibration_data is None:
            raise self.error(
                "Internal error processing accelerometer data %s" % (data,)
            )
        calibration_data.set_numpy(self.numpy)
        return calibration_data

    def _calc_vibr_threshold(self, freq_bins, psd):
        # Everything here depends only on (freq_bins, psd), which are fixed
        # across the hundreds/thousands of per-candidate-frequency calls a
        # single fit makes -- so the result is cached on the exact array
        # objects (identity compare; the cache holds references, so the ids
        # cannot be recycled while cached) and computed once per fit.
        cached = self._vibr_threshold_cache
        if cached is not None and cached[0] is freq_bins and cached[1] is psd:
            return cached[2], cached[3]
        np = self.numpy
        # Mainline Klipper's flat acceptance threshold: the input shaper can
        # only reduce the amplitude of vibrations by SHAPER_VIBRATION_REDUCTION
        # times, so vibrations below that level can be ignored.
        vibr_threshold = np.full_like(
            psd, psd.max() / shaper_defs.SHAPER_VIBRATION_REDUCTION
        )
        # That single threshold is calibrated to the dominant peak alone, so
        # a shaper that leaves a materially weaker but still significant
        # secondary resonance unshaped can score near-perfect (its vals*psd
        # never clears a threshold set by a much taller, unrelated peak) --
        # hiding exactly the case a multimode shaper exists to solve. Near
        # each genuinely detected resonance peak, lower the threshold to that
        # peak's own acceptance level so leaving it unshaped costs score.
        #
        # Only peaks passing the same prominence/separation criteria as the
        # multimode auto-detection count: lowering the threshold near mere
        # local maxima (e.g. noise ridges or a taller peak's shoulder) forces
        # near-total suppression of background content that no realizable
        # shaper could deliver, and lets the noise floor out-vote the actual
        # resonances in the fit. For a single-peak PSD (the common case) the
        # only detected peak is the dominant one and its own level equals the
        # global threshold, so scoring matches mainline exactly.
        peaks = self._detect_resonance_peaks(
            freq_bins,
            psd,
            MIN_FREQ,
            freq_bins.max(),
            max_peaks=MULTIMODE_MAX_PEAKS,
        )
        for peak_freq in peaks:
            i = int(np.argmin(np.abs(freq_bins - peak_freq)))
            band = np.abs(freq_bins - freq_bins[i]) <= 12.0
            vibr_threshold[band] = np.minimum(
                vibr_threshold[band],
                psd[i] / shaper_defs.SHAPER_VIBRATION_REDUCTION,
            )
        all_vibrations = np.maximum(psd - vibr_threshold, 0).sum()
        self._vibr_threshold_cache = (
            freq_bins,
            psd,
            vibr_threshold,
            all_vibrations,
        )
        return vibr_threshold, all_vibrations

    def _estimate_remaining_vibrations(self, freq_bins, vals, psd):
        # Calculate the acceptable level of remaining vibrations.
        # Note that these are not true remaining vibrations, but rather
        # just a score to compare different shapers between each other.
        vibr_threshold, all_vibrations = self._calc_vibr_threshold(
            freq_bins, psd
        )
        remaining_vibrations = self.numpy.maximum(
            vals * psd - vibr_threshold, 0
        ).sum()
        return remaining_vibrations / all_vibrations

    def _get_shaper_smoothing(self, shaper, accel=5000, scv=5.0):
        half_accel = accel * 0.5

        A, T = shaper
        inv_D = 1.0 / sum(A)
        n = len(T)
        ts = shaper_defs.get_shaper_offset(A, T)

        # Calculate offset for 90 and 180 degrees turn
        offset_90_x = offset_90_y = offset_180 = 0.0
        for i in range(n):
            if T[i] >= ts:
                # Calculate offset for one of the axes
                offset_90_x += (
                    A[i] * (scv + half_accel * (T[i] - ts)) * (T[i] - ts)
                )
            else:
                offset_90_y += (
                    A[i] * (scv - half_accel * (T[i] - ts)) * (T[i] - ts)
                )
            offset_180 += A[i] * half_accel * (T[i] - ts) ** 2
        offset_90 = inv_D * math.sqrt(offset_90_x**2 + offset_90_y**2)
        offset_180 *= inv_D
        return max(offset_90, abs(offset_180))

    def _calc_smoother_integrals(self, smoother):
        # The smoothing offsets are linear in the accel- and scv-dependent
        # terms, so the smoother geometry integrals below depend only on the
        # smoother itself. Compute them once and cache, since find_max_accel
        # evaluates the smoothing for many accel values of the same smoother.
        cache_key = (smoother[1], tuple(smoother[0]))
        cached = self._smoother_integrals_cache.get(cache_key)
        if cached is not None:
            return cached
        np = self.numpy
        C, t_sm = smoother
        hst = 0.5 * t_sm
        t, dt = np.linspace(-hst, hst, 100, retstep=True)
        w = np.zeros(shape=t.shape)
        for c in C[::-1]:
            w = w * (-t) + c
        w *= 1.0 / _trapz(np, w, dx=dt)
        t -= _trapz(np, t * w, dx=dt)
        tw = t * w
        t2w = t * tw
        pos = t >= 0
        neg = t < 0
        integrals = (
            _trapz(np, t2w, dx=dt),  # full-range int(t^2 w), for offset_180
            _trapz(np, tw[pos], dx=dt),  # int_{t>=0}(t w)
            _trapz(np, t2w[pos], dx=dt),  # int_{t>=0}(t^2 w)
            _trapz(np, tw[neg], dx=dt),  # int_{t<0}(t w)
            _trapz(np, t2w[neg], dx=dt),  # int_{t<0}(t^2 w)
        )
        self._smoother_integrals_cache[cache_key] = integrals
        return integrals

    def _get_smoother_smoothing(self, smoother, accel=5000, scv=5.0):
        i_180, jp1, jp2, jn1, jn2 = self._calc_smoother_integrals(smoother)
        half_accel = accel * 0.5
        offset_180 = half_accel * i_180
        offset_90_x = scv * jp1 + half_accel * jp2
        offset_90_y = scv * jn1 - half_accel * jn2
        offset_90 = math.sqrt(offset_90_x**2 + offset_90_y**2)
        return max(offset_90, abs(offset_180))

    def fit_shaper(
        self,
        shaper_cfg,
        calibration_data,
        shaper_freqs,
        damping_ratio,
        scv,
        max_smoothing,
        test_damping_ratios,
        max_freq,
        estimate_shaper,
        get_shaper_smoothing,
    ):
        np = self.numpy

        damping_ratio = damping_ratio or shaper_defs.DEFAULT_DAMPING_RATIO
        test_damping_ratios = test_damping_ratios or TEST_DAMPING_RATIOS

        shaper = shaper_cfg.init_func(1.0, damping_ratio)

        test_freq_bins = np.arange(0.0, 10.0, 0.01)
        test_shaper_vals = np.zeros(shape=test_freq_bins.shape)
        # Exact damping ratio of the printer is unknown, pessimizing
        # remaining vibrations over possible damping values
        for dr in test_damping_ratios:
            vals = estimate_shaper(self.numpy, shaper, dr, test_freq_bins)
            test_shaper_vals = np.maximum(test_shaper_vals, vals)

        # NOTE: max_freq must NOT be expanded to cover test_freqs.max()
        # below. That was only ever needed for an explicit --shaper_freq
        # range, and every caller that supplies one already pre-expands its
        # own max_freq to comfortably cover it (see calibrate_shaper.py's
        # main()) before calling in. Once MAX_SHAPER_FREQ (the fallback
        # ceiling for the default, unbounded search) was raised above the
        # typical max_freq default of 200, folding test_freqs.max() into
        # that max() silently inflated max_freq to ~300 for every ordinary
        # fit with no explicit frequency range -- computing each shaper's
        # `vals` against a wider freq_bins slice than any caller (e.g. the
        # plotting code, which truncates independently to its own max_freq)
        # expects, causing a shape mismatch.
        max_freq = max_freq or MAX_FREQ

        if not shaper_freqs:
            shaper_freqs = (None, None, None)
        if isinstance(shaper_freqs, tuple):
            # The default sweep is capped at max_freq: the PSD is truncated
            # there, so a design frequency above it puts the shaper's notch
            # entirely outside the scored data -- it can never win, and
            # every grid point costs a find_max_accel bisection. The
            # MAX_SHAPER_FREQ headroom above MAX_FREQ only matters when the
            # caller actually measured that high (max_freq > 200).
            freq_end = shaper_freqs[1] or min(MAX_SHAPER_FREQ, max_freq)
            freq_start = min(
                shaper_freqs[0] or shaper_cfg.min_freq, freq_end - 1e-7
            )
            freq_step = shaper_freqs[2] or 0.2
            test_freqs = np.arange(freq_start, freq_end, freq_step)
        else:
            test_freqs = np.array(shaper_freqs)

        freq_bins = calibration_data.freq_bins
        psd = calibration_data.psd_sum[freq_bins <= max_freq]
        freq_bins = freq_bins[freq_bins <= max_freq]

        best_res = None
        results = []
        for test_freq in test_freqs[::-1]:
            shaper = shaper_cfg.init_func(test_freq, damping_ratio)
            shaper_smoothing = get_shaper_smoothing(shaper, scv=scv)
            if max_smoothing and shaper_smoothing > max_smoothing and best_res:
                return best_res, results
            shaper_vals = np.interp(
                freq_bins, test_freq_bins * test_freq, test_shaper_vals
            )
            shaper_vibrations = self._estimate_remaining_vibrations(
                freq_bins, shaper_vals, psd
            )
            max_accel = self.find_max_accel(shaper, scv, get_shaper_smoothing)
            # The score trying to minimize vibrations, but also accounting
            # the growth of smoothing. The formula itself does not have any
            # special meaning, it simply shows good results on real user data
            shaper_score = shaper_smoothing * (
                shaper_vibrations**1.5 + shaper_vibrations * 0.2 + 0.01
            )
            results.append(
                CalibrationResult(
                    name=shaper_cfg.name,
                    freq=test_freq,
                    vals=shaper_vals,
                    vibrs=shaper_vibrations,
                    smoothing=shaper_smoothing,
                    score=shaper_score,
                    max_accel=max_accel,
                )
            )
            if best_res is None or best_res.vibrs > results[-1].vibrs:
                # The current frequency is better for the shaper.
                best_res = results[-1]
        # Try to find an 'optimal' shapper configuration: the one that is not
        # much worse than the 'best' one, but gives much less smoothing
        selected = best_res
        for res in results[::-1]:
            if (
                res.vibrs < best_res.vibrs * 1.1 + 0.0005
                and res.score < selected.score
            ):
                selected = res
        # The full per-frequency results list is returned alongside the
        # selection so find_best_shaper can run mainline's strictly-better
        # upgrade walk (a same-type candidate at another frequency that beats
        # the current cross-shaper best on BOTH vibrations and smoothing
        # takes over the recommendation).
        return selected, results

    def _bisect(self, func, eps=1e-8):
        left = right = 1.0
        if not func(eps):
            return 0.0
        while not func(left):
            right = left
            left *= 0.5
        if right == left:
            while func(right):
                right *= 2.0
        while right - left > eps:
            middle = (left + right) * 0.5
            if func(middle):
                left = middle
            else:
                right = middle
        return left

    def find_max_accel(self, s, scv, get_smoothing):
        # Just some empirically chosen value which produces good projections
        # for max_accel without much smoothing
        TARGET_SMOOTHING = 0.12
        max_accel = self._bisect(
            lambda test_accel: (
                get_smoothing(s, test_accel, scv) <= TARGET_SMOOTHING
            ),
            1e-2,
        )
        return max_accel

    def _detect_resonance_peaks(
        self,
        freq_bins,
        psd,
        min_freq,
        max_freq,
        min_prominence=0.12,
        min_separation=8.0,
        max_peaks=2,
    ):
        # Look for well-separated local maxima in the PSD that could
        # each be shaped independently by a multimode shaper. Returns
        # up to `max_peaks` frequencies, sorted by descending PSD
        # magnitude (i.e. most significant first). Defaults to 2 since
        # most callers (e.g. _find_peak_cluster_bounds) only care whether
        # a narrow band splits into two; find_best_shaper passes
        # MULTIMODE_MAX_PEAKS explicitly for the top-level peak search.
        np = self.numpy
        mask = (freq_bins >= min_freq) & (freq_bins <= max_freq)
        freqs = freq_bins[mask]
        vals = psd[mask]
        if freqs.shape[0] < 3:
            return []
        is_peak = (vals[1:-1] > vals[:-2]) & (vals[1:-1] > vals[2:])
        idx = np.nonzero(is_peak)[0] + 1
        if idx.shape[0] == 0:
            return []
        peak_freqs = freqs[idx]
        peak_vals = vals[idx]
        order = np.argsort(peak_vals)[::-1]
        top_val = peak_vals[order[0]]
        selected = []
        for i in order:
            v = peak_vals[i]
            if v < min_prominence * top_val:
                break
            f = float(peak_freqs[i])
            if any(abs(f - sf) < min_separation for sf in selected):
                continue
            selected.append(f)
            if len(selected) >= max_peaks:
                break
        return selected

    def _find_peak_cluster_bounds(
        self, freq_bins, psd, center_freq, window=10.0
    ):
        # A peak reported by _detect_resonance_peaks (whose default
        # min_separation=8.0 merges anything closer together into just the
        # single tallest one) can actually be two distinguishable, genuinely
        # separate resonances only a few Hz apart -- confirmed against real
        # captures (e.g. two peaks 6 Hz apart on one axis, each independently
        # resolvable with a tighter separation, each with its own real PSD
        # dip between them). fit_multimode_shaper's local frequency search is
        # centered on the single reported peak and would otherwise never
        # learn the second one exists -- missing that the best compromise
        # frequency sits between the two, not at either individually (this
        # was verified directly: for a base/damping matched to real capture
        # data, the score-minimizing design frequency was the sub-peaks'
        # midpoint, beating either sub-peak alone by a clear margin).
        #
        # Re-scan a narrow band around the reported peak with a much
        # tighter separation to check. Returns (lo, hi) bracketing both
        # sub-peaks if a second one is found, or (center_freq, center_freq)
        # -- a no-op bound that leaves the caller's own default window
        # unchanged -- if the peak looks like an ordinary, single resonance.
        sub_peaks = self._detect_resonance_peaks(
            freq_bins,
            psd,
            max(0.0, center_freq - window),
            center_freq + window,
            min_separation=4.0,
            max_peaks=2,
        )
        if len(sub_peaks) < 2:
            return center_freq, center_freq
        return min(sub_peaks), max(sub_peaks)

    def _estimate_damping_ratio(
        self, freq_bins, psd, f0, max_span_lo=None, max_span_hi=None
    ):
        # Half-power (-3 dB) bandwidth method: for a lightly damped 2nd
        # order resonance, the two frequencies either side of the peak
        # where the PSD drops to half its peak value bracket a bandwidth
        # of approximately 2 * zeta * f0. Only meaningful for an isolated,
        # reasonably narrow peak, hence the search span limits (to avoid
        # running into a neighboring peak) and the sanity-clipped result.
        #
        # The two directions are bounded independently: a real resonance
        # can decay asymmetrically (steep on one side, a gradual shoulder
        # on the other -- e.g. when it sits on the shoulder of a broader
        # nearby structure), so a single symmetric span tight enough to
        # protect against a neighboring peak on one side can cut the
        # search short on the other side, which has nothing to protect
        # against. Callers that know of a nearby second peak should
        # tighten max_span_lo/max_span_hi accordingly (see
        # find_best_shaper) -- only on the side that peak is actually on.
        np = self.numpy
        if f0 <= 0.0 or freq_bins.shape[0] < 3:
            return None
        i0 = int(np.argmin(np.abs(freq_bins - f0)))
        p0 = psd[i0]
        if p0 <= 0.0:
            return None
        half = 0.5 * p0
        default_span = max(15.0, 0.3 * f0)
        span_lo = max_span_lo if max_span_lo is not None else default_span
        span_hi = max_span_hi if max_span_hi is not None else default_span
        n = freq_bins.shape[0]

        def find_crossing(step, span):
            i = i0
            while (
                0 <= i + step < n
                and psd[i + step] > half
                and abs(freq_bins[i + step] - f0) <= span
            ):
                i += step
            j = i + step
            if not (0 <= j < n) or abs(freq_bins[j] - f0) > span:
                return None
            f_i, p_i = freq_bins[i], psd[i]
            f_j, p_j = freq_bins[j], psd[j]
            if p_i == p_j:
                return float(f_j)
            t = (half - p_i) / (p_j - p_i)
            return float(f_i + t * (f_j - f_i))

        f_lo = find_crossing(-1, span_lo)
        f_hi = find_crossing(1, span_hi)
        if f_lo is None or f_hi is None or f_hi <= f_lo:
            return None
        zeta = (f_hi - f_lo) / (2.0 * f0)
        if zeta < 0.005 or zeta > 0.5:
            return None
        return zeta

    def fit_multimode_shaper(
        self,
        base_cfgs,
        calibration_data,
        peaks,
        damping_ratios,
        scv,
        max_smoothing,
        test_damping_ratios,
        max_freq,
    ):
        # A multimode shaper's structure is scale-invariant in its N-1
        # freq[i]/freq[0] ratios (not in freq[0] alone). For N=2 (a single
        # ratio) the search jointly sweeps that ratio and then refines
        # freq[0] locally, same as the original 2-peak search. Jointly
        # sweeping all N-1 ratios the same way would cost (grid points) **
        # (N-1) shaper evaluations, which blows up fast (e.g. 9**3 = 729 for
        # 4 peaks) -- for N>2 each ratio is instead refined one at a time
        # (coordinate descent, holding the others fixed at the detected
        # peaks' own frequency ratio and freq[0] anchored at peaks[0]) before
        # the same freq[0] local refinement. This is exact for N=2 and an
        # approximation for N>2 that keeps the search tractable.
        np = self.numpy
        n = len(peaks)

        damping_ratios = list(damping_ratios)
        for i, dr in enumerate(damping_ratios):
            if not dr:
                damping_ratios[i] = (
                    damping_ratios[i - 1]
                    if i > 0
                    else shaper_defs.DEFAULT_DAMPING_RATIO
                )
        test_damping_ratios = test_damping_ratios or TEST_DAMPING_RATIOS
        base_names = [cfg.name for cfg in base_cfgs]
        name = (
            base_names[0]
            if all(b == base_names[0] for b in base_names)
            else "/".join(base_names)
        )

        fb, psd_full = calibration_data.freq_bins, calibration_data.psd_sum

        def pessimized_vals(shaper, freqs):
            vals = np.zeros(shape=freqs.shape)
            for dr in test_damping_ratios:
                vals = np.maximum(vals, estimate_shaper(np, shaper, dr, freqs))
            return vals

        def cheap_pessimized_vals(shaper, freqs):
            # Closed-form response estimate: ~1000x fewer transcendental
            # ops than estimate_shaper's time-domain simulation. Slightly
            # different absolute values, but it ranks nearby candidates on
            # a coarse grid the same way, which is all the screening
            # passes below use it for -- anything they keep is re-scored
            # with the accurate estimator before it can influence the
            # final selection.
            vals = np.zeros(shape=freqs.shape)
            for dr in test_damping_ratios:
                vals = np.maximum(
                    vals, estimate_shaper_old(np, shaper, dr, freqs)
                )
            return vals

        # Score on a band extended past the top peak so its full notch
        # neighborhood is weighed, but remember the caller's own band: the
        # stored `vals` must match the freq_bins slice every other
        # candidate uses (fit_shaper never extends), or downstream
        # consumers that plot all candidates on one grid (e.g.
        # calibrate_shaper.py's plot_freq_response) hit a length mismatch.
        out_max_freq = max_freq or MAX_FREQ
        max_freq = max(out_max_freq, peaks[-1] * 1.2)
        freq_bins = fb[fb <= max_freq]
        psd = psd_full[fb <= max_freq]
        n_out = int(np.count_nonzero(fb <= out_max_freq))

        # Each secondary peak (i >= 1) gets its own ratio search range: if
        # it's actually an unresolved pair of close peaks, widen the sweep
        # to bracket both instead of just the one that got reported, adding
        # points so the extra range doesn't come at the cost of coarser
        # resolution right where it matters.
        ratio_grids = []
        current_ratios = []
        for i in range(1, n):
            base_ratio = peaks[i] / peaks[0]
            cluster_lo, cluster_hi = self._find_peak_cluster_bounds(
                fb, psd_full, peaks[i]
            )
            ratio_lo = min(base_ratio * 0.94, cluster_lo / peaks[0])
            ratio_hi = max(base_ratio * 1.06, cluster_hi / peaks[0])
            n_ratios = (
                5
                if (ratio_lo, ratio_hi)
                == (base_ratio * 0.94, base_ratio * 1.06)
                else 9
            )
            ratio_grids.append(np.linspace(ratio_lo, ratio_hi, n_ratios))
            current_ratios.append(base_ratio)

        if n > 2:
            # Coordinate descent: refine one ratio at a time, each
            # evaluated directly at freq[0] = peaks[0] (not swept -- that
            # happens in the freq[0] refinement pass below). Two-stage per
            # dimension: the cheap estimator ranks the whole grid, and only
            # the top candidates get the expensive accurate estimate --
            # the descent only has to pick a good starting ratio, so a
            # cheap ranking with an accurate head-to-head is enough.
            for dim, grid in enumerate(ratio_grids):
                screened = []
                for candidate in grid:
                    trial = list(current_ratios)
                    trial[dim] = candidate
                    trial_freqs = [peaks[0]] + [peaks[0] * r for r in trial]
                    shaper = shaper_defs.get_multimode_shaper(
                        base_names, trial_freqs, damping_ratios
                    )
                    if (
                        max_smoothing
                        and self._get_shaper_smoothing(shaper, scv=scv)
                        > max_smoothing
                    ):
                        continue
                    vals = cheap_pessimized_vals(shaper, freq_bins)
                    vibrs = self._estimate_remaining_vibrations(
                        freq_bins, vals, psd
                    )
                    screened.append((vibrs, candidate, shaper))
                if not screened:
                    continue
                screened.sort(key=lambda s: s[0])
                best_ratio, best_vibrs = current_ratios[dim], None
                for _, candidate, shaper in screened[:2]:
                    vals = pessimized_vals(shaper, freq_bins)
                    vibrs = self._estimate_remaining_vibrations(
                        freq_bins, vals, psd
                    )
                    if best_vibrs is None or vibrs < best_vibrs:
                        best_vibrs, best_ratio = vibrs, candidate
                current_ratios[dim] = best_ratio
            ratio_combos = [tuple(current_ratios)]
        else:
            candidate_ratios = list(ratio_grids[0]) if ratio_grids else []
            if len(candidate_ratios) > 3:
                # Screen the ratio sweep the same way: each surviving
                # ratio costs 3 accurate estimate_shaper calls on the
                # 500-point unit grid below, the dominant cost of the
                # whole fit, so rank at freq[0] = peaks[0] with the cheap
                # estimator and keep only the best 3 for the full
                # ratio x freq[0] search.
                screened = []
                for ratio in candidate_ratios:
                    shaper = shaper_defs.get_multimode_shaper(
                        base_names,
                        [peaks[0], peaks[0] * ratio],
                        damping_ratios,
                    )
                    vals = cheap_pessimized_vals(shaper, freq_bins)
                    vibrs = self._estimate_remaining_vibrations(
                        freq_bins, vals, psd
                    )
                    screened.append((vibrs, ratio))
                screened.sort(key=lambda s: s[0])
                candidate_ratios = [r for _, r in screened[:3]]
            ratio_combos = [(r,) for r in candidate_ratios] or [()]

        # Same idea for peak[0]'s own local refinement window.
        cluster1_lo, cluster1_hi = self._find_peak_cluster_bounds(
            fb, psd_full, peaks[0]
        )
        freq_start = max(
            base_cfgs[0].min_freq, min(peaks[0] - 4.0, cluster1_lo)
        )
        freq_end = max(peaks[0] + 4.0, cluster1_hi)
        test_freqs0 = np.arange(freq_start, freq_end, 0.5)
        if test_freqs0.shape[0] == 0:
            test_freqs0 = np.array([peaks[0]])

        # Coarser than the single-mode fit's 0.01 step: this grid is
        # rebuilt on every ratio combination (unlike single-mode, which
        # pays this cost only once per base), so keep it cheaper.
        test_freq_bins = np.arange(0.0, 10.0, 0.02)
        best_res = None
        results = []
        for ratio_combo in ratio_combos:
            unit_freqs = (1.0,) + ratio_combo
            unit_shaper = shaper_defs.get_multimode_shaper(
                base_names, list(unit_freqs), damping_ratios
            )
            test_shaper_vals = pessimized_vals(unit_shaper, test_freq_bins)
            for test_freq0 in test_freqs0:
                test_freqs = [test_freq0 * f for f in unit_freqs]
                shaper = shaper_defs.get_multimode_shaper(
                    base_names, test_freqs, damping_ratios
                )
                shaper_smoothing = self._get_shaper_smoothing(shaper, scv=scv)
                if max_smoothing and shaper_smoothing > max_smoothing:
                    continue
                shaper_vals = np.interp(
                    freq_bins, test_freq_bins * test_freq0, test_shaper_vals
                )
                shaper_vibrations = self._estimate_remaining_vibrations(
                    freq_bins, shaper_vals, psd
                )
                shaper_score = shaper_smoothing * (
                    shaper_vibrations**1.5 + shaper_vibrations * 0.2 + 0.01
                )
                result = CalibrationResult(
                    name="multimode_" + name,
                    freq=test_freq0,
                    vals=shaper_vals[:n_out],
                    vibrs=shaper_vibrations,
                    smoothing=shaper_smoothing,
                    score=shaper_score,
                    # max_accel is expensive (a bisection over the
                    # smoothing) so it is only computed for the single
                    # selected candidate below, not every grid point.
                    max_accel=0.0,
                    bases=tuple(base_names),
                    freqs=tuple(test_freqs),
                    damping_ratios=tuple(damping_ratios),
                )
                results.append(result)
                if best_res is None or best_res.vibrs > result.vibrs:
                    best_res = result
        if best_res is None:
            return None
        selected = best_res
        for res in results[::-1]:
            if (
                res.vibrs < best_res.vibrs * 1.1 + 0.0005
                and res.score < selected.score
            ):
                selected = res
        selected_shaper = shaper_defs.get_multimode_shaper(
            base_names, list(selected.freqs), damping_ratios
        )
        max_accel = self.find_max_accel(
            selected_shaper, scv, self._get_shaper_smoothing
        )
        return selected._replace(max_accel=max_accel)

    def find_best_shaper(
        self,
        calibration_data,
        shapers=None,
        damping_ratio=None,
        scv=None,
        shaper_freqs=None,
        max_smoothing=None,
        test_damping_ratios=None,
        max_freq=None,
        test_multimode=True,
        multimode_bias=1.0,
        logger=None,
    ):
        best_shaper = None
        all_shapers = []
        # Only auto-detect multimode shapers on a default run: an explicit
        # shaper list or fixed shaper_freqs means the user has already
        # decided what to test.
        use_multimode = test_multimode and shapers is None and not shaper_freqs
        shapers = shapers or AUTOTUNE_SHAPERS
        # Fit every candidate concurrently (they are fully independent),
        # then process the results strictly in the traditional smoothers-
        # then-shapers order so the logged output and the order-dependent
        # best-candidate selection below are identical to a serial run.
        fit_tasks = [
            (cfg, "smoother", estimate_smoother, self._get_smoother_smoothing)
            for cfg in shaper_defs.INPUT_SMOOTHERS
            if cfg.name in shapers
        ] + [
            (cfg, "shaper", estimate_shaper, self._get_shaper_smoothing)
            for cfg in shaper_defs.INPUT_SHAPERS
            if cfg.name in shapers
        ]
        fit_results = self.background_process_exec_parallel(
            self.fit_shaper,
            [
                (
                    cfg,
                    calibration_data,
                    shaper_freqs,
                    damping_ratio,
                    scv,
                    max_smoothing,
                    test_damping_ratios,
                    max_freq,
                    estimator,
                    get_smoothing,
                )
                for cfg, _, estimator, get_smoothing in fit_tasks
            ],
        )
        for (cfg, kind, _, _), (shaper, results) in zip(
            fit_tasks, fit_results
        ):
            if (
                best_shaper is None
                or shaper.score * 1.2 < best_shaper.score
                or (
                    shaper.score * 1.05 < best_shaper.score
                    and shaper.smoothing * 1.1 < best_shaper.smoothing
                )
            ):
                # Either the candidate significantly improves the score (by
                # 20%), or it improves the score and smoothing (by 5% and
                # 10% resp.)
                best_shaper = shaper
            # Mainline's upgrade walk: any of this candidate's other tested
            # frequencies that beats the current best on BOTH remaining
            # vibrations and smoothing is strictly better and takes over.
            for s in results[::-1]:
                if (
                    s.vibrs < best_shaper.vibrs
                    and s.smoothing < best_shaper.smoothing
                ):
                    best_shaper = shaper = s
            if logger is not None:
                logger(
                    "Fitted %s '%s' frequency = %.1f Hz "
                    "(vibration score = %.2f%%, smoothing ~= %.3f,"
                    " combined score = %.3e)"
                    % (
                        kind,
                        shaper.name,
                        shaper.freq,
                        shaper.vibrs * 100.0,
                        shaper.smoothing,
                        shaper.score,
                    )
                )
                logger(
                    "To avoid too much smoothing with '%s', suggested "
                    "max_accel <= %.0f mm/sec^2"
                    % (shaper.name, round(shaper.max_accel / 100.0) * 100.0)
                )
            all_shapers.append(shaper)
        if best_shaper is not None and best_shaper.name == "zv":
            # Mainline's ZV demotion: ZV is the narrowest shaper and wins
            # mostly on its minimal smoothing -- if any other candidate
            # leaves meaningfully (10%+) less vibration, prefer it.
            for tuned_shaper in all_shapers:
                if (
                    tuned_shaper.name != "zv"
                    and tuned_shaper.vibrs * 1.1 < best_shaper.vibrs
                ):
                    best_shaper = tuned_shaper
                    break
        if use_multimode:
            # Only worth the (much larger) search if the PSD actually shows
            # two or more distinct, well-separated resonances; on a typical
            # single-peak printer this is a no-op.
            fb = calibration_data.freq_bins
            psd = calibration_data.psd_sum
            peaks = self._detect_resonance_peaks(
                fb,
                psd,
                MIN_FREQ,
                max_freq or MAX_FREQ,
                max_peaks=MULTIMODE_MAX_PEAKS,
            )
            # Report the damping ratio at each detected peak: useful on its
            # own, and used below as the multimode fit's per-peak design
            # assumption instead of a single shared default. When another
            # peak is nearby, bound the half-power search on THAT side well
            # short of it so it doesn't pick up the neighbor's slope instead
            # of its own; the opposite side has no neighbor to protect
            # against and is left at _estimate_damping_ratio's own
            # generous default (a resonance can decay asymmetrically --
            # steep on the side facing a neighbor, a gradual shoulder on
            # the other -- and a symmetric span tight enough for the
            # former can otherwise cut the latter's search short).
            damping_estimates = []
            for i, f in enumerate(peaks):
                lower = [pf for j, pf in enumerate(peaks) if j != i and pf < f]
                upper = [pf for j, pf in enumerate(peaks) if j != i and pf > f]
                span_lo = max(5.0, (f - max(lower)) * 0.4) if lower else None
                span_hi = max(5.0, (min(upper) - f) * 0.4) if upper else None
                damping_estimates.append(
                    self._estimate_damping_ratio(
                        fb, psd, f, max_span_lo=span_lo, max_span_hi=span_hi
                    )
                )
            if logger is not None:
                for f, dr_est in zip(peaks, damping_estimates):
                    if dr_est is not None:
                        logger(
                            "Estimated damping ratio at %.1f Hz peak ~= %.3f"
                            % (f, dr_est)
                        )
            if len(peaks) >= 2:
                # Order by frequency (ascending), not detection magnitude:
                # fit_multimode_shaper's search is centered on the
                # lowest-frequency peak and derives the rest from it.
                ordered = sorted(zip(peaks, damping_estimates))
                ordered_peaks = [f for f, _ in ordered]
                # A missing (None) estimate at peak i falls back to the
                # previous peak's (possibly also defaulted) value. The seed
                # must itself never be None (damping_ratio is None on a
                # default run): the pulse-count pre-filter below convolves a
                # trial shaper from this list directly, before
                # fit_multimode_shaper's own None-handling would kick in.
                ordered_damping_ratios = []
                prev = damping_ratio or shaper_defs.DEFAULT_DAMPING_RATIO
                for _, dr_est in ordered:
                    prev = dr_est or prev
                    ordered_damping_ratios.append(prev)
                n_modes = len(ordered_peaks)
                base_cfgs_by_name = {
                    c.name: c for c in shaper_defs.INPUT_SHAPERS
                }
                # Every combination of bases scales as len(BASES)**n_modes;
                # only worth it for exactly 2 peaks (9 combos). Beyond that,
                # search a single base shared across all peaks instead of
                # letting it blow up (e.g. 3**4 = 81 combos for 4 peaks).
                if n_modes == 2:
                    base_combos = [
                        (b1, b2)
                        for b1 in MULTIMODE_AUTOTUNE_BASES
                        for b2 in MULTIMODE_AUTOTUNE_BASES
                    ]
                else:
                    base_combos = [
                        (b,) * n_modes for b in MULTIMODE_AUTOTUNE_BASES
                    ]
                combo_args = []
                for combo in base_combos:
                    base_cfgs = [base_cfgs_by_name.get(b) for b in combo]
                    if any(c is None for c in base_cfgs):
                        continue
                    # Skip combos whose convolution would exceed the
                    # firmware's pulse buffer (MAX_SHAPER_PULSES):
                    # MultiModeInputShaperParams would reject the resulting
                    # config anyway, and checking here -- the convolution
                    # itself is cheap, unlike the fit below -- avoids
                    # wasting the expensive per-candidate search on a
                    # shaper that could never be saved. This is what keeps
                    # higher-impulse bases (mzv, ei) from being tried at
                    # higher peak counts, where their impulse counts
                    # multiply out past the limit (e.g. 3 impulses ** 4
                    # peaks = 81 > 32).
                    trial_A, _ = shaper_defs.get_multimode_shaper(
                        list(combo), ordered_peaks, ordered_damping_ratios
                    )
                    if len(trial_A) > shaper_defs.MAX_SHAPER_PULSES:
                        continue
                    combo_args.append(
                        (
                            base_cfgs,
                            calibration_data,
                            ordered_peaks,
                            ordered_damping_ratios,
                            scv,
                            max_smoothing,
                            test_damping_ratios,
                            max_freq,
                        )
                    )
                multimode_results = [
                    result
                    for result in self.background_process_exec_parallel(
                        self.fit_multimode_shaper, combo_args
                    )
                    if result is not None
                ]
                if multimode_results:
                    # Only surface the single best multimode candidate: the
                    # others differ only by base shaper(s) and just clutter
                    # the graph and console.
                    multimode = min(multimode_results, key=lambda r: r.score)
                    if logger is not None:
                        freqs_str = " / ".join(
                            "%.1f" % (f,) for f in multimode.freqs
                        )
                        drs_str = " / ".join(
                            "%.3f" % (d,) for d in multimode.damping_ratios
                        )
                        logger(
                            "Fitted multimode shaper '%s' frequencies "
                            "= %s Hz (damping ratios %s, "
                            "vibration score = %.2f%%, smoothing ~= %.3f, "
                            "combined score = %.3e)"
                            % (
                                multimode.name[len("multimode_") :],
                                freqs_str,
                                drs_str,
                                multimode.vibrs * 100.0,
                                multimode.smoothing,
                                multimode.score,
                            )
                        )
                        logger(
                            "To avoid too much smoothing with multimode "
                            "'%s', suggested max_accel <= %.0f mm/sec^2"
                            % (
                                multimode.name[len("multimode_") :],
                                round(multimode.max_accel / 100.0) * 100.0,
                            )
                        )
                    all_shapers.append(multimode)
                    # Multimode requires manually maintaining extra
                    # frequency/damping ratio entries, so it is held to a
                    # configurable margin (multimode_bias) before it
                    # displaces the recommendation: 1.0 (the default) takes
                    # multimode on any genuine score improvement, values
                    # above 1.0 require it to win by that margin (e.g. 1.3
                    # only on a decisive win), and values below 1.0 prefer
                    # multimode even when it scores slightly worse.
                    if (
                        best_shaper is None
                        or multimode.score * multimode_bias < best_shaper.score
                    ):
                        best_shaper = multimode
        return best_shaper, all_shapers

    def _autosave_option(self, configfile, section, option):
        # The raw current value of an autosave (SAVE_CONFIG-managed) option,
        # or None if it isn't currently set. Used to only touch/clear an
        # option when a prior save actually left one behind, instead of
        # unconditionally writing reset values into an already-clean config.
        autosave = getattr(configfile, "autosave", None)
        if autosave is None or not autosave.fileconfig.has_option(
            section, option
        ):
            return None
        return autosave.fileconfig.get(section, option)

    # Canonical option layout for the [input_shaper] autosave section: all
    # X-axis options first, then Y, each axis in this stem order.
    _SAVE_OPTION_STEMS = [
        "shaper_type",
        "shaper_base",
        "shaper_freq",
        "damping_ratio",
    ]

    def _sort_autosave_options(self, configfile):
        # SAVE_CONFIG writes the autosave section in option insertion
        # order, so repeated SHAPER_CALIBRATE runs that add/remove options
        # (e.g. switching an axis between multimode and a single-frequency
        # type) gradually scramble the block. Rewrite the section in a
        # stable canonical order instead: x options first, then y, each in
        # _SAVE_OPTION_STEMS order, with anything unrecognized after them.
        section = "input_shaper"
        autosave = getattr(configfile, "autosave", None)
        if autosave is None or not autosave.fileconfig.has_section(section):
            return
        fileconfig = autosave.fileconfig
        options = fileconfig.options(section)

        def sort_key(option):
            for axis_rank, axis in enumerate(("x", "y")):
                if option.endswith("_" + axis):
                    stem = option[:-2]
                    try:
                        stem_rank = self._SAVE_OPTION_STEMS.index(stem)
                    except ValueError:
                        stem_rank = len(self._SAVE_OPTION_STEMS)
                    return (0, axis_rank, stem_rank, option)
            return (1, 0, 0, option)

        ordered = sorted(options, key=sort_key)
        if ordered == options:
            return
        values = [(opt, fileconfig.get(section, opt)) for opt in ordered]
        for option in options:
            fileconfig.remove_option(section, option)
        for option, value in values:
            fileconfig.set(section, option, value)

    def save_params(self, configfile, axis, shaper):
        if axis == "xy":
            self.save_params(configfile, "x", shaper)
            self.save_params(configfile, "y", shaper)
            return
        section = "input_shaper"
        if shaper.freqs is not None:
            configfile.set(section, "shaper_type_" + axis, "multimode")
            configfile.set(
                section, "shaper_base_" + axis, ", ".join(shaper.bases)
            )
            configfile.set(
                section,
                "shaper_freq_" + axis,
                ", ".join("%.1f" % (f,) for f in shaper.freqs),
            )
            configfile.set(
                section,
                "damping_ratio_" + axis,
                ", ".join("%.6f" % (d,) for d in shaper.damping_ratios),
            )
        else:
            configfile.set(section, "shaper_type_" + axis, shaper.name)
            configfile.set(
                section, "shaper_freq_" + axis, "%.1f" % (shaper.freq,)
            )
            # shaper_base_<axis> is only ever read by MultiModeInputShaperParams,
            # so it's stale once a non-multimode type is selected. Removed
            # outright (not blanked to "") since some readers -- e.g.
            # TypedInputSmootherParams.damping_ratio -- parse the option with
            # getfloat() and would fail to start on an empty value.
            configfile.remove_option(section, "shaper_base_" + axis)
            # damping_ratio_<axis> is shared with every other shaper type
            # (unlike the option above), so it's only removed if a prior
            # multimode save left a comma-separated list there, which every
            # other shaper type parses as a single float and would fail to
            # start on.
            raw = self._autosave_option(
                configfile, section, "damping_ratio_" + axis
            )
            if raw is not None and "," in raw:
                configfile.remove_option(section, "damping_ratio_" + axis)
        self._sort_autosave_options(configfile)

    def apply_params(self, input_shaper, axis, shaper):
        if axis == "xy":
            self.apply_params(input_shaper, "x", shaper)
            self.apply_params(input_shaper, "y", shaper)
            return
        gcode = self.printer.lookup_object("gcode")
        axis = axis.upper()
        if shaper.freqs is not None:
            params = {
                "SHAPER_TYPE_" + axis: "multimode",
                "SHAPER_BASE_" + axis: ", ".join(shaper.bases),
                "SHAPER_FREQ_" + axis: ", ".join(str(f) for f in shaper.freqs),
                "DAMPING_RATIO_" + axis: ", ".join(
                    str(d) for d in shaper.damping_ratios
                ),
            }
        else:
            params = {
                "SHAPER_TYPE_" + axis: shaper.name,
                "SHAPER_FREQ_" + axis: shaper.freq,
            }
        input_shaper.cmd_SET_INPUT_SHAPER(
            gcode.create_gcode_command(
                "SET_INPUT_SHAPER", "SET_INPUT_SHAPER", params
            )
        )

    def save_calibration_data(
        self,
        output,
        calibration_data,
        shapers=None,
        max_freq=None,
        accel_per_hz=None,
    ):
        try:
            max_freq = max_freq or MAX_FREQ
            with open(output, "w") as csvfile:
                csvfile.write("freq,psd_x,psd_y,psd_z,psd_xyz,accel_per_hz")
                if shapers:
                    for shaper in shapers:
                        if shaper.freqs is not None:
                            # Frequencies joined with '/' so the label stays
                            # a single comma-delimited CSV field.
                            freqs_label = "/".join(
                                "%.1f" % (f,) for f in shaper.freqs
                            )
                            csvfile.write(
                                ",%s(%s)" % (shaper.name, freqs_label)
                            )
                        else:
                            csvfile.write(
                                ",%s(%.1f)" % (shaper.name, shaper.freq)
                            )
                csvfile.write("\n")
                num_freqs = calibration_data.freq_bins.shape[0]
                for i in range(num_freqs):
                    if calibration_data.freq_bins[i] >= max_freq:
                        break
                    csvfile.write(
                        "%.1f,%.3e,%.3e,%.3e,%.3e,%.1f"
                        % (
                            calibration_data.freq_bins[i],
                            calibration_data.psd_x[i],
                            calibration_data.psd_y[i],
                            calibration_data.psd_z[i],
                            calibration_data.psd_sum[i],
                            accel_per_hz,
                        )
                    )
                    if shapers:
                        for shaper in shapers:
                            csvfile.write(",%.3f" % (shaper.vals[i],))
                    csvfile.write("\n")
        except IOError as e:
            raise self.error("Error writing to file '%s': %s", output, str(e))
