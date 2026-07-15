# Kinematic input shaper to minimize motion vibrations in XY plane
#
# Copyright (C) 2019-2020  Kevin O'Connor <kevin@koconnor.net>
# Copyright (C) 2020-2023  Dmitry Butyugin <dmbutyugin@google.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import collections

from klippy import chelper

from . import extruder_smoother, shaper_defs


def parse_float_list(list_str):
    def parse_str(s):
        res = []
        for line in s.split("\n"):
            for coeff in line.split(","):
                res.append(float(coeff.strip()))
        return res

    try:
        return parse_str(list_str)
    except:
        return None


class TypedInputShaperParams:
    shapers = {s.name: s.init_func for s in shaper_defs.INPUT_SHAPERS}

    def __init__(self, axis, shaper_type, config):
        self.axis = axis
        self.shaper_type = shaper_type
        self.damping_ratio = shaper_defs.DEFAULT_DAMPING_RATIO
        self.shaper_freq = 0.0
        if config is not None:
            if shaper_type not in self.shapers:
                raise config.error(
                    "Unsupported shaper type: %s" % (shaper_type,)
                )
            self.damping_ratio = config.getfloat(
                "damping_ratio_" + axis,
                self.damping_ratio,
                minval=0.0,
                maxval=1.0,
            )
            self.shaper_freq = config.getfloat(
                "shaper_freq_" + axis, self.shaper_freq, minval=0.0
            )

    def get_type(self):
        return self.shaper_type

    def get_axis(self):
        return self.axis

    def update(self, shaper_type, gcmd):
        if shaper_type not in self.shapers:
            raise gcmd.error("Unsupported shaper type: %s" % (shaper_type,))
        axis = self.axis.upper()
        self.damping_ratio = gcmd.get_float(
            "DAMPING_RATIO_" + axis, self.damping_ratio, minval=0.0, maxval=1.0
        )
        self.shaper_freq = gcmd.get_float(
            "SHAPER_FREQ_" + axis, self.shaper_freq, minval=0.0
        )
        self.shaper_type = shaper_type

    def get_shaper(self):
        if not self.shaper_freq:
            A, T = shaper_defs.get_none_shaper()
        else:
            A, T = self.shapers[self.shaper_type](
                self.shaper_freq, self.damping_ratio
            )
        return len(A), A, T

    def get_status(self):
        return collections.OrderedDict(
            [
                ("shaper_type", self.shaper_type),
                ("shaper_freq", "%.3f" % (self.shaper_freq,)),
                ("damping_ratio", "%.6f" % (self.damping_ratio,)),
            ]
        )


class CustomInputShaperParams:
    SHAPER_TYPE = "custom"

    def __init__(self, axis, config):
        self.axis = axis
        self.n, self.A, self.T = 0, [], []
        if config is not None:
            shaper_a_str = config.get("shaper_a_" + axis)
            shaper_t_str = config.get("shaper_t_" + axis)
            self.n, self.A, self.T = self._parse_custom_shaper(
                shaper_a_str, shaper_t_str, config.error
            )

    def get_type(self):
        return self.SHAPER_TYPE

    def get_axis(self):
        return self.axis

    def update(self, shaper_type, gcmd):
        if shaper_type != self.SHAPER_TYPE:
            raise gcmd.error("Unsupported shaper type: %s" % (shaper_type,))
        axis = self.axis.upper()
        shaper_a_str = gcmd.get("SHAPER_A_" + axis, None)
        shaper_t_str = gcmd.get("SHAPER_T_" + axis, None)
        if (shaper_a_str is None) != (shaper_t_str is None):
            raise gcmd.error(
                "Both SHAPER_A_%s and SHAPER_T_%s parameters"
                " must be provided" % (axis, axis)
            )
        if shaper_a_str is not None:
            self.n, self.A, self.T = self._parse_custom_shaper(
                shaper_a_str, shaper_t_str, gcmd.error
            )

    def _parse_custom_shaper(self, custom_a_str, custom_t_str, parse_error):
        A = parse_float_list(custom_a_str)
        if A is None:
            raise parse_error("Invalid shaper A string: '%s'" % (custom_a_str,))
        if min([abs(a) for a in A]) < 0.001:
            raise parse_error("All shaper A coefficients must be non-zero")
        if sum(A) < 0.001:
            raise parse_error(
                "Shaper A parameter must sum up to a positive number"
            )
        T = parse_float_list(custom_t_str)
        if T is None:
            raise parse_error("Invalid shaper T string: '%s'" % (custom_t_str,))
        if T != sorted(T):
            raise parse_error("Shaper T parameter is not ordered: %s" % (T,))
        if len(A) != len(T):
            raise parse_error(
                "Shaper A and T parameters must have the same length:"
                " %d vs %d"
                % (
                    len(A),
                    len(T),
                )
            )
        dur = T[-1] - T[0]
        if len(T) > 1 and dur < 0.001:
            raise parse_error(
                "Shaper duration is too small (%.6f sec)" % (dur,)
            )
        if dur > 0.2:
            raise parse_error(
                "Shaper duration is too large (%.6f sec)" % (dur,)
            )
        return len(A), A, T

    def get_shaper(self):
        return self.n, self.A, self.T

    def get_status(self):
        return collections.OrderedDict(
            [
                ("shaper_type", self.SHAPER_TYPE),
                ("shaper_a", ",".join(["%.6f" % (a,) for a in self.A])),
                ("shaper_t", ",".join(["%.6f" % (t,) for t in self.T])),
            ]
        )


class TwoModeInputShaperParams:
    # "2mode" shaper: convolution of a base shaper tuned to each of N >= 2
    # resonance frequencies, placing a notch at every one of them (N is not
    # fixed at 2 despite the type name, kept for config compatibility).
    # Configured per axis with shaper_freq_<axis>, an optional
    # shaper_base_<axis> (default mzv), and optional per-peak
    # damping_ratio_<axis>, each of which accepts either a single value or
    # a comma-separated list (e.g. "shaper_freq_x: 45.2, 79.5, 132.6"); a
    # list shorter than the others is broadcast if it has exactly one
    # entry. The legacy two-value form (shaper_freq_<axis> /
    # shaper_freq2_<axis>, shaper_base_<axis> / shaper_base2_<axis>,
    # damping_ratio_<axis> / damping_ratio2_<axis>) is still accepted and
    # is equivalent to a 2-entry list; combining both styles on the same
    # option is a config error. The extruder gets a fitted smoother
    # counterpart (see extruder_smoother.get_multi_mode_extruder_smoother),
    # same as the named single-mode shapers.
    SHAPER_TYPE = "2mode"
    DEFAULT_BASE = "mzv"

    def __init__(self, axis, config):
        self.axis = axis
        self.bases = [self.DEFAULT_BASE]
        self.damping_ratios = [shaper_defs.DEFAULT_DAMPING_RATIO]
        self.freqs = [0.0]
        self.n, self.A, self.T = 0, [], []
        if config is not None:
            get_raw = lambda key: config.get(key, None)
            self.bases = self._parse_field(
                get_raw,
                "shaper_base_" + axis,
                "shaper_base2_" + axis,
                [self.DEFAULT_BASE],
                lambda s: s.strip().lower(),
                config.error,
            )
            for base in self.bases:
                self._check_base(base, config.error)
            self.damping_ratios = self._parse_field(
                get_raw,
                "damping_ratio_" + axis,
                "damping_ratio2_" + axis,
                [shaper_defs.DEFAULT_DAMPING_RATIO],
                float,
                config.error,
                minval=0.0,
                maxval=1.0,
            )
            self.freqs = self._parse_field(
                get_raw,
                "shaper_freq_" + axis,
                "shaper_freq2_" + axis,
                [0.0],
                float,
                config.error,
                minval=0.0,
            )
            self._build_shaper(config.error)

    def _parse_field(
        self, get_raw, key, key2, default_values, parser, error,
        minval=None, maxval=None,
    ):
        # A field may be a single legacy value, a comma-separated list (the
        # multi-mode form), or a single value plus a legacy "<key>2"
        # secondary value (equivalent to a 2-entry list); the latter two
        # styles may not be combined on the same field.
        raw = get_raw(key)
        if raw is None:
            values = list(default_values)
        else:
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            values = [parser(p) for p in parts] if parts else list(
                default_values
            )
        raw2 = get_raw(key2)
        if raw2 is not None:
            if len(values) > 1:
                raise error(
                    "%s: cannot combine a comma-separated list in '%s' "
                    "with the legacy '%s' option; use one style or the "
                    "other" % (self.axis, key, key2)
                )
            values = values + [parser(raw2.strip())]
        for v in values:
            if minval is not None and v < minval:
                raise error(
                    "%s: value %s in '%s' is below minimum %s"
                    % (self.axis, v, key, minval)
                )
            if maxval is not None and v > maxval:
                raise error(
                    "%s: value %s in '%s' is above maximum %s"
                    % (self.axis, v, key, maxval)
                )
        return values

    def _reconcile(self, values, n, name, error):
        # Broadcast a single shared value across all N peaks, or require an
        # exact per-peak match; anything else is an unambiguous config
        # mistake (e.g. exactly 2 bases given for 3 frequencies).
        if len(values) == n:
            return list(values)
        if len(values) == 1:
            return list(values) * n
        raise error(
            "%s: '%s' has %d value(s), expected 1 or %d (matching the "
            "number of frequencies)" % (self.axis, name, len(values), n)
        )

    def _check_base(self, base, error):
        if base not in shaper_defs.TWO_MODE_BASES:
            raise error(
                "Unsupported 2mode base shaper '%s' (choose one of: %s)"
                % (base, ", ".join(sorted(shaper_defs.TWO_MODE_BASES)))
            )

    def _build_shaper(self, error):
        freqs = self.freqs
        if len(freqs) < 2 or any(f <= 0.0 for f in freqs):
            self.n, self.A, self.T = 0, [], []
            return
        n = len(freqs)
        bases = self._reconcile(self.bases, n, "shaper_base", error)
        damping_ratios = self._reconcile(
            self.damping_ratios, n, "damping_ratio", error
        )
        for base in bases:
            self._check_base(base, error)
        A, T = shaper_defs.get_multi_mode_shaper(bases, freqs, damping_ratios)
        # The discrete shaper mechanism has a fixed-size pulse buffer
        # (MAX_SHAPER_PULSES in kin_shaper.h); a convolution silently
        # exceeding it would otherwise disable shaping without warning.
        if len(A) > shaper_defs.MAX_SHAPER_PULSES:
            raise error(
                "2mode shaper for axis %s: base(s) '%s' produce %d impulses,"
                " more than the %d the firmware supports; use shorter"
                " base shaper(s) (e.g. zv or mzv)"
                % (
                    self.axis,
                    "/".join(bases),
                    len(A),
                    shaper_defs.MAX_SHAPER_PULSES,
                )
            )
        self.bases, self.damping_ratios = bases, damping_ratios
        self.n, self.A, self.T = len(A), A, T

    def get_type(self):
        return self.SHAPER_TYPE

    def get_axis(self):
        return self.axis

    def update(self, shaper_type, gcmd):
        if shaper_type != self.SHAPER_TYPE:
            raise gcmd.error("Unsupported shaper type: %s" % (shaper_type,))
        axis = self.axis.upper()
        # Unlike config parsing, a gcode field that's entirely absent from
        # this command keeps the CURRENT full list (a partial update, e.g.
        # SET_INPUT_SHAPER SHAPER_FREQ_X=... alone leaves the bases as they
        # were); a field that IS given replaces its whole list, so changing
        # one entry of an N>2 list means restating all N values.
        get_raw = lambda key: gcmd.get(key, None)
        self.bases = self._parse_field(
            get_raw,
            "SHAPER_BASE_" + axis,
            "SHAPER_BASE2_" + axis,
            self.bases,
            lambda s: s.strip().lower(),
            gcmd.error,
        )
        for base in self.bases:
            self._check_base(base, gcmd.error)
        self.damping_ratios = self._parse_field(
            get_raw,
            "DAMPING_RATIO_" + axis,
            "DAMPING_RATIO2_" + axis,
            self.damping_ratios,
            float,
            gcmd.error,
            minval=0.0,
            maxval=1.0,
        )
        self.freqs = self._parse_field(
            get_raw,
            "SHAPER_FREQ_" + axis,
            "SHAPER_FREQ2_" + axis,
            self.freqs,
            float,
            gcmd.error,
            minval=0.0,
        )
        self._build_shaper(gcmd.error)

    def get_shaper(self):
        return self.n, self.A, self.T

    def get_status(self):
        return collections.OrderedDict(
            [
                ("shaper_type", self.SHAPER_TYPE),
                ("shaper_base", ",".join(self.bases)),
                (
                    "shaper_freq",
                    ",".join("%.3f" % (f,) for f in self.freqs),
                ),
                (
                    "damping_ratio",
                    ",".join("%.6f" % (d,) for d in self.damping_ratios),
                ),
            ]
        )


class AxisInputShaper:
    def __init__(self, params):
        self.params = params
        self.n, self.A, self.T = params.get_shaper()
        self.t_offs = shaper_defs.get_shaper_offset(self.A, self.T)
        self.saved = None

    def get_name(self):
        return "shaper_" + self.get_axis()

    def get_type(self):
        return self.params.get_type()

    def get_axis(self):
        return self.params.get_axis()

    def is_extruder_smoothing(self, exact_mode):
        # Custom shapers have no fitted extruder smoother counterpart,
        # they are applied to the extruder exactly (as a convolution)
        return (
            not exact_mode
            and self.A
            and self.get_type() != CustomInputShaperParams.SHAPER_TYPE
        )

    def is_enabled(self):
        return self.n > 0

    def update(self, shaper_type, gcmd):
        self.params.update(shaper_type, gcmd)
        self.n, self.A, self.T = self.params.get_shaper()
        self.t_offs = shaper_defs.get_shaper_offset(self.A, self.T)

    def update_stepper_kinematics(self, sk):
        ffi_main, ffi_lib = chelper.get_ffi()
        axis = self.get_axis().encode()
        success = (
            ffi_lib.input_shaper_set_shaper_params(
                sk, axis, self.n, self.A, self.T
            )
            == 0
        )
        if not success:
            self.disable_shaping()
            ffi_lib.input_shaper_set_shaper_params(
                sk, axis, self.n, self.A, self.T
            )
        return success

    def update_extruder_kinematics(self, sk, exact_mode):
        ffi_main, ffi_lib = chelper.get_ffi()
        axis = self.get_axis().encode()
        if not self.is_extruder_smoothing(exact_mode):
            # Make sure to disable any active input smoothing
            coeffs, smooth_time = [], 0.0
            success = (
                ffi_lib.extruder_set_smoothing_params(
                    sk, axis, len(coeffs), coeffs, smooth_time, 0.0
                )
                == 0
            )
            success = (
                ffi_lib.extruder_set_shaper_params(
                    sk, axis, self.n, self.A, self.T
                )
                == 0
            )
        else:
            shaper_type = self.get_type()
            status = self.params.get_status()
            if shaper_type == TwoModeInputShaperParams.SHAPER_TYPE:
                bases = status["shaper_base"].split(",")
                freqs = [float(f) for f in status["shaper_freq"].split(",")]
                damping_ratios = [
                    float(d) for d in status["damping_ratio"].split(",")
                ]
                smoother_fn = (
                    extruder_smoother.get_multi_mode_extruder_smoother
                )
                C_e, t_sm = smoother_fn(
                    bases,
                    freqs,
                    damping_ratios,
                    self.T[-1] - self.T[0],
                    normalize_coeffs=False,
                )
            else:
                damping_ratio = float(
                    status.get(
                        "damping_ratio", shaper_defs.DEFAULT_DAMPING_RATIO
                    )
                )
                C_e, t_sm = extruder_smoother.get_extruder_smoother(
                    shaper_type,
                    self.T[-1] - self.T[0],
                    damping_ratio,
                    normalize_coeffs=False,
                )
            smoother_offset = self.t_offs - 0.5 * t_sm
            success = (
                ffi_lib.extruder_set_smoothing_params(
                    sk, axis, len(C_e), C_e, t_sm, smoother_offset
                )
                == 0
            )
        if not success:
            self.disable_shaping()
            ffi_lib.extruder_set_shaper_params(sk, axis, self.n, self.A, self.T)
        return success

    def disable_shaping(self):
        was_enabled = False
        if self.saved is None and self.n:
            self.saved = (self.n, self.A, self.T)
            was_enabled = True
        A, T = shaper_defs.get_none_shaper()
        self.n, self.A, self.T = len(A), A, T
        return was_enabled

    def enable_shaping(self):
        if self.saved is None:
            # Input shaper was not disabled
            return False
        self.n, self.A, self.T = self.saved
        self.saved = None
        return True

    def report(self, gcmd):
        info = " ".join(
            [
                "%s_%s:%s" % (key, self.get_axis(), value)
                for (key, value) in self.params.get_status().items()
            ]
        )
        gcmd.respond_info(info)


class TypedInputSmootherParams:
    smoothers = {s.name: s.init_func for s in shaper_defs.INPUT_SMOOTHERS}

    def __init__(self, axis, smoother_type, config):
        self.axis = axis
        self.smoother_type = smoother_type
        self.smoother_freq = 0.0
        self.damping_ratio = shaper_defs.DEFAULT_DAMPING_RATIO
        if config is not None:
            if smoother_type not in self.smoothers:
                raise config.error(
                    "Unsupported shaper type: %s" % (smoother_type,)
                )
            # Accept shaper_freq_* as an alias: SHAPER_CALIBRATE and other
            # tools store smoother recommendations under that name
            shaper_freq = config.getfloat(
                "shaper_freq_" + axis, self.smoother_freq, minval=0.0
            )
            self.smoother_freq = config.getfloat(
                "smoother_freq_" + axis, shaper_freq, minval=0.0
            )
            self.damping_ratio = config.getfloat(
                "damping_ratio_" + axis,
                self.damping_ratio,
                minval=0.0,
                maxval=1.0,
            )

    def get_type(self):
        return self.smoother_type

    def get_axis(self):
        return self.axis

    def update(self, smoother_type, gcmd):
        if smoother_type not in self.smoothers:
            raise gcmd.error("Unsupported shaper type: %s" % (smoother_type,))
        axis = self.axis.upper()
        shaper_freq = gcmd.get_float(
            "SHAPER_FREQ_" + axis, self.smoother_freq, minval=0.0
        )
        self.smoother_freq = gcmd.get_float(
            "SMOOTHER_FREQ_" + axis, shaper_freq, minval=0.0
        )
        self.damping_ratio = gcmd.get_float(
            "DAMPING_RATIO_" + axis, self.damping_ratio, minval=0.0, maxval=1.0
        )
        self.smoother_type = smoother_type

    def get_smoother(self):
        if not self.smoother_freq:
            C, tsm = shaper_defs.get_none_smoother()
        else:
            C, tsm = self.smoothers[self.smoother_type](
                self.smoother_freq, normalize_coeffs=False
            )
        return len(C), C, tsm

    def get_status(self):
        return collections.OrderedDict(
            [
                ("shaper_type", self.smoother_type),
                ("smoother_freq", "%.3f" % (self.smoother_freq,)),
                ("damping_ratio", "%.6f" % (self.damping_ratio,)),
            ]
        )


class CustomInputSmootherParams:
    SHAPER_TYPE = "smoother"

    def __init__(self, axis, config):
        self.axis = axis
        self.coeffs, self.smooth_time = shaper_defs.get_none_smoother()
        if config is not None:
            self.smooth_time = config.getfloat(
                "smooth_time_" + axis, self.smooth_time, minval=0.0
            )
            self.coeffs = list(
                reversed(config.getfloatlist("coeffs_" + axis, self.coeffs))
            )

    def get_type(self):
        return self.SHAPER_TYPE

    def get_axis(self):
        return self.axis

    def update(self, shaper_type, gcmd):
        if shaper_type != self.SHAPER_TYPE:
            raise gcmd.error("Unsupported shaper type: %s" % (shaper_type,))
        axis = self.axis.upper()
        self.smooth_time = gcmd.get_float(
            "SMOOTH_TIME_" + axis, self.smooth_time
        )
        coeffs_str = gcmd.get("COEFFS_" + axis, None)
        if coeffs_str is not None:
            try:
                coeffs = parse_float_list(coeffs_str)
                coeffs.reverse()
            except:
                raise gcmd.error("Invalid format for COEFFS parameter")
            self.coeffs = coeffs

    def get_smoother(self):
        return len(self.coeffs), self.coeffs, self.smooth_time

    def get_status(self):
        return collections.OrderedDict(
            [
                ("shaper_type", self.SHAPER_TYPE),
                (
                    "shaper_coeffs",
                    ",".join(["%.9e" % (a,) for a in reversed(self.coeffs)]),
                ),
                ("shaper_smooth_time", self.smooth_time),
            ]
        )


class AxisInputSmoother:
    def __init__(self, params):
        self.params = params
        self.n, self.coeffs, self.smooth_time = params.get_smoother()
        self.t_offs = shaper_defs.get_smoother_offset(
            self.coeffs, self.smooth_time, normalized=False
        )
        self.saved_smooth_time = 0.0

    def get_name(self):
        return "smoother_" + self.get_axis()

    def get_type(self):
        return self.params.get_type()

    def get_axis(self):
        return self.params.get_axis()

    def is_extruder_smoothing(self, exact_mode):
        return True

    def is_enabled(self):
        return self.smooth_time > 0.0

    def update(self, shaper_type, gcmd):
        self.params.update(shaper_type, gcmd)
        self.n, self.coeffs, self.smooth_time = self.params.get_smoother()
        self.t_offs = shaper_defs.get_smoother_offset(
            self.coeffs, self.smooth_time, normalized=False
        )

    def update_stepper_kinematics(self, sk):
        ffi_main, ffi_lib = chelper.get_ffi()
        axis = self.get_axis().encode()
        success = (
            ffi_lib.input_shaper_set_smoother_params(
                sk, axis, self.n, self.coeffs, self.smooth_time
            )
            == 0
        )
        if not success:
            self.disable_shaping()
            ffi_lib.input_shaper_set_smoother_params(
                sk, axis, self.n, self.coeffs, self.smooth_time
            )
        return success

    def update_extruder_kinematics(self, sk, exact_mode):
        ffi_main, ffi_lib = chelper.get_ffi()
        axis = self.get_axis().encode()
        # Make sure to disable any active input shaping
        A, T = shaper_defs.get_none_shaper()
        ffi_lib.extruder_set_shaper_params(sk, axis, len(A), A, T)
        smoother_type = self.get_type()
        if exact_mode or smoother_type == CustomInputSmootherParams.SHAPER_TYPE:
            # Custom smoothers have no fitted extruder counterpart, apply
            # the smoother itself exactly
            success = (
                ffi_lib.extruder_set_smoothing_params(
                    sk, axis, self.n, self.coeffs, self.smooth_time, self.t_offs
                )
                == 0
            )
        else:
            status = self.params.get_status()
            damping_ratio = float(
                status.get("damping_ratio", shaper_defs.DEFAULT_DAMPING_RATIO)
            )
            C_e, t_sm = extruder_smoother.get_extruder_smoother(
                smoother_type,
                self.smooth_time,
                damping_ratio,
                normalize_coeffs=False,
            )
            success = (
                ffi_lib.extruder_set_smoothing_params(
                    sk, axis, len(C_e), C_e, t_sm, self.t_offs
                )
                == 0
            )
        if not success:
            self.disable_shaping()
            ffi_lib.extruder_set_smoothing_params(
                sk, axis, self.n, self.coeffs, self.smooth_time, 0.0
            )
        return success

    def disable_shaping(self):
        was_enabled = False
        if self.smooth_time:
            self.saved_smooth_time = self.smooth_time
            was_enabled = True
        self.smooth_time = 0.0
        return was_enabled

    def enable_shaping(self):
        if not self.saved_smooth_time:
            # Input smoother was not disabled
            return False
        self.smooth_time = self.saved_smooth_time
        self.saved_smooth_time = 0.0
        return True

    def report(self, gcmd):
        info = " ".join(
            [
                "%s_%s:%s" % (key, self.get_axis(), value)
                for (key, value) in self.params.get_status().items()
            ]
        )
        gcmd.respond_info(info)


class ShaperFactory:
    def __init__(self):
        pass

    def _create_shaper(self, axis, type_name, config=None):
        if type_name == CustomInputSmootherParams.SHAPER_TYPE:
            return AxisInputSmoother(CustomInputSmootherParams(axis, config))
        if type_name == CustomInputShaperParams.SHAPER_TYPE:
            return AxisInputShaper(CustomInputShaperParams(axis, config))
        if type_name == TwoModeInputShaperParams.SHAPER_TYPE:
            return AxisInputShaper(TwoModeInputShaperParams(axis, config))
        if type_name in TypedInputShaperParams.shapers:
            return AxisInputShaper(
                TypedInputShaperParams(axis, type_name, config)
            )
        if type_name in TypedInputSmootherParams.smoothers:
            return AxisInputSmoother(
                TypedInputSmootherParams(axis, type_name, config)
            )
        return None

    def create_shaper(self, axis, config):
        shaper_type = config.get("shaper_type", "mzv")
        shaper_type = config.get("shaper_type_" + axis, shaper_type).lower()
        shaper = self._create_shaper(axis, shaper_type, config)
        if shaper is None:
            raise config.error("Unsupported shaper type '%s'" % (shaper_type,))
        return shaper

    def update_shaper(self, shaper, gcmd):
        shaper_type = gcmd.get("SHAPER_TYPE", None)
        if shaper_type is None:
            shaper_type = gcmd.get(
                "SHAPER_TYPE_" + shaper.get_axis().upper(), shaper.get_type()
            )
        shaper_type = shaper_type.lower()
        try:
            shaper.update(shaper_type, gcmd)
            return shaper
        except gcmd.error:
            pass
        shaper = self._create_shaper(shaper.get_axis(), shaper_type)
        if shaper is None:
            raise gcmd.error("Unsupported shaper type '%s'" % (shaper_type,))
        shaper.update(shaper_type, gcmd)
        return shaper


class InputShaper:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.printer.register_event_handler("klippy:connect", self.connect)
        self.toolhead = None
        self.extruders = []
        self.exact_mode = 0
        self.config_extruder_names = config.getlist("enabled_extruders", [])
        self.shaper_factory = ShaperFactory()
        self.shapers = [
            self.shaper_factory.create_shaper("x", config),
            self.shaper_factory.create_shaper("y", config),
        ]
        self.input_shaper_stepper_kinematics = []
        self.orig_stepper_kinematics = []
        # Register gcode commands
        gcode = self.printer.lookup_object("gcode")
        gcode.register_command(
            "SET_INPUT_SHAPER",
            self.cmd_SET_INPUT_SHAPER,
            desc=self.cmd_SET_INPUT_SHAPER_help,
        )
        gcode.register_command(
            "ENABLE_INPUT_SHAPER",
            self.cmd_ENABLE_INPUT_SHAPER,
            desc=self.cmd_ENABLE_INPUT_SHAPER_help,
        )
        gcode.register_command(
            "DISABLE_INPUT_SHAPER",
            self.cmd_DISABLE_INPUT_SHAPER,
            desc=self.cmd_DISABLE_INPUT_SHAPER_help,
        )

    def get_shapers(self):
        return self.shapers

    def connect(self):
        self.toolhead = self.printer.lookup_object("toolhead")
        for en in self.config_extruder_names:
            extruder = self.printer.lookup_object(en)
            if not hasattr(extruder, "get_extruder_steppers"):
                raise self.printer.config_error(
                    "Invalid extruder '%s' in [input_shaper]" % (en,)
                )
            self.extruders.append(extruder)
        # Configure initial values
        self._update_input_shaping(error=self.printer.config_error)

    def _get_input_shaper_stepper_kinematics(self, stepper):
        # Lookup stepper kinematics
        sk = stepper.get_stepper_kinematics()
        if sk in self.orig_stepper_kinematics:
            # Already processed this stepper kinematics unsuccessfully
            return None
        if sk in self.input_shaper_stepper_kinematics:
            return sk
        self.orig_stepper_kinematics.append(sk)
        ffi_main, ffi_lib = chelper.get_ffi()
        is_sk = ffi_main.gc(ffi_lib.input_shaper_alloc(), ffi_lib.free)
        stepper.set_stepper_kinematics(is_sk)
        res = ffi_lib.input_shaper_set_sk(is_sk, sk)
        if res < 0:
            stepper.set_stepper_kinematics(sk)
            return None
        self.input_shaper_stepper_kinematics.append(is_sk)
        return is_sk

    def _update_input_shaping(self, error=None):
        self.toolhead.flush_step_generation()
        ffi_main, ffi_lib = chelper.get_ffi()
        kin = self.toolhead.get_kinematics()
        failed_shapers = []
        for s in kin.get_steppers():
            if s.get_trapq() is None:
                continue
            is_sk = self._get_input_shaper_stepper_kinematics(s)
            if is_sk is None:
                continue
            old_delay = ffi_lib.input_shaper_get_step_gen_window(is_sk)
            for shaper in self.shapers:
                if shaper in failed_shapers:
                    continue
                if not shaper.update_stepper_kinematics(is_sk):
                    failed_shapers.append(shaper)
            new_delay = ffi_lib.input_shaper_get_step_gen_window(is_sk)
            if old_delay != new_delay:
                self.toolhead.note_step_generation_scan_time(
                    new_delay, old_delay
                )
        for e in self.extruders:
            for es in e.get_extruder_steppers():
                failed_shapers.extend(
                    es.update_input_shaping(self.shapers, self.exact_mode)
                )
        if failed_shapers:
            error = error or self.printer.command_error
            raise error(
                "Failed to configure shaper(s) %s with given parameters"
                % (", ".join([s.get_name() for s in failed_shapers]))
            )

    def disable_shaping(self):
        for shaper in self.shapers:
            shaper.disable_shaping()
        self._update_input_shaping()

    def enable_shaping(self):
        for shaper in self.shapers:
            shaper.enable_shaping()
        self._update_input_shaping()

    cmd_SET_INPUT_SHAPER_help = "Set cartesian parameters for input shaper"

    def cmd_SET_INPUT_SHAPER(self, gcmd):
        if gcmd.get_command_parameters():
            self.shapers = [
                self.shaper_factory.update_shaper(shaper, gcmd)
                for shaper in self.shapers
            ]
            self._update_input_shaping()
        for shaper in self.shapers:
            shaper.report(gcmd)

    cmd_ENABLE_INPUT_SHAPER_help = "Enable input shaper for given objects"

    def cmd_ENABLE_INPUT_SHAPER(self, gcmd):
        self.toolhead.flush_step_generation()
        axes = gcmd.get("AXIS", "")
        msg = ""
        for axis_str in axes.split(","):
            axis = axis_str.strip().lower()
            if not axis:
                continue
            shapers = [s for s in self.shapers if s.get_axis() == axis]
            if not shapers:
                raise gcmd.error("Invalid AXIS='%s'" % (axis_str,))
            for s in shapers:
                if s.enable_shaping():
                    msg += "Enabled input shaper for AXIS='%s'\n" % (axis_str,)
                else:
                    msg += (
                        "Cannot enable input shaper for AXIS='%s': "
                        "was not disabled\n" % (axis_str,)
                    )
        extruders = gcmd.get("EXTRUDER", "")
        self.exact_mode = gcmd.get_int("EXACT", self.exact_mode)
        for en in extruders.split(","):
            extruder_name = en.strip()
            if not extruder_name:
                continue
            extruder = self.printer.lookup_object(extruder_name)
            if not hasattr(extruder, "get_extruder_steppers"):
                raise gcmd.error("Invalid EXTRUDER='%s'" % (en,))
            if extruder not in self.extruders:
                self.extruders.append(extruder)
                msg += "Enabled input shaper for '%s'\n" % (en,)
            else:
                msg += "Input shaper already enabled for '%s'\n" % (en,)
        self._update_input_shaping()
        gcmd.respond_info(msg)

    cmd_DISABLE_INPUT_SHAPER_help = "Disable input shaper for given objects"

    def cmd_DISABLE_INPUT_SHAPER(self, gcmd):
        self.toolhead.flush_step_generation()
        axes = gcmd.get("AXIS", "")
        msg = ""
        for axis_str in axes.split(","):
            axis = axis_str.strip().lower()
            if not axis:
                continue
            shapers = [s for s in self.shapers if s.get_axis() == axis]
            if not shapers:
                raise gcmd.error("Invalid AXIS='%s'" % (axis_str,))
            for s in shapers:
                if s.disable_shaping():
                    msg += "Disabled input shaper for AXIS='%s'\n" % (axis_str,)
                else:
                    msg += (
                        "Cannot disable input shaper for AXIS='%s': not "
                        "enabled or was already disabled\n" % (axis_str,)
                    )
        extruders = gcmd.get("EXTRUDER", "")
        for en in extruders.split(","):
            extruder_name = en.strip()
            if not extruder_name:
                continue
            extruder = self.printer.lookup_object(extruder_name)
            if extruder in self.extruders:
                to_re_enable = [s for s in self.shapers if s.disable_shaping()]
                for es in extruder.get_extruder_steppers():
                    es.update_input_shaping(self.shapers, self.exact_mode)
                for shaper in to_re_enable:
                    shaper.enable_shaping()
                self.extruders.remove(extruder)
                msg += "Disabled input shaper for '%s'\n" % (en,)
            else:
                msg += "Input shaper not enabled for '%s'\n" % (en,)
        self._update_input_shaping()
        gcmd.respond_info(msg)


def load_config(config):
    return InputShaper(config)
