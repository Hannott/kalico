# Persists a compact resonance-vibration model -- a handful of peak
# frequency/damping/weight triples per axis, extracted from a completed
# SHAPER_CALIBRATE run -- the same way [bed_mesh] persists a probed profile
# via SAVE_CONFIG. resonance_ramp.py's notch-pair search uses this at print
# time to shape acceleration against the machine's actually measured
# resonances, instead of a frequency a human picked by hand.
#
# Copyright (C) 2026
# This file may be distributed under the terms of the GNU GPLv3 license.
from . import resonance_ramp, shaper_calibrate

AXES = ("x", "y")


class ResonanceModel:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.peaks = {}
        for axis in AXES:
            freqs = config.getfloatlist("peak_freqs_" + axis, ())
            dampings = config.getfloatlist("peak_dampings_" + axis, ())
            weights = config.getfloatlist("peak_weights_" + axis, ())
            if len(freqs) != len(dampings) or len(freqs) != len(weights):
                raise config.error(
                    f"resonance_model: peak_freqs_{axis}, peak_dampings_{axis} and"
                    f" peak_weights_{axis} must have the same number of"
                    " entries"
                )
            self.peaks[axis] = list(zip(freqs, dampings, weights))
        gcode = self.printer.lookup_object("gcode")
        gcode.register_command(
            "SAVE_RESONANCE_MODEL",
            self.cmd_SAVE_RESONANCE_MODEL,
            desc=self.cmd_SAVE_RESONANCE_MODEL_help,
        )

    def get_model(self, axis):
        """A resonance_ramp.PeakModel for the given axis ("x"/"y"), empty
        if nothing has been saved for it yet."""
        return resonance_ramp.PeakModel(self.peaks.get(axis, []))

    def get_status(self, eventtime):
        return {
            "peak_freqs_x": tuple(f for f, _d, _w in self.peaks.get("x", [])),
            "peak_freqs_y": tuple(f for f, _d, _w in self.peaks.get("y", [])),
        }

    cmd_SAVE_RESONANCE_MODEL_help = (
        "Extract the significant resonance peaks (frequency, damping,"
        " relative weight) from the last completed SHAPER_CALIBRATE run"
        " and persist them via SAVE_CONFIG, like a bed_mesh profile."
        " AXIS=x/y/xy (default xy) selects which axes to update."
    )

    def cmd_SAVE_RESONANCE_MODEL(self, gcmd):
        axis_arg = gcmd.get("AXIS", "xy").lower()
        if axis_arg not in ("x", "y", "xy"):
            raise gcmd.error(
                "SAVE_RESONANCE_MODEL: AXIS must be one of x, y, xy"
            )
        resonance_tester = self.printer.lookup_object("resonance_tester", None)
        if resonance_tester is None:
            raise gcmd.error(
                "SAVE_RESONANCE_MODEL requires a [resonance_tester] section"
            )
        last_data = getattr(resonance_tester, "last_calibration_data", None)
        if not last_data:
            raise gcmd.error(
                "SAVE_RESONANCE_MODEL: no calibration data available --"
                " run SHAPER_CALIBRATE (or TEST_RESONANCES) first"
            )
        helper = shaper_calibrate.ShaperCalibrate(self.printer)
        configfile = self.printer.lookup_object("configfile")
        reports = []
        for axis in AXES:
            if axis not in axis_arg:
                continue
            data = last_data.get(axis)
            if data is None:
                continue
            peaks = helper.extract_resonance_peaks(data)
            if not peaks:
                gcmd.respond_info(
                    "SAVE_RESONANCE_MODEL: no significant peaks found for"
                    f" axis {axis}, leaving it unchanged"
                )
                continue
            self.peaks[axis] = peaks
            configfile.set(
                "resonance_model",
                "peak_freqs_" + axis,
                ", ".join(f"{f:.3f}" for f, _d, _w in peaks),
            )
            configfile.set(
                "resonance_model",
                "peak_dampings_" + axis,
                ", ".join(f"{d:.6f}" for _f, d, _w in peaks),
            )
            configfile.set(
                "resonance_model",
                "peak_weights_" + axis,
                ", ".join(f"{w:.6f}" for _f, _d, w in peaks),
            )
            reports.append(
                "{}: {}".format(
                    axis,
                    ", ".join(f"{f:.2f}Hz" for f, _d, _w in peaks),
                )
            )
        gcmd.respond_info(
            "Resonance model saved for {}. The SAVE_CONFIG command will"
            " update the printer config file and restart the printer.".format(
                "; ".join(reports) if reports else "no axes"
            )
        )


def load_config(config):
    return ResonanceModel(config)
