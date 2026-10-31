#!/usr/bin/env python3
# Graph MEASURE_MOTOR_HARMONICS results
#
# Copyright (C) 2026  Åsmund Collin <aakjaergaard@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import csv
import optparse
import os

import matplotlib


def parse_csv(filename):
    with open(filename) as f:
        header = f.readline().lstrip("# ").strip()
        rows = list(csv.DictReader(f))
    data = {}
    for row in rows:
        key = (int(row["harmonic"]), row["direction"])
        noise = float(row["noise_um"]) if row["noise_um"] else float("nan")
        data.setdefault(key, []).append(
            (
                float(row["speed"]),
                float(row["freq"]),
                float(row["disp_um"]),
                noise,
            )
        )
    return header, data


def plot(filename, output):
    header, data = parse_csv(filename)
    if output:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The top panel shows where each harmonic is worst while printing; the
    # bottom one lines the harmonics up on the printer's resonances.
    fig, (ax_speed, ax_freq) = plt.subplots(2, 1, figsize=(10, 10))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, harmonic in enumerate(sorted({h for h, _ in data})):
        color = colors[idx % len(colors)]
        for direction, style in (("fwd", "-"), ("bwd", "--")):
            points = data.get((harmonic, direction))
            if not points:
                continue
            speeds, freqs, disps, noises = zip(*sorted(points))
            label = "H%d %s" % (harmonic, direction)
            ax_speed.plot(speeds, disps, style, color=color, label=label)
            ax_freq.plot(freqs, disps, style, color=color, label=label)
            if direction == "fwd":
                ax_speed.plot(speeds, noises, ":", color=color, alpha=0.5)
    ax_speed.set_title(
        "Motor harmonics (%s)\ndotted: noise level" % (header,), fontsize=10
    )
    ax_speed.set_xlabel("Motor speed (mm/s)")
    ax_freq.set_title("Same data by vibration frequency", fontsize=10)
    ax_freq.set_xlabel("Frequency (Hz)")
    for ax in (ax_speed, ax_freq):
        ax.set_ylabel("Toolhead displacement amplitude (um)")
        ax.set_yscale("log")
        ax.grid(which="both", alpha=0.3)
        ax.legend(loc="best", fontsize="small")
    if output:
        fig.savefig(output)
    else:
        plt.show()


def main():
    usage = "%prog [options] <motor_harmonics csv>"
    opts = optparse.OptionParser(usage)
    opts.add_option(
        "-o",
        "--output",
        type="string",
        dest="output",
        default=None,
        help="filename of output graph",
    )
    options, args = opts.parse_args()
    if len(args) != 1:
        opts.error("Incorrect number of arguments")
    if not os.path.exists(args[0]):
        opts.error("File not found: %s" % (args[0],))
    plot(args[0], options.output)


if __name__ == "__main__":
    main()
