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
            (float(row["speed"]), float(row["disp_um"]), noise)
        )
    return header, data


def plot(filename, output):
    header, data = parse_csv(filename)
    if output:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, harmonic in enumerate(sorted({h for h, _ in data})):
        color = colors[idx % len(colors)]
        for direction, style in (("fwd", "-"), ("bwd", "--")):
            points = data.get((harmonic, direction))
            if not points:
                continue
            speeds, disps, noises = zip(*sorted(points))
            ax.plot(
                speeds,
                disps,
                style,
                color=color,
                label="H%d %s" % (harmonic, direction),
            )
            if direction == "fwd":
                ax.plot(speeds, noises, ":", color=color, alpha=0.5)
    ax.set_title(
        "Motor harmonics (%s)\ndotted: noise level" % (header,), fontsize=10
    )
    ax.set_xlabel("Motor speed (mm/s)")
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
