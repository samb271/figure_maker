#!/usr/bin/env python3
"""Build side-by-side bar charts from a CSV file.

CSV format
----------
The file is split into one or more *env blocks* separated by blank lines.
Each block becomes its own bar-chart panel, left to right, and every panel
shares the same y scale.

A block looks like::

    # Env name                (optional, a line starting with '#')
    <bar label>, <value>      (one line per bar)
    <bar label>, <value±err>
    ...

Rules
-----
- Each entry line is one bar: "label, value" or "label, value±err"
  (the error draws a cap on top of the bar).
- Prefix a label with '~' to hatch that bar (e.g. "~DINO-WM (O.S.)"); the
  '~' is dropped from the legend.
- End a label with "= <other label>" to give the bar the same colour as
  that other bar (e.g. "~DINO-WM (O.S.) = DINO-WM").
- A label reused across panels keeps one colour and one legend entry.

Titles are passed on the command line: --title is the overall figure title,
--ytitle labels the shared y axis, and per-panel titles come from the '#'
lines in the CSV.
"""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_ERR_SEPARATORS = ("±", "+/-", "+-")

_OUTPUT_DIR = "outputs"

_FONT_SIZES = {
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 17,
}


def _parse_value(raw):
    """Return (value, err) for a cell, err == 0.0 when none is given."""
    s = raw.strip()
    for sep in _ERR_SEPARATORS:
        if sep in s:
            val, err = s.split(sep, 1)
            return float(val), abs(float(err))
    return float(s), 0.0


def _split_blocks(path):
    with open(path, newline="") as f:
        lines = f.read().splitlines()

    blocks, cur = [], []
    for line in lines:
        if line.strip() == "":
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append(cur)
    return blocks


def _parse_block(block_lines):
    rows = [r for r in csv.reader(block_lines) if r and any(c.strip() for c in r)]

    title = ""
    if rows and rows[0][0].lstrip().startswith("#"):
        title = rows.pop(0)[0].lstrip().lstrip("#").strip()

    if not rows:
        raise ValueError("an env block needs at least one bar line")

    bars = []
    for row in rows:
        label = row[0].strip()
        hatched = label.startswith("~")
        if hatched:
            label = label[1:].strip()
        label, _, color_ref = label.partition("=")
        label, color_ref = label.strip(), color_ref.strip() or None

        cells = [c for c in row[1:] if c.strip() != ""]
        if len(cells) != 1:
            raise ValueError(f"bar {label!r} needs exactly one value, got {len(cells)}")
        try:
            value, err = _parse_value(cells[0])
        except ValueError as e:
            raise ValueError(f"non-numeric value for bar {label!r}: {e}")

        bars.append({"label": label, "value": value, "err": err,
                     "hatched": hatched, "color_ref": color_ref})

    return {"title": title, "bars": bars}


def read_csv(path):
    blocks = _split_blocks(path)
    if not blocks:
        raise ValueError("CSV is empty")
    return [_parse_block(b) for b in blocks]


def resolve_output(output, csv_path):
    """Where to save the figure: relative paths land under outputs/."""
    out = output or (os.path.splitext(os.path.basename(csv_path))[0] + ".pdf")
    if not os.path.isabs(out):
        out = os.path.join(_OUTPUT_DIR, out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    return out


def build_figure(panels, title, ylabel):
    plt.rcParams.update(_FONT_SIZES)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n + 0.4, 4.3),
                             sharey=True, squeeze=False)
    axes = axes[0]

    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    color_of = {}
    n_used = 0
    handles = {}

    def color_for(bar):
        nonlocal n_used
        label, ref = bar["label"], bar["color_ref"]
        if label in color_of:
            return color_of[label]
        if ref and ref not in color_of:
            color_of[ref] = palette[n_used % len(palette)]
            n_used += 1
        if ref:
            color_of[label] = color_of[ref]
        else:
            color_of[label] = palette[n_used % len(palette)]
            n_used += 1
        return color_of[label]

    for ax, panel in zip(axes, panels):
        bars = panel["bars"]
        xs = np.arange(len(bars))
        for x, bar in zip(xs, bars):
            color = color_for(bar)
            bc = ax.bar(x, bar["value"], 0.7, color=color,
                        yerr=bar["err"] or None, capsize=3,
                        hatch="//" if bar["hatched"] else None,
                        edgecolor="white" if bar["hatched"] else "none")
            handles.setdefault(bar["label"], bc)

        ax.set_xticks([])
        ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.3)
        ax.set_axisbelow(True)
        ax.margins(x=0.08)
        if panel["title"]:
            ax.set_title(panel["title"])

    if ylabel:
        axes[0].set_ylabel(ylabel)
    if title:
        fig.suptitle(title)

    ncol = min(len(handles), 5)
    legend_rows = 1 + (len(handles) - 1) // ncol
    bottom = 0.04 + 0.05 * legend_rows

    fig.tight_layout(rect=(0, bottom, 1, 0.97 if title else 1.0))
    fig.legend(handles.values(), handles.keys(),
               loc="upper center", bbox_to_anchor=(0.5, bottom),
               ncol=ncol,
               frameon=True, framealpha=0.9, edgecolor="0.8", fancybox=False)
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="path to the input CSV file")
    p.add_argument("-o", "--output",
                   help="path to save the figure (extension picks the format). "
                        "Relative paths land under outputs/; default is "
                        "outputs/<csv name>.pdf. Pass --show for a window instead.")
    p.add_argument("--show", action="store_true",
                   help="open an interactive window instead of saving a file")
    p.add_argument("--title", default="", help="overall figure title")
    p.add_argument("--ytitle", default="", help="shared y-axis title")
    p.add_argument("--dpi", type=int, default=150, help="output DPI (default 150)")
    args = p.parse_args(argv)

    try:
        panels = read_csv(args.csv)
    except (OSError, ValueError) as e:
        p.error(str(e))

    fig = build_figure(panels, args.title, args.ytitle)

    if args.show and not args.output:
        plt.show()
    else:
        out = resolve_output(args.output, args.csv)
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.03)
        print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
