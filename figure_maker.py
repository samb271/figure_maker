#!/usr/bin/env python3
"""Build an arbitrary matplotlib figure from a CSV file.

CSV format
----------
The file is split into one or more *panel blocks* separated by blank lines.
Each block is drawn as its own subplot, left to right.

A block looks like::

    # Panel title                     (optional, a line starting with '#')
    <ignored label>, <x1>, <x2>, ...  (header: the x-axis values)
    <series name>,   <y1>, <y2>, ...  (one entry line per series)
    ...

Rules
-----
- One entry line with one y value per x value  -> a solid line with markers.
- One entry line with exactly one y value      -> a dashed horizontal line
  (a constant across the whole x axis).
- Prefix a series name with '~' to draw that row as a dashed line (markers
  kept, e.g. "~Oracle,0.9,0.92,0.95"); the '~' is dropped from the label.
- End a series name with "= <other series>" to give it the same colour as
  that other series (e.g. "~DINO-WM (O.S.) = DINO-WM"). Both keep their own
  legend entry and linestyle.
- A y cell may carry an error as ``mean±err`` (or ``mean+/-err``); this draws
  a shaded band around the line. Plain numbers are fine too.
- Header x values that all parse as numbers give a numeric axis; otherwise
  they become evenly spaced categorical ticks.
- A series name that appears in several panels keeps one colour and one
  shared legend entry.
- All panels share one y-range by default; pass --fit-y to scale each panel
  to its own min/max instead. --xlog / --ylog switch an axis to log scale.

For side-by-side bar charts, use bar_maker.py instead.

Titles are passed on the command line: --title is the overall figure title,
--xtitle / --ytitle label the axes, and per-panel titles come from the '#'
lines in the CSV.
"""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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


def _parse_cell(raw):
    """Return (mean, err) for a y cell, err == 0.0 when none is given."""
    s = raw.strip()
    for sep in _ERR_SEPARATORS:
        if sep in s:
            mean, err = s.split(sep, 1)
            return float(mean), abs(float(err))
    return float(s), 0.0


def _parse_x_values(raw):
    """Return (x_positions, x_ticklabels_or_None)."""
    try:
        return [float(v) for v in raw], None
    except ValueError:
        return list(range(len(raw))), list(raw)


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

    if len(rows) < 2:
        raise ValueError("a panel needs a header line and at least one entry line")

    header = [c.strip() for c in rows[0]]
    x_raw = header[1:]
    if not x_raw:
        raise ValueError("header must list at least one x-axis value")
    x_pos, x_labels = _parse_x_values(x_raw)

    entries = []
    for row in rows[1:]:
        name = row[0].strip()
        dashed = name.startswith("~")
        if dashed:
            name = name[1:].strip()
        name, _, color_ref = name.partition("=")
        name, color_ref = name.strip(), color_ref.strip() or None

        cells = [c for c in row[1:] if c.strip() != ""]
        try:
            parsed = [_parse_cell(c) for c in cells]
        except ValueError as e:
            raise ValueError(f"non-numeric y value in series {name!r}: {e}")
        means = [m for m, _ in parsed]
        errs = [e for _, e in parsed]

        if len(means) == 1:
            constant = True
        elif len(means) == len(x_pos):
            constant = False
        else:
            raise ValueError(
                f"series {name!r} has {len(means)} y values; expected 1 "
                f"(constant) or {len(x_pos)} (one per x value)"
            )
        entries.append({"name": name, "means": means, "errs": errs,
                        "constant": constant, "dashed": dashed,
                        "color_ref": color_ref})

    return {"title": title, "x_pos": x_pos, "x_labels": x_labels, "entries": entries}


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


def _pad_range(entries, frac=0.08):
    """(ymin, ymax) covering every value/band in a panel, with a margin."""
    vals = []
    for entry in entries:
        for m, e in zip(entry["means"], entry["errs"]):
            vals += [m - e, m + e]
    lo, hi = min(vals), max(vals)
    span = hi - lo
    pad = span * frac if span > 1e-9 else max(abs(hi) * 0.1, 0.05)
    return lo - pad, hi + pad


def build_figure(panels, title, xlabel, ylabel, fit_y=False,
                 xlog=False, ylog=False, band_alpha=0.15):
    plt.rcParams.update(_FONT_SIZES)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n + 0.4, 4.3),
                             sharey=not fit_y, squeeze=False)
    axes = axes[0]

    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    color_of = {}
    n_used = 0
    handles = {}

    def color_for(entry):
        """Colour for a series: reuse '= other' target, else next palette slot."""
        nonlocal n_used
        name, ref = entry["name"], entry["color_ref"]
        if name in color_of:
            return color_of[name]
        if ref and ref not in color_of:
            color_of[ref] = palette[n_used % len(palette)]
            n_used += 1
        if ref:
            color_of[name] = color_of[ref]
        else:
            color_of[name] = palette[n_used % len(palette)]
            n_used += 1
        return color_of[name]

    for ax, panel in zip(axes, panels):
        for entry in panel["entries"]:
            means, errs = entry["means"], entry["errs"]
            color = color_for(entry)

            if entry["constant"]:
                h = ax.axhline(means[0], linestyle="--", color=color)
                if errs[0]:
                    ax.axhspan(means[0] - errs[0], means[0] + errs[0],
                               color=color, alpha=band_alpha, linewidth=0)
            else:
                h, = ax.plot(panel["x_pos"], means, color=color, marker="o",
                             linestyle="--" if entry["dashed"] else "-")
                if any(errs):
                    lo = [m - e for m, e in zip(means, errs)]
                    hi = [m + e for m, e in zip(means, errs)]
                    ax.fill_between(panel["x_pos"], lo, hi,
                                    color=color, alpha=band_alpha, linewidth=0)

            handles.setdefault(entry["name"], h)

        ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.3)
        ax.set_axisbelow(True)

        if ylog:
            ax.set_yscale("log")
        elif fit_y:
            ax.set_ylim(*_pad_range(panel["entries"]))

        if panel["x_labels"] is not None:
            ax.set_xticks(panel["x_pos"])
            ax.set_xticklabels(panel["x_labels"])
        elif xlog:
            ax.set_xscale("log")
            ax.set_xticks(panel["x_pos"])
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        if panel["title"]:
            ax.set_title(panel["title"])
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel and fit_y:
            ax.set_ylabel(ylabel)

    if ylabel and not fit_y:
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
    p.add_argument("--xtitle", default="", help="x-axis title (all panels)")
    p.add_argument("--ytitle", default="", help="y-axis title")
    p.add_argument("--fit-y", dest="fit_y", action="store_true",
                   help="scale each panel's y-axis to its own min/max (default: "
                        "all panels share one y-range)")
    p.add_argument("--xlog", action="store_true",
                   help="use a logarithmic x axis (numeric x values only)")
    p.add_argument("--ylog", action="store_true",
                   help="use a logarithmic y axis")
    p.add_argument("--dpi", type=int, default=150, help="output DPI (default 150)")
    args = p.parse_args(argv)

    try:
        panels = read_csv(args.csv)
    except (OSError, ValueError) as e:
        p.error(str(e))

    if args.xlog and any(p["x_labels"] is not None for p in panels):
        p.error("--xlog needs numeric x values; some panels have categorical labels")

    fig = build_figure(panels, args.title, args.xtitle, args.ytitle,
                       fit_y=args.fit_y, xlog=args.xlog, ylog=args.ylog)

    if args.show and not args.output:
        plt.show()
    else:
        out = resolve_output(args.output, args.csv)
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.03)
        print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
