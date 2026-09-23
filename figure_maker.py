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
- Prefix a series name with '!' to mark it as background context rather than
  an evaluated method (e.g. "!Coverage,0.9,0.7,0.5"): it is drawn as a
  shaded grey area with a dotted outline, behind everything else, with a
  patch-style legend swatch instead of a line+marker - so it can't be
  mistaken for a baseline or an evaluated method. Leave its y values blank
  to skip it for a panel where it hasn't been computed yet.
- All panels share one y-range by default; pass --fit-y to scale each panel
  to its own min/max instead. --xlog / --ylog switch an axis to log scale (--xlog-panel N limits the
  x log scale to panel N).
  With a shared y-range, only the first panel shows tick labels by default;
  pass --all-yticks to repeat them on every panel (the y title still appears
  only once).

For side-by-side bar charts, use bar_maker.py instead.

Titles are passed on the command line: --title is the overall figure title,
--xtitle / --ytitle label the axes, and per-panel titles come from the '#'
lines in the CSV.
"""

import argparse
import csv
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_ERR_SEPARATORS = ("±", "+/-", "+-")

_OUTPUT_DIR = "outputs"

_MIN_LEGEND_FONT = 9  # smallest legend text used to keep it on one row

_CONTEXT_COLOR = "0.55"  # fixed neutral grey for '!'-prefixed context series

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
        context = name.startswith("!")
        if context:
            name = name[1:].strip()
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

        if context and not means:
            continue  # not computed yet for this panel

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
                        "color_ref": color_ref, "context": context})

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


def _overlap_offsets(entries, x_pos, frac=0.015):
    """Small per-entry y-offset (visual only) so series with identical values
    don't fully overlap. Entries are grouped by their (expanded) y values.

    A group whose shared value is flat (a horizontal line) is left at offset
    0 here and returned separately in `flat_groups` - those get spread by
    exactly one line-width later, once the axes' pixel scale is known. A
    group that varies with x is spread immediately, symmetrically around its
    true value, by multiples of `frac` * panel range."""
    n_x = len(x_pos)

    def expanded(entry):
        means = entry["means"]
        if entry["constant"]:
            means = means * n_x
        return tuple(round(m, 9) for m in means)

    groups = {}
    for entry in entries:
        groups.setdefault(expanded(entry), []).append(entry)

    vals = [m + e for entry in entries for m, e in zip(entry["means"], entry["errs"])]
    vals += [m - e for entry in entries for m, e in zip(entry["means"], entry["errs"])]
    span = (max(vals) - min(vals)) if vals else 0
    delta = span * frac

    offsets = {}
    flat_groups = []
    for key, group in groups.items():
        if len(group) < 2:
            offsets[id(group[0])] = 0.0
        elif len(set(key)) == 1:
            flat_groups.append(group)
            for entry in group:
                offsets[id(entry)] = 0.0
        else:
            n = len(group)
            for i, entry in enumerate(group):
                offsets[id(entry)] = (i - (n - 1) / 2) * delta
    return offsets, flat_groups


def _linewidth_to_data(ax, handle):
    """Convert a line's rendered width (points) to a y-axis data delta."""
    lw_px = handle.get_linewidth() * ax.figure.dpi / 72.0
    inv = ax.transData.inverted()
    return abs(inv.transform((0, lw_px))[1] - inv.transform((0, 0))[1])


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


def build_figure(panels, title, xlabels, ylabel, fit_y=False,
                 xlog_panels=(), ylog=False, band_alpha=0.15, all_yticks=False,
                 height=4.3):
    plt.rcParams.update(_FONT_SIZES)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n + 0.4, height),
                             sharey=not fit_y, squeeze=False)
    axes = axes[0]

    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    color_of = {}
    n_used = 0
    handles = {}

    def color_for(entry):
        """Colour for a series: reuse '= other' target, else next palette slot.
        Context series always get the same fixed grey, outside the palette."""
        nonlocal n_used
        if entry["context"]:
            return _CONTEXT_COLOR
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

    flat_dup_pending = []  # (ax, handle, index_in_group, group_size)

    for i, (ax, panel) in enumerate(zip(axes, panels)):
        plotted_entries = [e for e in panel["entries"] if not e["context"]]
        offsets, flat_groups = _overlap_offsets(plotted_entries, panel["x_pos"])
        entry_handle = {}
        for entry in panel["entries"]:
            color = color_for(entry)

            if entry["context"]:
                means = (entry["means"] * len(panel["x_pos"])
                         if entry["constant"] else entry["means"])
                ax.fill_between(panel["x_pos"], 0, means, color=color,
                                alpha=0.15, linewidth=0, zorder=1)
                ax.plot(panel["x_pos"], means, color=color, linestyle=":",
                       linewidth=1.5, zorder=1)
                handles.setdefault(entry["name"], mpatches.Patch(
                    facecolor=color, alpha=0.3, edgecolor=color,
                    linestyle=":", label=entry["name"]))
                continue

            offset = offsets[id(entry)]
            means = [m + offset for m in entry["means"]]
            errs = entry["errs"]

            # Dashed lines draw above solid ones (regardless of CSV row order) so
            # that an exact overlap shows as an alternating pattern instead of
            # the solid line's unbroken fill fully hiding the dashed one.
            zorder = 3 if entry["dashed"] else 2

            if entry["constant"]:
                h = ax.axhline(means[0], linestyle="--", color=color, zorder=zorder)
                if errs[0]:
                    ax.axhspan(means[0] - errs[0], means[0] + errs[0],
                               color=color, alpha=band_alpha, linewidth=0)
            else:
                h, = ax.plot(panel["x_pos"], means, color=color, marker="o",
                             linestyle="--" if entry["dashed"] else "-",
                             zorder=zorder)
                if any(errs):
                    lo = [m - e for m, e in zip(means, errs)]
                    hi = [m + e for m, e in zip(means, errs)]
                    ax.fill_between(panel["x_pos"], lo, hi,
                                    color=color, alpha=band_alpha, linewidth=0)

            handles.setdefault(entry["name"], h)
            entry_handle[id(entry)] = h

        for group in flat_groups:
            n_g = len(group)
            for i, entry in enumerate(group):
                flat_dup_pending.append((ax, entry_handle[id(entry)], i, n_g))

        ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.3)
        ax.set_axisbelow(True)

        if ylog:
            ax.set_yscale("log")
        elif fit_y:
            ax.set_ylim(*_pad_range(panel["entries"]))

        if panel["x_labels"] is not None:
            ax.set_xticks(panel["x_pos"])
            ax.set_xticklabels(panel["x_labels"])
        elif i in xlog_panels:
            ax.set_xscale("log")
            ax.set_xticks(panel["x_pos"])
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        if panel["title"]:
            ax.set_title(panel["title"])
        if xlabels:
            ax.set_xlabel(xlabels[i] if len(xlabels) > 1 else xlabels[0])
        if ylabel and fit_y:
            ax.set_ylabel(ylabel)

    if all_yticks and not fit_y:
        for ax in axes[1:]:
            ax.tick_params(labelleft=True)

    if ylabel and not fit_y:
        axes[0].set_ylabel(ylabel)
    if title:
        fig.suptitle(title)

    # The legend is anchored at the figure's bottom edge, and its column count
    # is lowered until it fits the figure width. The saved size is then always
    # exactly the figsize (no bbox_inches="tight"), so a PDF's width doesn't
    # depend on how long the legend is and every figure scales the same way
    # when included in a paper.
    renderer = fig.canvas.get_renderer()

    def make_legend(ncol, fontsize):
        legend = fig.legend(handles.values(), handles.keys(),
                            loc="lower center", bbox_to_anchor=(0.5, 0.0),
                            ncol=ncol, borderaxespad=0.2, fontsize=fontsize,
                            columnspacing=1.2, handletextpad=0.5,
                            frameon=True, framealpha=0.9, edgecolor="0.8",
                            fancybox=False)
        bbox = legend.get_window_extent(renderer).transformed(
            fig.transFigure.inverted())
        return legend, bbox

    # Prefer a single row: shrink the legend text (down to a floor) before
    # resorting to wrapping it onto several rows.
    full = _FONT_SIZES["legend.fontsize"]
    for fontsize in range(full, _MIN_LEGEND_FONT - 1, -1):
        legend, bbox = make_legend(len(handles), fontsize)
        if bbox.width <= 0.98:
            break
        legend.remove()
    else:
        ncol = min(len(handles), 5)
        while True:
            legend, bbox = make_legend(ncol, full)
            if ncol == 1 or bbox.width <= 0.98:
                break
            legend.remove()
            ncol -= 1
    bottom = bbox.y1 + 0.02

    fig.tight_layout(rect=(0, bottom, 1, 0.97 if title else 1.0))

    if flat_dup_pending:
        # transData is only final once the layout above has settled.
        fig.canvas.draw()
        for ax, handle, i, n_g in flat_dup_pending:
            lw_data = _linewidth_to_data(ax, handle) * 1.5
            offset = (i - (n_g - 1) / 2) * lw_data
            handle.set_ydata([y + offset for y in handle.get_ydata()])

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
    p.add_argument("--xtitle", action="append", default=[],
                   help="x-axis title. Give it once to use it on every panel, "
                        "or repeat it once per panel (left to right) for "
                        "different titles, e.g. --xtitle 'Number of subgoals' "
                        "--xtitle 'Soft-min temperature $\\tau$'")
    p.add_argument("--ytitle", default="", help="y-axis title")
    p.add_argument("--fit-y", dest="fit_y", action="store_true",
                   help="scale each panel's y-axis to its own min/max (default: "
                        "all panels share one y-range)")
    p.add_argument("--all-yticks", dest="all_yticks", action="store_true",
                   help="show y-axis tick labels on every panel instead of just "
                        "the first (only applies when panels share one y-range, "
                        "i.e. without --fit-y); the y title is still shown once")
    p.add_argument("--xlog", action="store_true",
                   help="use a logarithmic x axis (numeric x values only)")
    p.add_argument("--xlog-panel", dest="xlog_panel", type=int, action="append",
                   default=[], metavar="N",
                   help="log x axis on panel N only (1 = leftmost); repeat for "
                        "several panels")
    p.add_argument("--ylog", action="store_true",
                   help="use a logarithmic y axis")
    p.add_argument("--height", type=float, default=4.3,
                   help="figure height in inches (default 4.3); lower it to "
                        "squish the plots vertically")
    p.add_argument("--dpi", type=int, default=150, help="output DPI (default 150)")
    args = p.parse_args(argv)

    try:
        panels = read_csv(args.csv)
    except (OSError, ValueError) as e:
        p.error(str(e))

    if any(not 1 <= n <= len(panels) for n in args.xlog_panel):
        p.error(f"--xlog-panel must be between 1 and {len(panels)}")
    xlog_panels = (set(range(len(panels))) if args.xlog
                   else {n - 1 for n in args.xlog_panel})
    if any(panels[i]["x_labels"] is not None for i in xlog_panels):
        p.error("log x axis needs numeric x values; a selected panel has "
                "categorical labels")

    if len(args.xtitle) > 1 and len(args.xtitle) != len(panels):
        p.error(f"got {len(args.xtitle)} --xtitle values for {len(panels)} panels; "
                "give one (used for all panels) or one per panel")

    fig = build_figure(panels, args.title, args.xtitle, args.ytitle,
                       fit_y=args.fit_y, xlog_panels=xlog_panels, ylog=args.ylog,
                       all_yticks=args.all_yticks, height=args.height)

    if args.show and not args.output:
        plt.show()
    else:
        out = resolve_output(args.output, args.csv)
        fig.savefig(out, dpi=args.dpi)
        print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
