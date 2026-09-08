# figure_maker

Two small scripts that turn a plain CSV into a publication-style matplotlib
figure with side-by-side panels, a shared legend, and a tight PDF crop.

- **`figure_maker.py`** — line plots (curves over an x axis).
- **`bar_maker.py`** — grouped bar charts (one bar per row, one panel per env).

Both read the same block-structured CSV, share the same styling, and write to
`outputs/` (git-ignored).

## Install

Needs `matplotlib` and `numpy`:

```sh
python -m venv .venv
.venv/bin/pip install matplotlib numpy
```

## Usage

```sh
.venv/bin/python figure_maker.py figure_maker_template.csv \
  --xtitle "Budget" --ytitle "Success rate"

.venv/bin/python bar_maker.py bar_maker_template.csv \
  --ytitle "Success rate"
```

Each run writes `outputs/<csv name>.pdf`. Use `-o name.png` for another
format/name (relative paths still land in `outputs/`), or `--show` for an
interactive window.

Common options: `--title` (figure suptitle), `--ytitle`, `--dpi`.
`figure_maker.py` also has `--xtitle`, `--fit-y`, `--xlog`, `--ylog`.

## CSV format

The file is split into **blocks** separated by blank lines; each block becomes
one panel, left to right. A block may start with a `# Title` line.

### `figure_maker.py`

```
# Env A
method,10,20,50,100          <- header: first cell ignored, rest are x values
Baseline,0.30,0.45,0.60,0.68 <- one line per series
Method X,0.35,0.55,0.75±0.03,0.88±0.02
~Method X (variant) = Method X,0.40,0.60,0.80,0.90
Oracle,0.95                  <- single value -> dashed horizontal constant
```

- **Full row** → solid line with markers.
- **Single value** → dashed horizontal line (a constant across the x axis).
- **`~name`** → dashed line (markers kept).
- **`name = other`** → same colour as series `other`; keeps its own legend
  entry and linestyle.
- **`mean±err`** cell (or `mean+/-err`) → shaded band around the line.
- Header x values that all parse as numbers give a numeric axis; otherwise
  they become evenly spaced categorical ticks.
- Panels share one y-range by default; `--fit-y` scales each to its own data.
- Titles may contain matplotlib mathtext, e.g. `--xtitle 'Temperature $\tau$'`.

### `bar_maker.py`

```
# Env A
Baseline, 0.42
Method X, 0.61±0.04
~Method X (variant) = Method X, 0.55
Method Y, 0.78
Oracle, 0.95
```

- Each line is **one bar**: `label, value` or `label, value±err` (error → cap).
- **`~label`** → hatched bar.
- **`label = other`** → same colour as bar `other`.
- A label reused across panels keeps one colour and one legend entry.
- All panels share the same y scale.

See [`figure_maker_template.csv`](figure_maker_template.csv) and
[`bar_maker_template.csv`](bar_maker_template.csv) for runnable examples.
