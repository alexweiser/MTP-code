# HOWTO: thesis figures with figstyle

AI Use: This script was created solely by Anthorpic´s Claude.

All thesis figures use one common style (`thesis.mplstyle`, style "D": thin grey frame,
light grid, Paul Tol "muted" colours, TeX Gyre Heros 9/8 pt like the thesis text).
Figures are made at their **final printed size** and are never rescaled afterwards, so
the text size is the same in every figure. Sizes are fractions of the thesis
`\textwidth` (448.13 pt = 6.20 in).

## 1. Import figstyle

A script in `Plotting/` can import it directly. A script in another folder
(e.g. `VCA_Scintillation/`) first adds `Plotting/` to the import path:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Plotting"))  # parents[1] = MTP-code/

import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

fs.use()   # style + thesis font; call once, before any plotting
```

## 2. Make the figure

Create the figure with `fs.figsize(...)`, then plot as usual:

```python
fig, ax = plt.subplots(figsize=fs.figsize("full", aspect=0.4))

for i, (label, (x, y)) in enumerate(my_data.items()):
    ax.plot(x, y, marker=fs.MARKERS[i], color=fs.COLORS[i], label=label)

ax.set_xlabel(r"Mean photons per pulse $\langle n \rangle$")
ax.set_ylabel(r"Detection efficiency $\eta$")
ax.legend()
```

| `fs.figsize(width, ...)` | Width |
|---|---|
| `"full"` | 1.00 × `\textwidth` |
| `"wide"` | 0.80 × `\textwidth` |
| `"half"` | 0.49 × `\textwidth` (two side by side) |
| `"third"` | 0.32 × `\textwidth` (three side by side) |
| `0.6` (any number ≤ 1) | that fraction of `\textwidth` |

Height: `aspect=` (height / width, default 0.4) or `height_in=` (fixed height in inches,
useful to give figures that sit side by side the same height).

Rules:
- Do not change font sizes, and do not use `bbox_inches="tight"` (it changes the size).
- Give each series a colour **and** a marker: `fs.COLORS[i]`, `fs.MARKERS[i]`.
  `fs.GREY` is for reference lines and annotations.
- Images (`imshow`): switch the grid off with `ax.grid(False)`; the colormap is magma.
- Several panels: make the subplots as usual, then call `fs.panel_labels(fig, axes)`
  just before saving. It adds (a), (b), (c) at the same height on every panel.

## 3. Save

```python
fs.save(fig, "sq1_efficiency_vs_mu",
        outdir=Path(__file__).resolve().parents[2] / "MTP-thesis" / "figures")
```

This writes `MTP-thesis/figures/sq1_efficiency_vs_mu.pdf` (vector, exact size; the
folder is created if needed). Add `png=True` for a PNG preview as well.

## 4. Run

- VS Code: select the `MTPvenv` interpreter (Cmd+Shift+P → "Python: Select Interpreter"), then Run.
- Terminal (from `MTP-code/`): `source MTPvenv/bin/activate`, then `python <folder>/<script>.py`.

If you get the warning "TeX Gyre Heros not found", add the font path
(`kpsewhich texgyreheros-regular.otf`) to `FONT_GLOBS` at the top of `figstyle.py`.

To start a new plotting script, copy `example_figstyle.py` and replace the dummy data.
Run it once to see the expected output in `Plotting/example_output/`.

## If the thesis layout changes

Update `TEXTWIDTH_PT` in `figstyle.py` (read `\textwidth` from `main.log` of the
thesis) and re-run all plotting scripts.

---
