# HOWTO: thesis figures with figstyle

AI Use: This script was created solely by Anthropic's Claude.

All thesis figures use one common style, so they look alike: thin grey frame, light
grid, Paul Tol "muted" colours, TeX Gyre Heros 9/8 pt like the thesis text.
figstyle only sets this **look**. Plotting, layout and saving are done in each script
as usual with matplotlib.

| File | What it is |
|---|---|
| `thesis.mplstyle` | the style: fonts, font sizes, colours, frame, grid, legend |
| `figstyle.py` | `use()`, `figsize()`, `COLORS`, `MARKERS`, `GREY` |
| `example_figstyle.py` | template script with dummy data; output in `example_output/` |
| `fonts/` | put the TeX Gyre Heros `.otf` files here |

## Use

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Plotting"))  # parents[1] = MTP-code/

import matplotlib.pyplot as plt
import figstyle as fs

fs.use()                                              # once, before plotting

fig, ax = plt.subplots(figsize=fs.figsize(1.0, aspect=0.4))
for i, (label, (x, y)) in enumerate(my_data.items()):
    ax.plot(x, y, color=fs.COLORS[i], marker=fs.MARKERS[i], label=label)
ax.set(xlabel=r"Mean photons per pulse $\langle n \rangle$", ylabel=r"Efficiency $\eta$")
ax.legend()

fig.savefig(OUT / "efficiency_vs_mu.pdf")             # OUT: chosen by your script
```

To start a new plotting script, copy `example_figstyle.py` and replace the dummy data.

## Size

`fs.figsize(width, aspect)` gives the size in inches: `width` is a fraction of the
thesis `\textwidth` (6.20 in), `aspect` is height / width.

| `width` | Use |
|---|---|
| `1.0` | full text width |
| `0.49` | two figures side by side |
| `0.32` | three figures side by side |

Include the figure in LaTeX **without** a width, so it is not scaled and the font sizes
stay as in the style: `\includegraphics{figures/efficiency_vs_mu}`.

## Rules

- Make the figure with `figsize=fs.figsize(...)` and do not change font sizes.
- Save with `fig.savefig(path)`, **without** `bbox_inches="tight"` (it changes the size).
- Same colour and marker for the same case in every figure: `fs.COLORS[i]`, `fs.MARKERS[i]`.
  `fs.GREY` for reference lines.
- No titles (the caption explains the figure). Panel labels: `ax.set_title("(a)")`,
  which the style puts bold at the top left.
- Images: `ax.grid(False)`. The default colormap is magma; for phase use `cmap="twilight"`.

## Font

The thesis font TeX Gyre Heros is included in `Plotting/fonts/` (four `.otf` files, free
under the GUST Font License, from the CTAN package `tex-gyre`), so it works on any
computer without installing anything. If the files are missing, you get a warning and
figures use Helvetica/Arial.

## If the thesis layout changes

Update `TEXTWIDTH_IN` in `figstyle.py` (`\textwidth` from `main.log` of the thesis, in
pt, divided by 72.27).

---
