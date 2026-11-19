"""figstyle: one place for the look and size of all thesis figures.

Author: Alexander Weiser
AI Use: This script was created solely by Anthorpic's Claude.

Every figure is made at its final printed size, as a fraction of the thesis \\textwidth,
and included in LaTeX without scaling, so the text size is the same in every figure.
See HOWTO.md for how to use it.
"""

from glob import glob
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------- Sizes
TEXTWIDTH_PT = 448.13                  # \textwidth of MTP-thesis (from main.log)
TEXTWIDTH_IN = TEXTWIDTH_PT / 72.27    # TeX points -> inches: 6.20 in

# Width presets as fractions of \textwidth ("half"/"third" leave room for the gap
# between subfigures).
WIDTHS = {"full": 1.0, "wide": 0.8, "half": 0.49, "third": 0.32}

# ---------------------------------------------------------------- Colours and markers
# Same colours and order as in thesis.mplstyle (Paul Tol "muted").
COLORS = ["#332288", "#CC6677", "#117733", "#AA4499",
          "#88CCEE", "#44AA99", "#999933", "#882255"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
GREY = "#8C8C8C"                       # frame colour; for reference lines, annotations

STYLE_FILE = Path(__file__).with_name("thesis.mplstyle")

# TeX Gyre Heros (the thesis text font) comes with MacTeX, but is not a system font,
# so matplotlib only finds it after we register the files.
FONT_GLOBS = [
    "/usr/local/texlive/*/texmf-dist/fonts/opentype/public/tex-gyre/texgyreheros-*.otf",
    "/Library/TeX/Root/texmf-dist/fonts/opentype/public/tex-gyre/texgyreheros-*.otf",
]


def use():
    """Register the thesis font and apply thesis.mplstyle. Call once per script."""
    font_files = [f for pattern in FONT_GLOBS for f in glob(pattern)]
    for f in font_files:
        font_manager.fontManager.addfont(f)
    if not font_files:
        warnings.warn("TeX Gyre Heros not found, using Helvetica/Arial instead. "
                      "Add its path to FONT_GLOBS in figstyle.py.")
    plt.style.use(STYLE_FILE)


def figsize(width="full", aspect=0.4, height_in=None):
    """Return (width, height) in inches.

    width     : "full", "wide", "half", "third", or a fraction of \\textwidth (<= 1).
    aspect    : height / width, used when height_in is not given.
    height_in : fixed height in inches.
    """
    frac = WIDTHS[width] if isinstance(width, str) else width
    if not 0 < frac <= 1:
        raise ValueError(f"width must be a fraction in (0, 1], got {frac}")
    w = frac * TEXTWIDTH_IN
    return (w, height_in if height_in is not None else aspect * w)


def panel_labels(fig, axes):
    """Put bold (a), (b), ... above each axes, all at the same height.

    Call just before saving. Makes room at the top of the figure for the labels.
    """
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.92))   # keep top 8 % free
    fig.canvas.draw()                                    # fix the axes positions
    y = axes[0].get_position().y1 + 0.02                 # just above the first axes
    for ax, letter in zip(axes, "abcdefgh"):
        x = ax.get_tightbbox().transformed(fig.transFigure.inverted()).x0  # incl. tick labels
        fig.text(x, y, f"({letter})", ha="left", va="bottom", fontweight="bold")


def save(fig, name, outdir="figures", png=False):
    """Save outdir/name.pdf (and outdir/name.png if png=True) at the exact figure size."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf")
    if png:
        fig.savefig(outdir / f"{name}.png")


# This was created by Claude.
