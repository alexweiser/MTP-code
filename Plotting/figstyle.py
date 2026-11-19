"""figstyle: the common look of all thesis figures.

Author: Alexander Weiser
AI Use: This script was created solely by Anthropic's Claude.

    use()       apply thesis.mplstyle and the thesis font (call once per script)
    figsize()   figure size in inches, as a fraction of the thesis \\textwidth
    COLORS, MARKERS, GREY

Everything else (plotting, layout, saving) is done in each script as usual.
See HOWTO.md.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent

# \textwidth of MTP-thesis (from main.log), converted from TeX points to inches: 6.20 in
TEXTWIDTH_IN = 448.13 / 72.27

# Same colours and order as in thesis.mplstyle (Paul Tol "muted", colour-blind safe)
COLORS = ["#332288", "#CC6677", "#117733", "#AA4499",
          "#88CCEE", "#44AA99", "#999933", "#882255"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
GREY = "#8C8C8C"                       # for reference lines and annotations


def use():
    """Load the thesis font from Plotting/fonts/ and apply thesis.mplstyle."""
    font_files = list((HERE / "fonts").glob("*.otf"))
    for f in font_files:
        font_manager.fontManager.addfont(f)        # make the font known to matplotlib
    if not font_files:
        warnings.warn("No thesis font in Plotting/fonts/, using Helvetica/Arial instead.")
    plt.style.use(HERE / "thesis.mplstyle")


def figsize(width=1.0, aspect=0.4):
    """Return (width, height) in inches for plt.subplots(figsize=...).

    width  : fraction of \\textwidth, e.g. 1.0 (full), 0.49 (two side by side),
             0.32 (three side by side)
    aspect : height / width
    """
    w = width * TEXTWIDTH_IN
    return (w, aspect * w)


# This was created by Claude.
