"""Template for a thesis figure made with figstyle (dummy data only).

Author: Alexander Weiser
AI Use: This script was created solely by Anthropic's Claude.

Run:     python Plotting/example_figstyle.py      (from MTP-code/)
Output:  Plotting/example_output/example_full.pdf/.png and example_half.pdf/.png
To start a new plotting script, copy this file and replace the dummy data.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Make figstyle.py importable. This file is in Plotting/ itself; from another folder
# (e.g. Turbulence_Propagation_Simulation/) use: Path(__file__).resolve().parents[1] / "Plotting"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

fs.use()

OUT = Path(__file__).resolve().parent / "example_output"   # where this script saves
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- dummy data (not physics)
mu = np.logspace(-1, 2.3, 14)                     # "mean photons per pulse"
mu_fine = np.logspace(-1, 2.3, 200)
cases = {r"$D/r_0 = 0$": 1, r"$D/r_0 = 2$": 3, r"$D/r_0 = 8$": 12}


def dummy_curve(mu, n):
    return 0.6 * n * (1 - np.exp(-mu / n)) / mu


rate = np.linspace(0, 2, 200)                     # "symbol rate" in GHz
x = np.linspace(-1, 1, 40)                        # "position" in mm
X, Y = np.meshgrid(x, x)
image = np.exp(-(X**2 + Y**2) / 0.2)              # "spot"

# ---------------------------------------------------------------- 1) full width, 3 panels
fig, (a, b, c) = plt.subplots(1, 3, figsize=fs.figsize(1.0, aspect=0.35))

for i, (label, n) in enumerate(cases.items()):
    # data: colour + marker; model: line in the same colour
    a.plot(mu, dummy_curve(mu, n), linestyle="none",
           color=fs.COLORS[i], marker=fs.MARKERS[i], label=label)
    a.plot(mu_fine, dummy_curve(mu_fine, n), color=fs.COLORS[i])
    b.plot(rate, 0.5 * np.exp(-rate * (2 + n / 3)), color=fs.COLORS[i])
a.set(xscale="log", xlabel=r"Mean photons per pulse $\langle n \rangle$",
      ylabel=r"Efficiency $\eta$", ylim=(0, 0.95), title="(a)")
a.legend(loc="upper right")
b.axhline(3.8e-3, color=fs.GREY, linestyle="--")  # reference line
b.set(yscale="log", xlabel="Symbol rate (GHz)", ylabel="BER", title="(b)")

im = c.imshow(image, extent=[-1, 1, -1, 1], origin="lower")
c.grid(False)                                     # no grid on images
c.set(xlabel=r"$x$ (mm)", ylabel=r"$y$ (mm)", title="(c)")
fig.colorbar(im, ax=c, label="Norm. intensity")

fig.savefig(OUT / "example_full.pdf")             # PDF for the thesis
fig.savefig(OUT / "example_full.png")             # PNG preview

# ---------------------------------------------------------------- 2) half width, 1 panel
fig, ax = plt.subplots(figsize=fs.figsize(0.49, aspect=0.75))
for i, (label, n) in enumerate(cases.items()):
    ax.plot(mu_fine, dummy_curve(mu_fine, n), label=label)   # colours come from the style
ax.set(xscale="log", xlabel=r"$\langle n \rangle$", ylabel=r"$\eta$")
ax.legend()
fig.savefig(OUT / "example_half.pdf")
fig.savefig(OUT / "example_half.png")

print("Saved to", OUT)


# This was created by Claude.
