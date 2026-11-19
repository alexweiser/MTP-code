"""Template for a thesis figure made with figstyle (dummy data only).

Author: Alexander Weiser
AI Use: This script was created solely by Anthorpic's Claude.

Run:     python Plotting/example_figstyle.py
Output:  Plotting/example_output/example_full.pdf/.png and example_half.pdf/.png
To start a new plotting script, copy this file and replace the dummy data.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Make figstyle.py importable. This file is in Plotting/ itself; from another folder
# (e.g. VCA_Scintillation/) use: Path(__file__).resolve().parents[1] / "Plotting"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

fs.use()
OUT = Path(__file__).resolve().parent / "example_output"
rng = np.random.default_rng(1)

# ---------------------------------------------------------------- dummy data (not physics)
mu = np.logspace(-1, 2.3, 14)                     # "mean photons per pulse"
mu_fine = np.logspace(-1, 2.3, 200)
cases = {r"$D/r_0 = 0$": 1, r"$D/r_0 = 2$": 3, r"$D/r_0 = 8$": 12}


def dummy_curve(mu, n):
    return 0.6 * n * (1 - np.exp(-mu / n)) / mu


rate = np.linspace(0, 2, 200)                     # "symbol rate" in GHz
x = np.linspace(-1, 1, 40)                        # "position" in mm
X, Y = np.meshgrid(x, x)
image = np.exp(-(X**2 + Y**2) / 0.2) * rng.uniform(0.3, 1, X.shape)  # noisy spot
image /= image.max()

# ---------------------------------------------------------------- 1) full width, 3 panels
fig, (a, b, c) = plt.subplots(1, 3, figsize=fs.figsize("full", height_in=2.2))

for i, (label, n) in enumerate(cases.items()):
    y = dummy_curve(mu, n) + rng.normal(0, 0.01, mu.size)
    a.errorbar(mu, y, yerr=0.02, fmt=fs.MARKERS[i], color=fs.COLORS[i], label=label)
    a.plot(mu_fine, dummy_curve(mu_fine, n), color=fs.COLORS[i])
    b.plot(rate, 0.5 * np.exp(-rate * (2 + n / 3)), color=fs.COLORS[i])
a.set(xscale="log", xlabel=r"Mean photons per pulse $\langle n \rangle$",
      ylabel=r"Efficiency $\eta$", ylim=(0, 0.7))
a.legend()
b.set(yscale="log", xlabel="Symbol rate (GHz)", ylabel="BER", ylim=(1e-3, 0.6))

im = c.imshow(image, extent=[-1, 1, -1, 1], origin="lower")
c.grid(False)                                     # no grid on images
c.set(xlabel=r"$x$ (mm)", ylabel=r"$y$ (mm)", xticks=[-1, 0, 1], yticks=[-1, 0, 1])
fig.colorbar(im, ax=c, fraction=0.05, label="Norm. intensity")

fs.panel_labels(fig, [a, b, c])
fs.save(fig, "example_full", OUT, png=True)

# ---------------------------------------------------------------- 2) half width, 1 panel
fig, ax = plt.subplots(figsize=fs.figsize("half", aspect=0.75))
for i, (label, n) in enumerate(cases.items()):
    ax.plot(mu_fine, dummy_curve(mu_fine, n), color=fs.COLORS[i], label=label)
ax.set(xscale="log", xlabel=r"$\langle n \rangle$", ylabel=r"$\eta$")
ax.legend()
fs.save(fig, "example_half", OUT, png=True)
print("Saved to", OUT)


# This was created by Claude.
