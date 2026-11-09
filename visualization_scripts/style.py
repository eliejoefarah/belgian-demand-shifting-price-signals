"""Shared figure style for the thesis figures (static matplotlib PNGs).

Colours: the dataviz skill's validated default categorical palette, light mode, slots in
fixed order (validated 9 Nov 2026 with validate_palette.js for slots 1-3: all checks pass;
slot 3 (aqua) is below 3:1 contrast on the surface, so every series also gets a line style
and a legend/label, never colour alone).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID, SURF = "#0b0b0b", "#5f5e5a", "#e6e5e0", "#fcfcfb"


def apply_style():
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.facecolor": SURF, "figure.facecolor": SURF, "savefig.facecolor": SURF,
    })


def tidy(ax):
    """Recessive grid, no top/right spines."""
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
