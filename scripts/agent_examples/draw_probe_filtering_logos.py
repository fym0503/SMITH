#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "figures" / "agent_probe_filtering_logos"

COLORS = {
    "bg": "#eef6fb",
    "ring": "#d6e7f1",
    "main": "#1f77b4",
    "soft": "#8ec3df",
    "muted": "#cfd8dc",
    "dark": "#27343b",
}


def setup() -> None:
    mpl.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "svg.fonttype": "none", "pdf.fonttype": 42})


def base_ax() -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    fig, ax = plt.subplots(figsize=(2.4, 2.4))
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((256, 256), 224, facecolor=COLORS["bg"], edgecolor="none"))
    ax.add_patch(Circle((256, 256), 207, facecolor="none", edgecolor=COLORS["ring"], linewidth=10))
    return fig, ax


def line(ax, xs, ys, color="main", lw=15, transform=None):
    ax.plot(xs, ys, color=COLORS[color], linewidth=lw, solid_capstyle="round", solid_joinstyle="round", transform=transform or ax.transData)


def save(fig, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"{stem}.{suffix}", bbox_inches="tight", dpi=600)
    plt.close(fig)


def designability() -> None:
    fig, ax = base_ax()
    ax.add_patch(FancyBboxPatch((132, 112), 244, 288, boxstyle="round,pad=0,rounding_size=28", facecolor="white", edgecolor=COLORS["muted"], linewidth=9))
    for y, x2 in [(176, 338), (226, 290), (276, 328)]:
        line(ax, [178, x2], [y, y], "soft", 11)
    trans = Affine2D().rotate_deg_around(256, 310, -32) + ax.transData
    ax.add_patch(FancyBboxPatch((160, 286), 216, 54, boxstyle="round,pad=0,rounding_size=14", facecolor="white", edgecolor=COLORS["main"], linewidth=13, transform=trans))
    for x, h in [(190, 24), (230, 16), (270, 24), (310, 16), (350, 24)]:
        line(ax, [x, x], [286, 286 + h], "main", 13, transform=trans)
    ptrans = Affine2D().rotate_deg_around(238, 260, -36) + ax.transData
    ax.add_patch(Rectangle((222, 130), 44, 222, facecolor=COLORS["main"], edgecolor="none", transform=ptrans))
    ax.add_patch(Polygon([[222, 352], [258, 352], [240, 394]], facecolor=COLORS["soft"], edgecolor="none", transform=ptrans))
    line(ax, [224, 264], [130, 130], "dark", 10, transform=ptrans)
    save(fig, "designability")


def specificity() -> None:
    fig, ax = base_ax()
    ax.add_patch(Circle((234, 220), 118, facecolor="white", edgecolor=COLORS["main"], linewidth=16))
    ax.add_patch(Circle((234, 220), 72, facecolor="none", edgecolor=COLORS["soft"], linewidth=16))
    ax.add_patch(Circle((234, 220), 25, facecolor=COLORS["main"], edgecolor="none"))
    line(ax, [318, 410], [304, 396], "main", 16)
    line(ax, [386, 416], [372, 402], "main", 16)
    line(ax, [128, 288], [376, 376], "muted", 16)
    line(ax, [128, 240], [414, 414], "muted", 16)
    line(ax, [146, 174], [376, 348], "soft", 16)
    line(ax, [234, 304], [220, 156], "main", 16)
    save(fig, "specificity")


def deployability() -> None:
    fig, ax = base_ax()
    ax.add_patch(FancyBboxPatch((124, 164), 264, 222, boxstyle="round,pad=0,rounding_size=28", facecolor="white", edgecolor=COLORS["muted"], linewidth=10))
    ax.add_patch(FancyBboxPatch((120, 164), 288, 68, boxstyle="round,pad=0,rounding_size=28", facecolor=COLORS["main"], edgecolor="none", alpha=0.12))
    line(ax, [176, 176, 264, 264], [164, 134, 134, 164], "main", 16)
    line(ax, [124, 388], [232, 232], "main", 16)
    for y, x2 in [(286, 340), (334, 306)]:
        ax.add_patch(Circle((180, y), 15, facecolor=COLORS["soft"], edgecolor="none"))
        line(ax, [214, x2], [y, y], "soft", 14)
    line(ax, [300, 340, 426], [332, 370, 266], "main", 16)
    save(fig, "deployability")


def main() -> None:
    setup()
    designability()
    specificity()
    deployability()
    print(f"Wrote logo assets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
