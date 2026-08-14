from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


METHOD_ORDER = ["SMITH", "PERSIST-class", "PERSIST", "ActiveSVM", "scGIST", "scGeneFit", "Spapros"]
METHOD_COLORS = {
    "SMITH": "#3E7FAF",
    "PERSIST-class": "#91B0D8",
    "PERSIST": "#ACC3E2",
    "ActiveSVM": "#C5D3E8",
    "scGIST": "#D5E0EE",
    "scGeneFit": "#E3EAF2",
    "Spapros": "#EEF2F6",
}


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def save_figure(fig, output_prefix: str | Path, *, tight: bool = False) -> dict[str, str]:
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {}
    exports = (("png", {"dpi": 350}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600}))
    for suffix, kwargs in exports:
        path = output_prefix.with_suffix(f".{suffix}")
        fig.savefig(
            path,
            bbox_inches="tight" if tight else None,
            pad_inches=0.02 if tight else 0.1,
            facecolor="white",
            **kwargs,
        )
        paths[suffix] = str(path)
    return paths


def save_method_legend(methods: list[str], output_prefix: str | Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(6.7, 0.42), facecolor="white")
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=METHOD_COLORS[method], edgecolor="black", linewidth=0.4)
        for method in methods
    ]
    ax.legend(
        handles,
        methods,
        loc="center",
        ncol=len(methods),
        fontsize=7,
        handlelength=1.25,
        columnspacing=1.0,
        borderaxespad=0,
    )
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    paths = save_figure(fig, output_prefix, tight=True)
    plt.close(fig)
    return paths
