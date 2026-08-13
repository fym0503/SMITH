from __future__ import annotations

from pathlib import Path

import matplotlib as mpl


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
        "axes.linewidth": 0.6,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def save_figure(fig, output_prefix: str | Path) -> dict[str, str]:
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {}
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        path = output_prefix.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        paths[suffix] = str(path)
    return paths

