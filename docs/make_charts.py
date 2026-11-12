"""Draw the capacity chart of docs/benchmarks.md from the raw results, in a light and a dark version.

    pip install matplotlib
    python docs/make_charts.py [docs/results/a100-2026-11-12] [--png]

Writes docs/assets/capacity-light.svg and docs/assets/capacity-dark.svg. The README picks one with <picture>.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

DOCS = Path(__file__).resolve().parent
ARGS = [a for a in sys.argv[1:] if a != "--png"]
PNG = "--png" in sys.argv
RESULTS = Path(ARGS[0]) if ARGS else DOCS / "results" / "a100-2026-11-12"
SLO_MS = 300

# Categorical slots 1-3 (validated all-pairs for CVD on the GitHub light / dark surfaces), plus chart chrome.
THEMES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  series=("#2a78d6", "#1baf7a", "#eb6834")),
    "dark": dict(surface="#0d1117", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 series=("#3987e5", "#199e70", "#d95926")),
}


def load():
    """(label, [(streams, finalize p95 ms, within SLO)]) per system; color follows the system, not its rank."""
    nemo = [json.loads(p.read_text()) for p in sorted((RESULTS / "nemo_vs_lite" / "sweep").glob("nemo_open_c*.json"))
            if not p.name.endswith("summary.json")]
    eng = [json.loads(p.read_text()) for p in (RESULTS / "capacity_engine").glob("engine_open_c*_p20.json")]
    ws = json.loads((RESULTS / "ws_bench" / "p20" / "results.json").read_text())
    pts = lambda rows, n, ok: sorted((r[n], r["finalize_ms"]["95"], r[ok]) for r in rows)  # noqa: E731
    return [
        ("nemotron-asr-engine · Python API", pts(eng, "target_concurrency", "slo_ok")),
        ("nemotron-asr-engine · WebSocket server", pts(ws, "level", "slo_ok")),
        ("NeMo cache-aware streaming", pts(nemo, "target_concurrency", "slo_ok")),
    ]


def draw(theme: str, data) -> None:
    t = THEMES[theme]
    plt.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans",
                         "font.size": 12})
    fig, ax = plt.subplots(figsize=(8.4, 4.9), dpi=100)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    # SLO band and line: everything below it is "fast enough"
    ax.axhline(SLO_MS, color=t["ink2"], lw=1.2, ls=(0, (5, 4)), zorder=1)
    ax.text(70, SLO_MS * 1.12, "SLO: finalize p95 ≤ 300 ms", color=t["ink2"], fontsize=11, va="bottom")

    for (label, rows), color in zip(data, t["series"]):
        x = [r[0] for r in rows]
        y = [r[1] for r in rows]
        ax.plot(x, y, color=color, lw=2, zorder=3, solid_capstyle="round")
        for xi, yi, ok in rows:  # filled = within SLO, hollow = missed (shape, not only color, carries pass/fail)
            ax.plot(xi, yi, "o", ms=8.5, mec=color, mew=2, mfc=color if ok else t["surface"], zorder=4)

    # direct labels at each system's capacity: the highest load still within the SLO
    callouts = {  # label -> (short name, offset in points, horizontal alignment)
        "NeMo cache-aware streaming": ("NeMo: 192 streams", (0, 14), "center"),
        "nemotron-asr-engine · WebSocket server": ("WebSocket: 256 streams", (-12, 4), "right"),
        "nemotron-asr-engine · Python API": ("Python API: 384 streams", (12, -2), "left"),
    }
    for label, rows in data:
        best = max((r for r in rows if r[2]), key=lambda r: r[0])
        text, off, ha = callouts[label]
        ax.annotate(text, (best[0], best[1]), xytext=off, textcoords="offset points", ha=ha, va="center",
                    fontsize=11, fontweight="bold", color=t["ink"])

    ax.set_yscale("log")
    ax.set_ylim(15, 40000)
    ax.set_xlim(40, 560)
    ax.set_xticks([64, 128, 192, 256, 384, 448, 512])
    ax.set_yticks([30, 100, 300, 1000, 3000, 10000])
    ax.set_yticklabels(["30 ms", "100 ms", "300 ms", "1 s", "3 s", "10 s"])
    ax.minorticks_off()
    ax.set_xlabel("Concurrent real-time streams", color=t["ink2"], labelpad=8)
    ax.set_ylabel("Final transcript latency, p95", color=t["ink2"], labelpad=8)
    ax.grid(axis="y", color=t["grid"], lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=11)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(t["ink2"])

    fig.text(0.075, 0.955, "Concurrent live streams on one A100 40 GB", color=t["ink"],
             fontsize=14, fontweight="bold", ha="left", va="top")
    fig.text(0.075, 0.905, "320 ms chunks, 20 ms packets in real time. Filled point = within the SLO, hollow = missed.",
             color=t["ink2"], fontsize=11, ha="left", va="top")
    handles = [Line2D([], [], color=c, lw=2, marker="o", ms=7, mfc=c, label=lbl) for (lbl, _), c in zip(data, t["series"])]
    leg = ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10.5, handlelength=1.8,
                    bbox_to_anchor=(0.0, 1.0))
    for txt in leg.get_texts():
        txt.set_color(t["ink"])

    fig.subplots_adjust(left=0.12, right=0.97, top=0.84, bottom=0.13)
    out = DOCS / "assets" / f"capacity-{theme}.svg"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, facecolor=t["surface"], metadata={"Date": None})
    # text stays text in the SVG; let the viewer's UI font render it (DejaVu Sans only sets the layout metrics)
    svg = out.read_text().replace("font-family: 'DejaVu Sans'", "font-family:-apple-system,'Segoe UI',Helvetica,Arial,'DejaVu Sans',sans-serif")
    out.write_text(svg)
    if PNG:  # quick preview, not committed
        fig.savefig(out.with_suffix(".png"), facecolor=t["surface"], dpi=110)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    data = load()
    for theme in THEMES:
        draw(theme, data)
