"""Regenerate all figures from logs/. Reads only logs, never live state, so any
figure can be rebuilt from a run that has already finished.

    python experiments/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.metrics import read_jsonl

LOGS = Path("logs")
HORIZONS = (1, 4, 16)

# Validated categorical slots (see the data-viz palette reference). Contrast against
# the surface is below 3:1 for slots 3-4, so every bar carries a direct value label.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dedcd6"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
POS, NEG = "#e34948", "#2a78d6"          # diverging pair: hurts / helps


def style(ax, title=None, ylabel=None, xlabel=None):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=1)
    ax.grid(axis="y", color=GRID, linewidth=1, alpha=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)


def save(fig, name):
    LOGS.mkdir(exist_ok=True)
    path = LOGS / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


def _load(name):
    p = LOGS / name
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------ E1 figures


def fig_e1_bars():
    base, mesh = _load("baselines_summary.json"), _load("exp01_summary.json")
    if not base or not mesh:
        return
    models = [("mesh (local learning)", mesh["mesh"]["test"]),
              ("copy-last", base["copy_last"]),
              ("linear filter", base["linear_ar"]),
              ("GRU (BPTT)", base.get("gru_matched") or base["gru"])]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(HORIZONS))
    w = 0.2
    for k, (name, vals) in enumerate(models):
        ys = [vals[f"mse_t{t}"] for t in HORIZONS]
        bars = ax.bar(x + (k - 1.5) * w, ys, w * 0.88, label=name,
                      color=SERIES[k], linewidth=0)
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.4f}", ha="center",
                    va="bottom", fontsize=6.5, color=INK2, rotation=90)
    ax.set_xticks(x, [f"{t}-step" for t in HORIZONS])
    ax.set_ylim(0, max(v[f"mse_t16"] for _, v in models) * 1.25)
    style(ax, "E1  Held-out prediction error by horizon", "frame MSE (lower is better)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, ncols=2)
    save(fig, "fig_e1_horizons.png")


def fig_e1_curve():
    path = LOGS / "exp01_learning.jsonl"
    base = _load("baselines_summary.json")
    if not path.exists():
        return
    rows = [r for r in read_jsonl(path) if r.get("kind") == "mesh_train"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot([r["t"] for r in rows], [r["mse_t1"] for r in rows],
            color=SERIES[0], linewidth=2, label="mesh, 1-step (train, windowed)")
    if base:
        for k, (name, key) in enumerate([("copy-last", "copy_last"),
                                         ("linear filter", "linear_ar")]):
            y = base[key]["mse_t1"]
            ax.axhline(y, color=SERIES[k + 1], linewidth=2, linestyle="--", alpha=0.9)
            ax.text(rows[-1]["t"], y, f"  {name}", va="center", fontsize=8,
                    color=INK2)
    ax.set_yscale("log")
    style(ax, "E1  Learning curve", "frame MSE (log)", "training tick")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    save(fig, "fig_e1_curve.png")


# ------------------------------------------------------------------ E2 figure


def fig_e2():
    res = _load("exp02_summary.json")
    if not res:
        return
    names = [("trained mesh", "trained_mesh"), ("untrained mesh", "untrained_mesh"),
             ("copy-last (no memory)", "copy_last_reference")]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(names))
    w = 0.34
    for k, cond in enumerate(("preserved", "perturbed")):
        ys = [res[key][cond]["mean"] for _, key in names]
        es = [res[key][cond]["sem"] or 0 for _, key in names]
        bars = ax.bar(x + (k - 0.5) * w, ys, w * 0.88, yerr=es, capsize=3,
                      label=f"{cond} trajectory", color=SERIES[k], linewidth=0,
                      error_kw=dict(ecolor=INK2, elinewidth=1))
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.4f}", ha="center",
                    va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x, [n for n, _ in names])
    style(ax, None, "local frame MSE, 4-step prediction")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    ax.set_title("E2  Prediction error after an object re-emerges from occlusion",
                 color=INK, fontsize=12, loc="left", pad=26)
    ratio = res["trained_mesh"].get("ratio")
    if ratio:
        ax.text(0, 1.02,
                f"a maintained representation would raise the orange bar; observed "
                f"ratio {ratio:.2f} (p={res['trained_mesh']['p_permutation']:.2f})",
                transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
    save(fig, "fig_e2_occlusion.png")


# ------------------------------------------------------------------ E3 figure


def fig_e3():
    res = _load("exp03_summary.json")
    if not res or "full" not in res:
        return
    full = res["full"]["mse_t16"]
    rows = sorted(((r["mse_t16"] / full, name) for name, r in res.items()
                   if name != "full"), reverse=True)
    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.42 * len(rows) + 1.6)))
    y = np.arange(len(rows))
    vals = [v - 1 for v, _ in rows]
    colors = [POS if v > 0 else NEG for v in vals]
    ax.barh(y, vals, 0.62, color=colors, linewidth=0)
    for yi, (v, _) in zip(y, rows):
        off = 0.004 if v >= 1 else -0.004
        ax.text(v - 1 + off, yi, f"{v:.2f}x", va="center",
                ha="left" if v >= 1 else "right", fontsize=8, color=INK2)
    ax.set_yticks(y, [n.replace("_", " ") for _, n in rows])
    ax.axvline(0, color=INK2, linewidth=1)
    lim = max(abs(min(vals)), abs(max(vals))) * 1.3 + 0.02
    ax.set_xlim(-lim, lim)

    # Ticks are labelled as ratios, so only place them where a ratio is meaningful.
    candidates = [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5]
    ticks = [c - 1 for c in candidates if abs(c - 1) <= lim]
    ax.set_xticks(ticks, [f"{t+1:g}x" for t in ticks])

    style(ax, None, None, "16-step error relative to the full model")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_title("E3  Effect of removing one ingredient",
                 color=INK, fontsize=12, loc="left", pad=26)
    ax.text(0, 1.015, "right = the full model was better, so the ingredient was doing work",
            transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
    save(fig, "fig_e3_ablations.png")


# ------------------------------------------------------------------ E4 figure


def fig_e4():
    path = LOGS / "exp04_silence.jsonl"
    if not path.exists():
        return
    rows = [r for r in read_jsonl(path) if r.get("kind") in ("driven", "silent")]
    if not rows:
        return
    driven = [r for r in rows if r["kind"] == "driven"]
    cut = len(driven)
    idx = np.arange(len(rows))

    # two measures on different scales -> two stacked panels, never a second y-axis
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True)
    panels = [("surprise", "mean prediction error", SERIES[0]),
              ("h_std", "state dispersion", SERIES[2])]
    for ax, (key, label, color) in zip(axes, panels):
        ax.plot(idx, [r[key] for r in rows], color=color, linewidth=2)
        ax.axvline(cut, color=INK2, linewidth=1, linestyle="--")
        style(ax, None, label)
    axes[0].set_title("E4  What the mesh does when sensory input is cut",
                      color=INK, fontsize=12, loc="left", pad=12)
    axes[0].text(cut, axes[0].get_ylim()[1], "  input cut", fontsize=9,
                 color=INK2, va="top")
    axes[1].set_xlabel("probe index (every 10 ticks)", color=INK2, fontsize=10)
    save(fig, "fig_e4_silence.png")


if __name__ == "__main__":
    fig_e1_bars()
    fig_e1_curve()
    fig_e2()
    fig_e3()
    fig_e4()
