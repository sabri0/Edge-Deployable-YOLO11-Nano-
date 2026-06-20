#!/usr/bin/env python3
"""
Generate the article figures + a consolidated results table from the saved
evaluation artefacts (metrics_eval.json, pairs_*.npz, uco_report_*_test.csv).

Outputs -> figures/:
  fig_bland_altman.png    - Bland-Altman (model knee vs OptiTrack GT) per method
  fig_jitter_box.png      - per-session signal variance (jitter sigma) per method
  fig_performance.png     - bias / MAPE / tracking-accuracy / fps bars
  results_table.csv       - every headline value, one row per method
"""
import os, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "figures"
os.makedirs(FIG, exist_ok=True)
METHODS = [("mediapipe_test", "MediaPipe", "pairs_mediapipe.npz",
            "uco_report_mediapipe_test.csv", "#d29922"),
           ("stock_n_test", "YOLO11n (stock)", "pairs_stock_n.npz",
            "uco_report_n_test.csv", "#2f81f7"),
           ("finetuned_n_test", "YOLO11n (fine-tuned)", "pairs_finetuned_n.npz",
            "uco_report_ft_test.csv", "#3fb950")]

metrics = json.load(open("metrics_eval.json")) if os.path.exists("metrics_eval.json") else {}
present = [(k, lab, npz, csvf, c) for k, lab, npz, csvf, c in METHODS
          if k in metrics and os.path.exists(npz)]


def bland_altman_fig():
    n = len(present)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2), dpi=120)
    if n == 1:
        axes = [axes]
    for ax, (k, lab, npz, _csv, col) in zip(axes, present):
        d = np.load(npz); m, r = d["method"], d["ref"]
        mean = (m + r) / 2; diff = m - r
        bias = float(np.mean(diff)); sd = float(np.std(diff, ddof=1))
        lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
        ax.scatter(mean, diff, s=4, alpha=.15, color=col, edgecolors="none")
        ax.axhline(bias, color="k", lw=1.5, label=f"bias {bias:+.1f}°")
        ax.axhline(lo, color="r", ls="--", lw=1, label=f"95% LoA [{lo:.0f}, {hi:.0f}]")
        ax.axhline(hi, color="r", ls="--", lw=1)
        ax.set_title(f"{lab}  (n={m.size})", fontsize=11)
        ax.set_xlabel("mean of model & GT knee angle (°)")
        ax.set_ylabel("model − GT (°)")
        ax.set_ylim(-90, 90); ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=.2)
    fig.suptitle("Bland–Altman vs OptiTrack ground truth (held-out subjects 0–9)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_bland_altman.png"); plt.close(fig)
    print("wrote", f"{FIG}/fig_bland_altman.png")


def jitter_box_fig():
    data, labels, colors = [], [], []
    for k, lab, _npz, csvf, col in present:
        if os.path.exists(csvf):
            rows = list(csv.DictReader(open(csvf)))
            s = [float(r["sigma_deg"]) for r in rows if r.get("sigma_deg")]
            if s:
                data.append(s); labels.append(lab); colors.append(col)
    if not data:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=120)
    try:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
    except TypeError:  # matplotlib < 3.9
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(.55)
    ax.set_ylabel("per-session signal variance σ (°)")
    ax.set_title("Signal jitter by method (lower = steadier)")
    ax.grid(alpha=.2, axis="y")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_jitter_box.png"); plt.close(fig)
    print("wrote", f"{FIG}/fig_jitter_box.png")


def performance_fig():
    labs = [lab for _k, lab, *_ in present]
    cols = [c for *_a, c in present]
    def val(k, key):
        return metrics[k].get(key, np.nan)
    panels = [("|bias| vs GT (°)", [abs(val(k, "bias")) for k, *_ in present]),
              ("MAPE (%)", [val(k, "mape_pct") for k, *_ in present]),
              ("tracking acc <5° (%)", [val(k, "tracking_acc_pct") for k, *_ in present]),
              ("speed (fps)", [val(k, "mean_fps") for k, *_ in present])]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), dpi=120)
    for ax, (title, vals) in zip(axes, panels):
        ax.bar(labs, vals, color=cols, alpha=.8)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.grid(alpha=.2, axis="y")
    fig.suptitle("Method comparison on held-out test subjects 0–9", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_performance.png"); plt.close(fig)
    print("wrote", f"{FIG}/fig_performance.png")


def results_table():
    cols = ["method", "sessions", "bias", "loa_lower", "loa_upper", "mape_pct",
            "tracking_acc_pct", "pearson_r", "mean_sigma_deg", "mean_fps",
            "mean_latency_ms", "n"]
    with open(f"{FIG}/results_table.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for k, lab, *_ in present:
            mm = metrics[k]
            w.writerow([lab] + [mm.get(c, "") for c in cols[1:]])
    print("wrote", f"{FIG}/results_table.csv")


if __name__ == "__main__":
    if not present:
        print("no evaluation artefacts found yet (run eval_testset.py + "
              "mediapipe_slr.py first)")
    else:
        bland_altman_fig(); jitter_box_fig(); performance_fig(); results_table()
        print("done -> figures/")
