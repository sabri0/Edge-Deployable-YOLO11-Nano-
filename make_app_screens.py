#!/usr/bin/env python3
"""Render an 'application screen' figure for the manuscript from REAL pipeline output
of the same clip shown in the live app (subject 11, exercise 06, cam4), styled like
the dashboard. Produces figures/fig_app_screen.png."""
import os, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from ultralytics import YOLO
from slr_core import savgol, parse_uco_p2d, bland_altman
from uco_slr_pipeline import process_video, summarise_session, angles_from_uco_gt

UCO = "datasets/clips_mp4"
SUBJ, EX, CAM, SIDE = "11", "06", "cam4", "r"   # mirrors the live-app demo clip
vid = os.path.join(UCO, SUBJ, EX, f"{CAM}.mp4")
dev = "0" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11n-pose.pt"); model.to("cuda:0" if dev == "0" else "cpu")

series = process_video(model, vid, side=SIDE, device=dev)
summ = summarise_session(series, 120.0, 160.0)
gt = angles_from_uco_gt(parse_uco_p2d(os.path.splitext(vid)[0] + "_p2d.txt", 3), SIDE)
n = min(len(gt), len(series["knee_raw"]))
ba = bland_altman(series["knee_raw"][:n], gt[:n])

BG, PANEL, LINE, INK, MUT = "#0d1117", "#161b22", "#30363d", "#e6edf3", "#8b949e"
GOOD, ACC = "#3fb950", "#2f81f7"
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(9, 8.0), dpi=130)
fig.patch.set_facecolor(BG)

# header
hax = fig.add_axes([0.06, 0.90, 0.91, 0.08]); hax.axis("off")
hax.text(0, 0.75, f"clip {SUBJ}/{EX}/{CAM}.mp4  ·  model = yolo11n-pose.pt  ·  "
         f"device = cuda:0  ·  resolved side = {SIDE}", color=MUT, fontsize=10,
         va="center")
hax.text(0, 0.15, "Test-video application — full SLR pipeline on the GPU",
         color=INK, fontsize=13, fontweight="bold", va="center")

# KPI tiles (two rows of four)
kpis = [("FRAMES / TRACKED", f"{summ['n_frames']} / {summ['n_tracked']}", ACC),
        ("SPEED (GPU)", f"{summ['fps']:.1f} fps", GOOD),
        ("LATENCY", f"{summ['latency_ms']:.2f} ms", GOOD),
        ("JITTER σ", f"{summ['sigma_deg']:.2f}°", GOOD),
        ("RMSSD", f"{summ['rmssd_deg']:.2f}°", GOOD),
        ("PEAK HIP FLEXION", f"{summ['peak_hip_flexion_deg']:.0f}°", GOOD),
        ("FRAME CONFORMITY", f"{summ['frame_conformity_pct']:.0f}%", GOOD),
        ("BLAND–ALTMAN n", f"{ba['n']}", ACC)]
for i, (lab, val, col) in enumerate(kpis):
    ax = fig.add_axes([0.06 + (i % 4) * 0.235, 0.785 - (i // 4) * 0.090, 0.21, 0.075])
    ax.axis("off"); ax.set_facecolor(PANEL)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=PANEL,
                               edgecolor=LINE, lw=1, zorder=0))
    ax.text(0.07, 0.70, lab, color=MUT, fontsize=7.5, va="center")
    ax.text(0.07, 0.30, val, color=col, fontsize=15, fontweight="bold", va="center")

# Bland-Altman banner
bax = fig.add_axes([0.06, 0.595, 0.91, 0.05]); bax.axis("off")
bax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=bax.transAxes, facecolor="#0f1c12",
                            edgecolor="#1d572f", lw=1))
bax.text(0.015, 0.5, "Bland–Altman vs OptiTrack GT (knee angle):", color=GOOD,
         fontsize=10, fontweight="bold", va="center")
bax.text(0.43, 0.5, f"bias {ba['bias']:+.2f}°,  95% LoA [{ba['loa_lower']:.1f}, "
         f"{ba['loa_upper']:.1f}]°,  MAPE {ba['mape_pct']:.1f}%,  n={ba['n']}",
         color=INK, fontsize=10, va="center")

# angle-over-time plot
pax = fig.add_axes([0.09, 0.08, 0.88, 0.45]); pax.set_facecolor(BG)
hip = series["hip_raw"]; knee = series["knee_raw"]
sm = savgol(hip) if hip.size else hip
pax.plot(np.arange(hip.size), hip, color=MUT, lw=1, alpha=.6, label="hip raw")
pax.plot(np.arange(hip.size), sm, color=GOOD, lw=2, label="hip smoothed")
pax.plot(np.arange(knee.size), knee, color=ACC, lw=1, alpha=.85, label="knee raw")
for s in pax.spines.values():
    s.set_color(LINE)
pax.tick_params(colors=MUT); pax.set_xlabel("tracked frame", color=MUT)
pax.set_ylabel("angle (deg)", color=MUT)
pax.legend(facecolor=PANEL, edgecolor=LINE, labelcolor=INK, fontsize=9)
fig.savefig("figures/fig_app_screen.png", facecolor=BG)
plt.close(fig)
print(f"clip {SUBJ}/{EX}/{CAM}: frames={summ['n_frames']} fps={summ['fps']:.1f} "
      f"sigma={summ['sigma_deg']} rmssd={summ['rmssd_deg']} "
      f"BA bias={ba['bias']:+.2f} LoA=[{ba['loa_lower']:.1f},{ba['loa_upper']:.1f}] "
      f"MAPE={ba['mape_pct']:.1f} n={ba['n']}")
print(">> wrote figures/fig_app_screen.png")
