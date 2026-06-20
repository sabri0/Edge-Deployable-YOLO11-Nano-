#!/usr/bin/env python3
"""Produce a worked single-clip example for the manuscript: run the fine-tuned model
on a held-out test clip, compute the per-clip session report + Bland-Altman vs the
OptiTrack GT, and render one annotated frame (skeleton + knee & trunk-thigh angles)."""
import os, json, cv2, numpy as np, torch
from ultralytics import YOLO
from slr_core import (COCO, knee_flexion_angle, hip_flexion_angle, parse_uco_p2d,
                      bland_altman)
from uco_slr_pipeline import (process_video, summarise_session, angles_from_uco_gt,
                              load_uco_side_map)

UCO = "datasets/clips_mp4"
CAM = "cam2"
FTW = "runs/pose/ft_runs/yolo11n_uco_slr/weights/best.pt"
os.makedirs("figures", exist_ok=True)

per_seq, per_ex = load_uco_side_map(UCO)
dev = "0" if torch.cuda.is_available() else "cpu"
model = YOLO(FTW); model.to("cuda:0" if dev == "0" else "cpu")

# Scan all held-out test clips and pick the REPRESENTATIVE (median-bias) one,
# rather than cherry-picking, so the worked example is honest.
clips = []
for s in [str(i) for i in range(10)]:
    for ex in ("03", "07"):
        v = os.path.join(UCO, s, ex, f"{CAM}.mp4")
        g = os.path.splitext(v)[0] + "_p2d.txt"
        if not (os.path.exists(v) and os.path.exists(g)):
            continue
        sd = per_seq.get((s, ex)) or per_ex.get(ex) or "r"
        ser = process_video(model, v, side=sd, device=dev)
        sm = summarise_session(ser, 120.0, 160.0)
        if sm is None:
            continue
        gtk = angles_from_uco_gt(parse_uco_p2d(g, 3), sd)
        nn = min(len(gtk), len(ser["knee_raw"]))
        b = bland_altman(ser["knee_raw"][:nn], gtk[:nn])
        clips.append((s, ex, sd, v, ser, sm, b))
        print(f"  scan {s}/{ex} side={sd}: bias={b['bias']:+.2f}")

biases = sorted(c[6]["bias"] for c in clips)
median_bias = biases[len(biases) // 2]
SUBJ, EX, side, vid, series, summ, ba = min(
    clips, key=lambda c: abs(c[6]["bias"] - median_bias))
print(f">> representative (median-bias) clip = {SUBJ}/{EX} "
      f"(bias {ba['bias']:+.2f}, cohort median {median_bias:+.2f})")

# annotate a representative mid-session frame
cap = cv2.VideoCapture(vid)
mid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.55)
cap.set(cv2.CAP_PROP_POS_FRAMES, mid); ok, frame = cap.read(); cap.release()
res = model.predict(frame, device=dev, verbose=False)[0]
kp = res.keypoints.xy.cpu().numpy()[0]
ka, ha = knee_flexion_angle(kp, side), hip_flexion_angle(kp, side)
def pt(name): return tuple(int(v) for v in kp[COCO[f"{side}_{name}"]])
for a, b in [("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle")]:
    if min(pt(a)) > 0 and min(pt(b)) > 0:
        cv2.line(frame, pt(a), pt(b), (0, 220, 0), 4)
for nm in ("shoulder", "hip", "knee", "ankle"):
    if min(pt(nm)) > 0:
        cv2.circle(frame, pt(nm), 7, (0, 0, 255), -1)
cv2.rectangle(frame, (8, 8), (430, 78), (0, 0, 0), -1)
cv2.putText(frame, f"knee = {ka:.1f} deg", (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
cv2.putText(frame, f"trunk-thigh (hip) = {ha:.1f} deg", (16, 66),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
cv2.imwrite("figures/fig_app_example.png", frame)

example = {
    "clip": f"{SUBJ}/{EX}/{CAM}.mp4", "resolved_side": side, "model": "fine-tuned yolo11n",
    "n_frames": summ["n_frames"], "n_tracked": summ["n_tracked"],
    "fps": summ["fps"], "latency_ms": summ["latency_ms"],
    "sigma_deg": summ["sigma_deg"], "rmssd_deg": summ["rmssd_deg"],
    "peak_hip_flexion_deg": summ["peak_hip_flexion_deg"],
    "frame_conformity_pct": summ["frame_conformity_pct"],
    "ba_bias": round(ba["bias"], 2), "ba_loa": [round(ba["loa_lower"], 1),
                                                round(ba["loa_upper"], 1)],
    "ba_mape_pct": round(ba["mape_pct"], 1), "ba_n": ba["n"],
    "example_frame_knee_deg": round(float(ka), 1),
    "example_frame_trunkthigh_deg": round(float(ha), 1),
}
json.dump(example, open("app_example.json", "w"), indent=2)
print(json.dumps(example, indent=2))
print(">> wrote figures/fig_app_example.png + app_example.json")
