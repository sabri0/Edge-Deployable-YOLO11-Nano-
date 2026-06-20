#!/usr/bin/env python3
"""
MediaPipe (BlazePose) baseline on the UCO SLR clips, scored exactly like the
YOLO pipeline: same knee-angle definition, same per-session metrics, same
Bland-Altman against the OptiTrack 2D ground truth. Runs on the held-out test
subjects so it is directly comparable to stock vs fine-tuned YOLO11n.

  python mediapipe_slr.py --subjects 0 1 2 3 4 5 6 7 8 9 --out uco_report_mediapipe_test.csv
"""
import argparse, os, time, csv
import numpy as np
import cv2
import mediapipe as mp

from slr_core import (joint_angle, parse_uco_p2d, bland_altman)
from uco_slr_pipeline import (summarise_session, angles_from_uco_gt,
                              load_uco_side_map)

# MediaPipe Pose (33 landmarks) indices
MP = {"l_shoulder": 11, "r_shoulder": 12, "l_hip": 23, "r_hip": 24,
      "l_knee": 25, "r_knee": 26, "l_ankle": 27, "r_ankle": 28}
UCO = "datasets/clips_mp4"


def process_video_mp(pose, vid, side):
    cap = cv2.VideoCapture(vid)
    knee, hip, lat = [], [], []
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        t0 = time.perf_counter()
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        dt = (time.perf_counter() - t0) * 1000.0
        if fi >= 10:
            lat.append(dt)
        fi += 1
        lm = res.pose_landmarks
        if lm is None:
            continue
        def px(name):
            p = lm.landmark[MP[f"{side}_{name}"]]
            return np.array([p.x * W, p.y * H]), p.visibility
        (sh, vs), (hp, vh), (kn, vk), (an, va) = (px("shoulder"), px("hip"),
                                                  px("knee"), px("ankle"))
        if min(vh, vk, va) < 0.5:
            continue
        knee.append(joint_angle(hp, kn, an))
        hip.append(joint_angle(sh, hp, kn))
    cap.release()
    return {"knee_raw": np.asarray(knee, float), "hip_raw": np.asarray(hip, float),
            "latency_ms": float(np.mean(lat)) if lat else float("nan"),
            "fps": (1000.0 / np.mean(lat)) if lat else float("nan"), "n_frames": fi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+",
                    default=[str(i) for i in range(10)])
    ap.add_argument("--exercises", nargs="+", default=["03", "07"])
    ap.add_argument("--out", default="uco_report_mediapipe_test.csv")
    args = ap.parse_args()

    per_seq, per_ex = load_uco_side_map(UCO)
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
    rows, ba_m, ba_r = [], [], []
    for s in args.subjects:
        for ex in args.exercises:
            vid = os.path.join(UCO, str(s), ex, "cam2.mp4")
            if not os.path.exists(vid):
                continue
            side = per_seq.get((str(s), ex)) or per_ex.get(ex) or "r"
            series = process_video_mp(pose, vid, side)
            summ = summarise_session(series, 120.0, 160.0)
            if summ is None:
                print(f"  [skip] subj {s} ex {ex}: no track"); continue
            summ.update(subject=s, exercise=ex, side=side, method="mediapipe")
            gtp = os.path.splitext(vid)[0] + "_p2d.txt"
            if os.path.exists(gtp):
                gt_knee = angles_from_uco_gt(parse_uco_p2d(gtp, 3), side)
                n = min(len(gt_knee), len(series["knee_raw"]))
                if n > 5:
                    ba_m.extend(series["knee_raw"][:n]); ba_r.extend(gt_knee[:n])
            rows.append(summ)
            print(f"  [ok] subj {s} ex {ex} side={side}: sigma={summ['sigma_deg']} "
                  f"deg, {summ['fps']} fps, conform={summ['frame_conformity_pct']}%")
    pose.close()
    summary = {"sessions": len(rows)}
    if rows:
        summary["mean_fps"] = round(float(np.mean([r["fps"] for r in rows])), 1)
        summary["mean_latency_ms"] = round(
            float(np.mean([r["latency_ms"] for r in rows])), 1)
        summary["mean_sigma_deg"] = round(
            float(np.mean([r["sigma_deg"] for r in rows])), 2)
    if ba_m:
        ba = bland_altman(ba_m, ba_r)
        m, r = np.asarray(ba_m, float), np.asarray(ba_r, float)
        np.savez("pairs_mediapipe.npz", method=m, ref=r)
        summary.update(bias=round(ba["bias"], 2),
                       loa_lower=round(ba["loa_lower"], 1),
                       loa_upper=round(ba["loa_upper"], 1),
                       mape_pct=round(ba["mape_pct"], 1), n=ba["n"],
                       tracking_acc_pct=round(float(np.mean(np.abs(m - r) < 5) * 100), 1),
                       pearson_r=round(float(np.corrcoef(m, r)[0, 1]), 3),
                       pairs_file="pairs_mediapipe.npz")
        print(f"\nMediaPipe Bland-Altman vs GT knee: bias={ba['bias']:.2f}, "
              f"LoA=[{ba['loa_lower']:.1f},{ba['loa_upper']:.1f}], "
              f"MAPE={ba['mape_pct']:.1f}%, n={ba['n']}, "
              f"acc<5deg={summary['tracking_acc_pct']}%")
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f">> wrote {len(rows)} rows -> {args.out}")
    # merge into metrics_eval.json
    import json
    path = "metrics_eval.json"
    prev = json.load(open(path)) if os.path.exists(path) else {}
    prev["mediapipe_test"] = summary
    json.dump(prev, open(path, "w"), indent=2)
    print(">> updated metrics_eval.json[mediapipe_test]")


if __name__ == "__main__":
    main()
