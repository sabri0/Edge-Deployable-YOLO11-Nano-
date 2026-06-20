#!/usr/bin/env python3
"""
uco_slr_pipeline.py
====================
Automated Straight Leg Raise (SLR) assessment with YOLO11-pose, validated against
the REAL public dataset:

    UCO Physical Rehabilitation dataset
    Aguilar-Ortega R, Berral-Soler R, Jimenez-Velasco I, et al.
    Sensors 2023; 23(21):8862.  DOI: 10.3390/s23218862
    Repo:   https://github.com/AVAuco/ucophyrehab
    Access: email inforeha@uco.es (name, affiliation, research purpose)

Why this dataset:
  * It is the dataset the reviewed manuscript actually cites.
  * It contains the supine Straight Leg Raise (exercises 03 / 07: "Lift the
    extended leg") and seated lower-limb exercises (01/05, 02/06).
  * It ships OptiTrack-derived 2D joint ground truth (hip/knee/ankle) in
    `camX_p2d.txt` / `dataset_2d.json` -- i.e. a marker-based motion-capture
    reference, which the manuscript wrongly described as unavailable.

What this script fixes relative to the manuscript:
  1. Angle definition. The manuscript's Eq.(1) computes the KNEE angle
     (hip-knee-ankle) but labels it "trunk-to-thigh". Here we compute BOTH,
     each correctly labelled, and use the shoulder keypoint (available from
     YOLO11-pose / COCO-17) for the true trunk-to-thigh (hip-flexion) angle.
  2. Reference standard. We validate against the dataset's own 2D ground truth.
  3. Honest reporting. Per-session aggregates are produced so downstream stats
     can cluster by session/subject instead of treating frames as independent.

Run modes
---------
  # A. Validate on the UCO dataset (after you obtain it):
  python uco_slr_pipeline.py --uco-root /path/to/ucophyrehab/data \
         --subjects 1 2 3 --exercises 03 07 --side r --model yolo11n-pose.pt

  # B. Run on any video (no dataset needed -- proves the pipeline end-to-end):
  python uco_slr_pipeline.py --video my_slr_clip.mp4 --side r

  # C. Live webcam:
  python uco_slr_pipeline.py --webcam 0 --side r

Dependencies:  pip install ultralytics scipy numpy opencv-python pandas
The first run downloads the model weight from Ultralytics (needs internet).
"""
from __future__ import annotations
import argparse, time, os, glob, csv
import numpy as np

from slr_core import (COCO, knee_flexion_angle, hip_flexion_angle, savgol,
                      bland_altman, signal_variance, conformity, parse_uco_p2d)

# ----------------------------------------------------------------------------- 
# Pose extraction
# -----------------------------------------------------------------------------
def load_model(weight="yolo11n-pose.pt", device=None):
    from ultralytics import YOLO
    model = YOLO(weight)
    if device is not None:
        # torch's Module.to() needs a valid device string; a bare GPU index
        # like "0" (accepted by ultralytics' predict) must become "cuda:0".
        torch_dev = f"cuda:{device}" if str(device).isdigit() else device
        model.to(torch_dev)
    return model


def keypoints_from_result(res):
    """Return (17,2) xy array for the most confident person, or None."""
    if res.keypoints is None or len(res.keypoints) == 0:
        return None
    xy = res.keypoints.xy.cpu().numpy()          # (n_persons, 17, 2)
    if xy.shape[0] == 0:
        return None
    # pick the person with the largest bounding box (closest / main subject)
    if res.boxes is not None and len(res.boxes) == xy.shape[0]:
        areas = (res.boxes.xywh[:, 2] * res.boxes.xywh[:, 3]).cpu().numpy()
        idx = int(np.argmax(areas))
    else:
        idx = 0
    return xy[idx]


def process_video(model, source, side="r", target_size=640, warmup=10, device=None):
    """Run YOLO11-pose over a video/stream. Returns dict of per-frame series
    plus mean inference latency (ms/frame)."""
    import cv2
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source: {source}")

    knee, hip, lat = [], [], []
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        res = model.predict(frame, imgsz=target_size, device=device, verbose=False)[0]
        dt = (time.perf_counter() - t0) * 1000.0
        if fi >= warmup:
            lat.append(dt)
        kp = keypoints_from_result(res)
        if kp is not None and np.all(kp[[COCO[f"{side}_hip"], COCO[f"{side}_knee"],
                                        COCO[f"{side}_ankle"]]] > 0):
            knee.append(knee_flexion_angle(kp, side))
            hip.append(hip_flexion_angle(kp, side))
        fi += 1
    cap.release()
    return {
        "knee_raw": np.asarray(knee, float),
        "hip_raw": np.asarray(hip, float),
        "latency_ms": float(np.mean(lat)) if lat else float("nan"),
        "fps": (1000.0 / np.mean(lat)) if lat else float("nan"),
        "n_frames": fi,
    }


# ----------------------------------------------------------------------------- 
# Per-session report (clusterable -> avoids the manuscript's pseudoreplication)
# -----------------------------------------------------------------------------
def summarise_session(series, hip_lo, hip_hi):
    raw = series["hip_raw"]
    if raw.size == 0:
        return None
    sm = savgol(raw)
    var = signal_variance(raw)                       # jitter on RAW signal
    # Peak SLR elevation = SMALLEST trunk-to-thigh angle (leg lifted toward vertical)
    con = conformity(sm, hip_lo, hip_hi, peak_fn=np.min)  # conformity on SMOOTHED
    return {
        "n_frames": series["n_frames"],
        "n_tracked": int(raw.size),
        "latency_ms": round(series["latency_ms"], 2),
        "fps": round(series["fps"], 1),
        "sigma_deg": round(var["sigma"], 2),
        "cv_pct": round(var["cv_pct"], 2),
        "rmssd_deg": round(var["rmssd"], 2),
        "peak_hip_flexion_deg": round(180 - con["peak_angle"], 1),
        "frame_conformity_pct": round(con["frame_conformity_pct"], 1),
        "session_conformant": con["session_conformant"],
    }


# ----------------------------------------------------------------------------- 
# UCO ground-truth validation
# -----------------------------------------------------------------------------
def angles_from_uco_gt(p2d, side="r"):
    """UCO lower-body p2d files carry 3 joints (hip,knee,ankle). We can compute
    the KNEE angle directly; the hip/trunk-thigh angle needs a trunk reference,
    which the 3-joint GT lacks -- documenting exactly why 3-joint tracking cannot
    yield a true trunk-to-thigh angle (the manuscript's core measurement error)."""
    hip, knee, ankle = p2d[:, 0], p2d[:, 1], p2d[:, 2]
    knee_ang = np.array([
        __import__("slr_core").joint_angle(hip[i], knee[i], ankle[i])
        for i in range(len(p2d))
    ])
    return knee_ang


def load_uco_side_map(uco_root):
    """Read UCO `dataset_2d.json` and return the leg each sequence is performed
    with, so the model measures the SAME limb the OptiTrack ground truth tracks.

    The dataset encodes side by the exercise protocol (the lower-limb exercises
    come in left/right pairs: 01/05, 02/06, 03/07, 04/08), confirmed by the JSON.
    Returns (per_seq, per_ex):
      per_seq[(folder, exercise)] = 'l' | 'r'   (authoritative, when present)
      per_ex[exercise]            = 'l' | 'r'   (protocol fallback across subjects)
    Returns ({}, {}) if the JSON cannot be found (caller then keeps the CLI side).
    """
    import json
    cands = [os.path.join(uco_root, "dataset_2d.json"),
             os.path.join(os.path.dirname(os.path.normpath(uco_root)),
                          "dataset_2d.json")]
    path = next((p for p in cands if os.path.exists(p)), None)
    if path is None:
        print("  [warn] dataset_2d.json not found; cannot auto-resolve side")
        return {}, {}
    data = json.load(open(path))["data"]
    s2c = {"left": "l", "right": "r"}
    per_seq, per_ex = {}, {}
    for e in data:
        side = s2c.get(e.get("side"))
        if side is None:
            continue
        per_seq[(str(e["folder"]), str(e["exercise"]))] = side
        per_ex.setdefault(str(e["exercise"]), side)
    return per_seq, per_ex


def validate_against_uco(model, uco_root, subjects, exercises, side="r",
                         hip_lo=120.0, hip_hi=160.0, device=None):
    rows, ba_method, ba_ref = [], [], []
    per_seq, per_ex = (load_uco_side_map(uco_root) if side == "auto"
                       else ({}, {}))
    for s in subjects:
        for ex in exercises:
            # Resolve which leg to measure for this (subject, exercise).
            if side == "auto":
                this_side = per_seq.get((str(s), str(ex))) or per_ex.get(str(ex)) or "r"
            else:
                this_side = side
            folder = os.path.join(uco_root, str(s), ex)
            # UCO ships clips as .mp4 (clips_mp4); keep .avi as a fallback.
            vids = sorted(glob.glob(os.path.join(folder, "cam*.mp4")) or
                          glob.glob(os.path.join(folder, "cam*.avi")))
            if not vids:
                print(f"  [skip] no videos in {folder}")
                continue
            # prefer frontal-ish camera 2 (most accurate per UCO study)
            vid = next((v for v in vids if "cam2" in v), vids[0])
            gt_path = os.path.splitext(vid)[0] + "_p2d.txt"
            series = process_video(model, vid, side=this_side, device=device)
            summ = summarise_session(series, hip_lo, hip_hi)
            if summ is None:
                continue
            summ.update(subject=s, exercise=ex, camera=os.path.basename(vid))
            # Bland-Altman: model KNEE angle vs GT KNEE angle (both 3-joint -> fair)
            if os.path.exists(gt_path):
                gt = parse_uco_p2d(gt_path, n_joints=3)
                gt_knee = angles_from_uco_gt(gt, this_side)
                n = min(len(gt_knee), len(series["knee_raw"]))
                if n > 5:
                    ba_method.extend(series["knee_raw"][:n])
                    ba_ref.extend(gt_knee[:n])
            summ.update(side=this_side)
            rows.append(summ)
            print(f"  [ok] subj {s} ex {ex} side={this_side}: sigma={summ['sigma_deg']} deg, "
                  f"{summ['fps']} fps, conform={summ['frame_conformity_pct']}%")
    ba = bland_altman(ba_method, ba_ref) if ba_method else None
    # also expose the paired arrays so callers can save/plot Bland-Altman.
    ba_pairs = (np.asarray(ba_method, float), np.asarray(ba_ref, float))
    return rows, ba, ba_pairs


# ----------------------------------------------------------------------------- 
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11n-pose.pt",
                    help="yolo11n-pose.pt (proposed) | yolo11m-pose.pt (reference)")
    ap.add_argument("--side", default="r", choices=["l", "r", "auto"],
                    help="'l'/'r' to force a leg; 'auto' resolves the correct "
                         "leg per (subject,exercise) from UCO dataset_2d.json "
                         "(03=left, 07=right SLR).")
    ap.add_argument("--uco-root", help="path to ucophyrehab/data")
    ap.add_argument("--subjects", nargs="+", default=["1", "2", "3"])
    ap.add_argument("--exercises", nargs="+", default=["03", "07"],
                    help="03/07=supine SLR; 01/05=seated lower-limb")
    ap.add_argument("--video", help="single video file (fallback mode)")
    ap.add_argument("--webcam", type=int, help="webcam index, e.g. 0")
    ap.add_argument("--out", default="slr_session_report.csv")
    ap.add_argument("--hip-lo", type=float, default=120.0)
    ap.add_argument("--hip-hi", type=float, default=160.0)
    ap.add_argument("--device", default=None,
                    help="inference device: '0'/'cuda' for GPU, 'cpu' for CPU. "
                         "Default: auto (GPU if available).")
    args = ap.parse_args()

    if not (args.uco_root or args.video or args.webcam is not None):
        ap.error("provide one of --uco-root, --video, or --webcam")

    # Resolve device: honour --device, else auto-pick GPU when CUDA is present.
    device = args.device
    if device is None:
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    print(f">> loading {args.model} on device={device}")
    model = load_model(args.model, device=device)

    if args.uco_root:
        rows, ba, _pairs = validate_against_uco(model, args.uco_root, args.subjects,
                                                args.exercises, args.side,
                                                args.hip_lo, args.hip_hi, device=device)
        if ba:
            print("\nBland-Altman (model KNEE angle vs UCO OptiTrack GT KNEE angle):")
            print(f"  bias={ba['bias']:.2f} deg, 95% LoA=[{ba['loa_lower']:.1f},"
                  f"{ba['loa_upper']:.1f}], MAPE={ba['mape_pct']:.1f}%, n={ba['n']}")
        _write_csv(rows, args.out)
    elif args.video or args.webcam is not None:
        src = args.video if args.video else args.webcam
        # 'auto' only resolves from the UCO metadata; for a lone clip default to 'r'.
        vid_side = "r" if args.side == "auto" else args.side
        series = process_video(model, src, side=vid_side, device=device)
        summ = summarise_session(series, args.hip_lo, args.hip_hi)
        print("\nSession summary:", summ)
        _write_csv([dict(summ, source=str(src))] if summ else [], args.out)


def _write_csv(rows, path):
    if not rows:
        print("no rows to write"); return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f">> wrote {len(rows)} session rows -> {path}")


if __name__ == "__main__":
    main()
