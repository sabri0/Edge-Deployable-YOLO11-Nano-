#!/usr/bin/env python3
"""
Three-way evaluation on the HELD-OUT test subjects (0-9), exercises 03/07,
side=auto, all scored identically against the OptiTrack 2D ground truth.

  * stock yolo11n-pose      -> uco_report_n_test.csv
  * fine-tuned yolo11n-pose -> uco_report_ft_test.csv
  (MediaPipe is run separately via mediapipe_slr.py once it is installed.)

Writes a metrics_eval.json with each method's pooled Bland-Altman + mean speed,
which the dashboard/article read.
"""
import os, json, csv
import numpy as np
from uco_slr_pipeline import load_model, validate_against_uco

TEST_SUBJECTS = [str(i) for i in range(10)]
EXERCISES = ["03", "07"]


def extra_metrics(method, ref):
    """Tracking accuracy (% frames |err|<5 deg) and Pearson r, for the article."""
    method, ref = np.asarray(method, float), np.asarray(ref, float)
    if method.size < 2:
        return {}
    acc = float(np.mean(np.abs(method - ref) < 5.0) * 100)
    r = float(np.corrcoef(method, ref)[0, 1])
    return {"tracking_acc_pct": round(acc, 1), "pearson_r": round(r, 3)}


def run(model_weight, out_csv, tag, device="0"):
    model = load_model(model_weight, device=device)
    rows, ba, pairs = validate_against_uco(model, "datasets/clips_mp4",
                                           TEST_SUBJECTS, EXERCISES,
                                           side="auto", device=device)
    keys = sorted({k for r in rows for k in r})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    # save paired arrays for reproducible Bland-Altman plots
    np.savez(f"pairs_{tag}.npz", method=pairs[0], ref=pairs[1])
    fps = np.mean([r["fps"] for r in rows]) if rows else float("nan")
    lat = np.mean([r["latency_ms"] for r in rows]) if rows else float("nan")
    sig = np.mean([r["sigma_deg"] for r in rows]) if rows else float("nan")
    summary = {"sessions": len(rows), "mean_fps": round(float(fps), 1),
               "mean_latency_ms": round(float(lat), 1),
               "mean_sigma_deg": round(float(sig), 2),
               "pairs_file": f"pairs_{tag}.npz"}
    if ba:
        summary.update(bias=round(ba["bias"], 2),
                       loa_lower=round(ba["loa_lower"], 1),
                       loa_upper=round(ba["loa_upper"], 1),
                       mape_pct=round(ba["mape_pct"], 1), n=ba["n"])
        summary.update(extra_metrics(pairs[0], pairs[1]))
    return summary


if __name__ == "__main__":
    import torch
    dev = "0" if torch.cuda.is_available() else "cpu"
    out = {}
    print(">> stock yolo11n on test subjects 0-9 ...")
    out["stock_n_test"] = run("yolo11n-pose.pt", "uco_report_n_test.csv",
                              "stock_n", dev)
    print("   ", out["stock_n_test"])
    ftw = "ft_runs/yolo11n_uco_slr/weights/best.pt"
    for cand in ("ft_runs/yolo11n_uco_slr/weights/best.pt",
                 os.path.join("runs", "pose", "ft_runs", "yolo11n_uco_slr",
                              "weights", "best.pt")):
        if os.path.exists(cand):
            ftw = cand; break
    print(">> fine-tuned yolo11n on test subjects 0-9 ...", ftw)
    out["finetuned_n_test"] = run(ftw, "uco_report_ft_test.csv",
                                  "finetuned_n", dev)
    print("   ", out["finetuned_n_test"])

    # merge into metrics_eval.json (mediapipe added later)
    path = "metrics_eval.json"
    prev = json.load(open(path)) if os.path.exists(path) else {}
    prev.update(out)
    json.dump(prev, open(path, "w"), indent=2)
    print(">> wrote", path)
