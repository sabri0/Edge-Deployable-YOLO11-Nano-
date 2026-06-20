#!/usr/bin/env python3
"""
SLR / PLSR assessment dashboard (Flask).
=======================================
Serves three things grounded in the REAL reproduced pipeline:

  1. /            -> dashboard: novelty, method corrections, and the actual
                     validation results read live from the uco_report_*.csv files
                     and metrics.json produced by uco_slr_pipeline.py.
  2. /try         -> upload an image (or short clip) and run YOLO11-pose on it,
                     returning an annotated frame plus the knee and true
                     trunk-to-thigh (hip-flexion) angles.
  3. /api/*       -> JSON endpoints backing the pages.

Run:
    C:/Users/sabri/slrv/Scripts/python.exe webapp/app.py
    then open http://127.0.0.1:5000
"""
from __future__ import annotations
import os, io, json, base64, glob, time
import numpy as np
from flask import Flask, render_template, request, jsonify

# --- locate project root (parent of this webapp/ folder) ---------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import sys
sys.path.insert(0, ROOT)
from slr_core import (COCO, knee_flexion_angle, hip_flexion_angle, savgol,
                      signal_variance, conformity)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB uploads

_MODELS = {}  # lazy cache: weight -> YOLO


def get_model(weight="yolo11n-pose.pt"):
    if weight not in _MODELS:
        from ultralytics import YOLO
        wpath = os.path.join(ROOT, weight)
        m = YOLO(wpath if os.path.exists(wpath) else weight)
        try:
            import torch
            if torch.cuda.is_available():
                m.to("cuda:0")
        except Exception:
            pass
        _MODELS[weight] = m
    return _MODELS[weight]


# ----------------------------------------------------------------------------
# Results loading
# ----------------------------------------------------------------------------
def _read_csv(path):
    import csv
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_reports():
    """Aggregate every uco_report_*.csv next to the pipeline into summary rows."""
    out = []
    label = {
        "uco_report_n_full.csv":  "YOLO11n  (side=r, all subjects)",
        "uco_report_m_full.csv":  "YOLO11m  (side=r, all subjects)",
        "uco_report_n_auto.csv":  "YOLO11n  (side=auto, corrected)",
        "uco_report.csv":         "YOLO11n  (subjects 1-3 demo)",
    }
    for path in sorted(glob.glob(os.path.join(ROOT, "uco_report*.csv"))):
        name = os.path.basename(path)
        rows = _read_csv(path)
        if not rows:
            continue
        def col(k):
            return np.array([float(r[k]) for r in rows if r.get(k) not in (None, "")])
        out.append({
            "file": name,
            "label": label.get(name, name),
            "n_sessions": len(rows),
            "mean_fps": round(float(np.mean(col("fps"))), 1),
            "mean_latency_ms": round(float(np.mean(col("latency_ms"))), 1),
            "mean_sigma_deg": round(float(np.mean(col("sigma_deg"))), 2),
            "mean_rmssd_deg": round(float(np.mean(col("rmssd_deg"))), 2),
            "mean_conformity_pct": round(float(np.mean(col("frame_conformity_pct"))), 1),
            "rows": rows,
        })
    return out


def load_metrics():
    p = os.path.join(ROOT, "metrics.json")
    if os.path.exists(p):
        return json.load(open(p))
    return {}


def load_eval():
    p = os.path.join(ROOT, "metrics_eval.json")
    if os.path.exists(p):
        return json.load(open(p))
    return {}


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html",
                           reports=load_reports(),
                           metrics=load_metrics(),
                           ev=load_eval())


@app.route("/figures/<path:name>")
def figures(name):
    from flask import send_from_directory
    return send_from_directory(os.path.join(ROOT, "figures"), name)


@app.route("/api/results")
def api_results():
    return jsonify({"reports": load_reports(), "metrics": load_metrics()})


# ----------------------------------------------------------------------------
# Try-the-model: run YOLO11-pose on an uploaded image
# ----------------------------------------------------------------------------
SKELETON = [("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle")]


def _annotate(img, kp, side, knee_ang, hip_ang):
    import cv2
    def pt(name):
        return tuple(int(v) for v in kp[COCO[f"{side}_{name}"]])
    for a, b in SKELETON:
        pa, pb = pt(a), pt(b)
        if min(pa) > 0 and min(pb) > 0:
            cv2.line(img, pa, pb, (0, 220, 0), 3)
    for name in ("shoulder", "hip", "knee", "ankle"):
        p = pt(name)
        if min(p) > 0:
            cv2.circle(img, p, 6, (0, 0, 255), -1)
    cv2.putText(img, f"knee={knee_ang:.1f}deg", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(img, f"trunk-thigh(hip)={hip_ang:.1f}deg", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    return img


@app.route("/try")
def try_page():
    return render_template("try.html")


# ----------------------------------------------------------------------------
# Test-video application: run the full per-frame pipeline on an uploaded clip
# ----------------------------------------------------------------------------
def _angle_plot(hip_raw, knee_raw):
    """Render hip (trunk-thigh) + knee angle time-series as a base64 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sm = savgol(hip_raw) if hip_raw.size else hip_raw
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=110)
    fig.patch.set_facecolor("#161b22"); ax.set_facecolor("#0d1117")
    x = np.arange(hip_raw.size)
    if hip_raw.size:
        ax.plot(x, hip_raw, color="#8b949e", lw=1, alpha=.6, label="hip raw")
        ax.plot(x, sm, color="#3fb950", lw=2, label="hip smoothed")
    if knee_raw.size:
        ax.plot(np.arange(knee_raw.size), knee_raw, color="#2f81f7", lw=1,
                alpha=.8, label="knee raw")
    for s in ax.spines.values():
        s.set_color("#30363d")
    ax.tick_params(colors="#8b949e"); ax.set_xlabel("tracked frame", color="#8b949e")
    ax.set_ylabel("angle (deg)", color="#8b949e")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


UCO_ROOT = os.path.join(ROOT, "datasets", "clips_mp4")
SLR_EX = ("03", "07")  # supine straight-leg-raise (03=left, 07=right)


@app.route("/api/dataset_videos")
def api_dataset_videos():
    """List subjects/exercises available in the local UCO clips_mp4 tree."""
    tree = {}
    if os.path.isdir(UCO_ROOT):
        for subj in sorted(os.listdir(UCO_ROOT), key=lambda x: (len(x), x)):
            sp = os.path.join(UCO_ROOT, subj)
            if not os.path.isdir(sp):
                continue
            exs = sorted(e for e in os.listdir(sp)
                         if glob.glob(os.path.join(sp, e, "cam*.mp4")))
            if exs:
                tree[subj] = exs
    return jsonify({"root_exists": os.path.isdir(UCO_ROOT),
                    "slr_exercises": list(SLR_EX), "tree": tree})


@app.route("/api/infer_dataset")
def api_infer_dataset():
    """Run the pipeline on a clip already in datasets/clips_mp4 and, when the
    OptiTrack p2d ground truth is present, report per-clip Bland-Altman."""
    from uco_slr_pipeline import (process_video, summarise_session,
                                  angles_from_uco_gt, load_uco_side_map)
    from slr_core import parse_uco_p2d, bland_altman
    subj = request.args.get("subject", "1")
    ex = request.args.get("exercise", "03")
    cam = request.args.get("camera", "cam2")
    weight = request.args.get("model", "yolo11n-pose.pt")
    side = request.args.get("side", "auto")

    folder = os.path.join(UCO_ROOT, subj, ex)
    vid = os.path.join(folder, f"{cam}.mp4")
    if not os.path.exists(vid):
        return jsonify({"error": f"clip not found: {subj}/{ex}/{cam}.mp4"}), 404

    if side == "auto":
        per_seq, per_ex = load_uco_side_map(UCO_ROOT)
        side = per_seq.get((str(subj), str(ex))) or per_ex.get(str(ex)) or "r"

    import torch
    dev = "0" if torch.cuda.is_available() else "cpu"
    model = get_model(weight)
    series = process_video(model, vid, side=side, device=dev)
    summ = summarise_session(series, 120.0, 160.0)
    if summ is None:
        return jsonify({"error": "no trackable person in this clip"}), 200
    summ = {k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in summ.items()}

    ba = None
    gt_path = os.path.splitext(vid)[0] + "_p2d.txt"
    if os.path.exists(gt_path):
        gt = parse_uco_p2d(gt_path, n_joints=3)
        gt_knee = angles_from_uco_gt(gt, side)
        n = min(len(gt_knee), len(series["knee_raw"]))
        if n > 5:
            b = bland_altman(series["knee_raw"][:n], gt_knee[:n])
            ba = {"bias": round(b["bias"], 2),
                  "loa_lower": round(b["loa_lower"], 1),
                  "loa_upper": round(b["loa_upper"], 1),
                  "mape_pct": round(b["mape_pct"], 1), "n": b["n"]}

    plot = _angle_plot(series["hip_raw"], series["knee_raw"])
    return jsonify({"device": "cuda:0" if dev == "0" else "cpu",
                    "clip": f"{subj}/{ex}/{cam}.mp4", "resolved_side": side,
                    "model": weight, "summary": summ, "bland_altman": ba,
                    "plot_png_b64": plot})


@app.route("/video")
def video_page():
    return render_template("video.html")


@app.route("/api/infer_video", methods=["POST"])
def api_infer_video():
    import tempfile
    from uco_slr_pipeline import process_video, summarise_session
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400
    side = request.args.get("side", "r")
    weight = request.args.get("model", "yolo11n-pose.pt")
    hip_lo = float(request.args.get("hip_lo", 120.0))
    hip_hi = float(request.args.get("hip_hi", 160.0))
    f = request.files["file"]
    suffix = os.path.splitext(f.filename or "clip.mp4")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(f.read()); tmp.close()
    try:
        import torch
        dev = "0" if torch.cuda.is_available() else "cpu"
        model = get_model(weight)
        series = process_video(model, tmp.name, side=side, device=dev)
        summ = summarise_session(series, hip_lo, hip_hi)
        if summ is None:
            return jsonify({"error": "no trackable person found in the clip "
                                     "(need visible hip/knee/ankle)"}), 200
        summ = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                for k, v in summ.items()}
        plot = _angle_plot(series["hip_raw"], series["knee_raw"])
        return jsonify({"device": "cuda:0" if dev == "0" else "cpu",
                        "side": side, "model": weight,
                        "summary": summ, "plot_png_b64": plot})
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.route("/api/infer", methods=["POST"])
def api_infer():
    import cv2
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400
    side = request.args.get("side", "r")
    weight = request.args.get("model", "yolo11n-pose.pt")
    data = np.frombuffer(request.files["file"].read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "could not decode image (send a JPG/PNG frame)"}), 400

    model = get_model(weight)
    import torch
    dev = "0" if (lambda: __import__("torch").cuda.is_available())() else "cpu"
    t0 = time.perf_counter()
    res = model.predict(img, device=dev, verbose=False)[0]
    dt = (time.perf_counter() - t0) * 1000.0

    if res.keypoints is None or len(res.keypoints) == 0:
        return jsonify({"error": "no person detected in the frame"}), 200
    xy = res.keypoints.xy.cpu().numpy()
    idx = 0
    if res.boxes is not None and len(res.boxes) == xy.shape[0]:
        areas = (res.boxes.xywh[:, 2] * res.boxes.xywh[:, 3]).cpu().numpy()
        idx = int(np.argmax(areas))
    kp = xy[idx]
    knee_ang = knee_flexion_angle(kp, side)
    hip_ang = hip_flexion_angle(kp, side)
    img = _annotate(img, kp, side, knee_ang, hip_ang)
    ok, buf = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buf).decode("ascii")
    return jsonify({
        "device": "cuda:0" if dev == "0" else "cpu",
        "latency_ms": round(dt, 1),
        "persons": int(xy.shape[0]),
        "side": side,
        "model": weight,
        "knee_angle_deg": round(float(knee_ang), 1),
        "trunk_thigh_angle_deg": round(float(hip_ang), 1),
        "annotated_jpg_b64": b64,
    })


if __name__ == "__main__":
    print(">> dashboard on http://127.0.0.1:5000  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=5000, debug=False)
