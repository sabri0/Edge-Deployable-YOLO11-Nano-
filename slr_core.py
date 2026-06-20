"""
slr_core.py
Core, dependency-light analytics for automated Straight Leg Raise (SLR) assessment.
Everything here is unit-testable WITHOUT downloading model weights, so the math is
guaranteed correct before it is wired to a live YOLO11-pose stream.

Author: (revised pipeline accompanying the manuscript review)
"""
from __future__ import annotations
import numpy as np

# ---- COCO-17 keypoint indices (the format YOLO11-pose returns) ----------------
COCO = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
}


def joint_angle(a, b, c):
    """Interior angle (degrees, 0-180) at vertex b formed by points a-b-c.
    Robust 2D implementation using atan2(|cross|, dot) — never blows up."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    ba, bc = a - b, c - b
    cross = ba[0] * bc[1] - ba[1] * bc[0]
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    return float(np.degrees(np.arctan2(abs(cross), dot)))


def knee_flexion_angle(kp, side="r"):
    """Angle at the knee (thigh vs. shank). 180 deg = fully extended leg."""
    return joint_angle(kp[COCO[f"{side}_hip"]], kp[COCO[f"{side}_knee"]],
                       kp[COCO[f"{side}_ankle"]])


def hip_flexion_angle(kp, side="r"):
    """TRUE trunk-to-thigh angle at the hip (trunk vs. thigh), using the shoulder
    as the trunk reference. This is the quantity the SLR clinically targets and
    the quantity the reviewed manuscript mislabelled: its Eq. (1) used
    hip-knee-ankle, which is the KNEE angle, not trunk-to-thigh. Requires a
    shoulder keypoint, which 3-joint (hip/knee/ankle) tracking cannot provide."""
    return joint_angle(kp[COCO[f"{side}_shoulder"]], kp[COCO[f"{side}_hip"]],
                       kp[COCO[f"{side}_knee"]])


def savgol(series, window=11, poly=3):
    """Savitzky-Golay smoothing with graceful fallback for short series."""
    from scipy.signal import savgol_filter
    s = np.asarray(series, float)
    if len(s) < window:
        window = len(s) - (1 - len(s) % 2)  # largest odd <= len
        if window <= poly:
            return s
    return savgol_filter(s, window, poly)


def bland_altman(method, reference):
    """Returns dict with bias, SD of differences, 95% limits of agreement, MAPE."""
    m, r = np.asarray(method, float), np.asarray(reference, float)
    diff = m - r
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    mape = float(np.mean(np.abs(diff) / np.clip(np.abs(r), 1e-6, None)) * 100)
    return {
        "bias": bias, "sd_diff": sd,
        "loa_lower": bias - 1.96 * sd, "loa_upper": bias + 1.96 * sd,
        "mape_pct": mape, "n": int(len(diff)),
    }


def signal_variance(series):
    """Jitter descriptors on the RAW (unfiltered) angle series."""
    s = np.asarray(series, float)
    sigma = float(np.std(s, ddof=1))
    mean = float(np.mean(s))
    cv = float(sigma / mean * 100) if mean else float("nan")
    rmssd = float(np.sqrt(np.mean(np.diff(s) ** 2))) if len(s) > 1 else 0.0
    return {"sigma": sigma, "cv_pct": cv, "rmssd": rmssd}


def conformity(angle_series, lo, hi, peak_fn=np.max, frac_threshold=0.80):
    """Frame- and session-level conformity for a target angle window [lo, hi].
    NOTE: the clinical window MUST match the chosen angle definition. For a
    supine SLR using hip flexion, a higher raise = a SMALLER trunk-to-thigh
    angle, so the window is on hip-flexion magnitude, not the 150-165 deg band
    the manuscript applied to a mislabelled angle."""
    s = np.asarray(angle_series, float)
    in_window = (s >= lo) & (s <= hi)
    frame_rate = float(np.mean(in_window))
    return {
        "frame_conformity_pct": frame_rate * 100,
        "session_conformant": bool(frame_rate >= frac_threshold),
        "peak_angle": float(peak_fn(s)),
    }


def parse_uco_p2d(path, n_joints=3):
    """Parser for UCO Physical Rehabilitation `camX_p2d.txt` 2D ground-truth files.
    Lower-body files carry hip/knee/ankle (3 joints). The exact delimiter is not
    documented in the repo README, so we accept comma/space/tab and tolerate an
    optional leading frame index. Returns an (n_frames, n_joints, 2) array.
    The authoritative source is dataset_2d.json; this is a convenience fallback."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.replace(",", " ").split()]
            if len(vals) == n_joints * 2 + 1:   # leading frame index
                vals = vals[1:]
            if len(vals) < n_joints * 2:
                continue
            rows.append(vals[: n_joints * 2])
    arr = np.asarray(rows, float).reshape(-1, n_joints, 2)
    return arr
