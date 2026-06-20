#!/usr/bin/env python3
"""
Build a YOLO-pose fine-tuning dataset from the UCO SLR clips.

UCO ground truth gives only 3 lower-limb joints (hip, knee, ankle) for the leg
being exercised. We therefore emit COCO-17 labels in which ONLY those 3 joints are
labelled (visibility=2) and the other 14 are masked (visibility=0 -> no keypoint
loss), with the person bounding box pseudo-labelled by the stock yolo11n-pose model.
Side (which leg) is resolved per (subject, exercise) from dataset_2d.json.

Split (subject-disjoint, no leakage):
  train = subjects 13..26   (~200 frames)
  val   = subjects 10..12   (~40 frames)
  test  = subjects 0..9     (held out for evaluation; NOT built here)
"""
import os, glob, random, cv2, numpy as np, torch
from slr_core import parse_uco_p2d
from uco_slr_pipeline import load_uco_side_map
from ultralytics import YOLO

UCO = "datasets/clips_mp4"
OUT = "ft_dataset"
SLR_EX = ("03", "07")
CAMERAS = ("cam0", "cam1", "cam2", "cam3", "cam4")  # use all 5 viewpoints
COCO_LR = {"l": (11, 13, 15), "r": (12, 14, 16)}  # hip, knee, ankle indices
# Subject-disjoint split. Test = subjects 0..9 is HELD OUT (never trained on).
SPLITS = {"train": range(13, 27), "val": range(10, 13)}
TARGET = {"train": 3500, "val": 500}              # dense sampling caps
random.seed(0)

per_seq, per_ex = load_uco_side_map(UCO)
dev = "0" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11n-pose.pt")
if dev == "0":
    model.to("cuda:0")


def label_line(bbox, kp_by_idx, W, H):
    x1, y1, x2, y2 = bbox
    cx, cy = ((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H
    bw, bh = (x2 - x1) / W, (y2 - y1) / H
    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for i in range(17):
        if i in kp_by_idx:
            x, y = kp_by_idx[i]
            parts.append(f"{x/W:.6f} {y/H:.6f} 2")
        else:
            parts.append("0 0 0")
    return " ".join(parts)


def person_bbox(frame):
    res = model.predict(frame, device=dev, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    xywh = res.boxes.xywh.cpu().numpy()
    areas = xywh[:, 2] * xywh[:, 3]
    i = int(np.argmax(areas))
    cx, cy, w, h = xywh[i]
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def build_split(split, subjects, target):
    img_dir = os.path.join(OUT, "images", split)
    lbl_dir = os.path.join(OUT, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    sessions = []
    for s in subjects:
        for ex in SLR_EX:
            for cam in CAMERAS:
                vid = os.path.join(UCO, str(s), ex, f"{cam}.mp4")
                gt = os.path.join(UCO, str(s), ex, f"{cam}_p2d.txt")
                if os.path.exists(vid) and os.path.exists(gt):
                    sessions.append((s, ex, vid, gt))
    per_session = max(1, target // max(1, len(sessions)))
    n = 0
    for s, ex, vid, gtp in sessions:
        side = per_seq.get((str(s), ex)) or per_ex.get(ex) or "r"
        idxs = COCO_LR[side]
        p2d = parse_uco_p2d(gtp, 3)
        cap = cv2.VideoCapture(vid)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(p2d)
        usable = min(total, len(p2d))
        # sample evenly in the middle 80% of the clip
        cand = np.linspace(int(usable * 0.1), int(usable * 0.9),
                           per_session * 3).astype(int)
        random.shuffle(list(cand))
        got = 0
        for fi in cand:
            if got >= per_session:
                break
            joints = p2d[fi]
            if np.any(joints <= 0):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok:
                continue
            H, W = frame.shape[:2]
            bbox = person_bbox(frame)
            if bbox is None:
                continue
            kp = {idxs[j]: (float(joints[j][0]), float(joints[j][1]))
                  for j in range(3)}
            cam = os.path.splitext(os.path.basename(vid))[0]
            stem = f"s{s}_e{ex}_{cam}_f{int(fi)}"
            cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), frame)
            with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
                f.write(label_line(bbox, kp, W, H) + "\n")
            got += 1
            n += 1
        cap.release()
    print(f"  [{split}] wrote {n} frames from {len(sessions)} sessions "
          f"(~{per_session}/session)")
    return n


def main():
    os.makedirs(OUT, exist_ok=True)
    counts = {sp: build_split(sp, subs, TARGET[sp]) for sp, subs in SPLITS.items()}
    yaml = (f"path: {os.path.abspath(OUT)}\n"
            "train: images/train\nval: images/val\n"
            "kpt_shape: [17, 3]\n"
            "flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]\n"
            "names:\n  0: person\n")
    with open(os.path.join(OUT, "uco_slr.yaml"), "w") as f:
        f.write(yaml)
    print("dataset:", counts, "-> wrote", os.path.join(OUT, "uco_slr.yaml"))


if __name__ == "__main__":
    main()
