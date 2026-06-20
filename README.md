# Automated Straight Leg Raise (SLR) assessment — YOLO11-pose + UCO dataset

A reproducible, edge-oriented pipeline for objective SLR assessment, validated
against a **real, public** rehabilitation dataset with motion-capture ground
truth. This repo also reproduces the figures, tables, and revised manuscript that
accompany the article.

---

## 1. What's in here

| File | Role |
|------|------|
| `slr_core.py` | Dependency-light geometry, smoothing, Bland–Altman, jitter, conformity, UCO parser. No model weights needed — the math is unit-testable on its own. |
| `uco_slr_pipeline.py` | YOLO11-pose extraction + UCO validation + single-video + webcam modes. |
| `build_finetune_dataset.py` | Builds a YOLO-pose fine-tuning dataset (subject-disjoint split) from the UCO clips. |
| `train_finetune.py` | Fine-tunes `yolo11n-pose` on the UCO SLR slice. |
| `eval_testset.py` | Three-way evaluation (stock / fine-tuned YOLO) on held-out test subjects. |
| `mediapipe_slr.py` | MediaPipe (BlazePose) baseline, scored identically. |
| `make_figures.py` | Renders the article figures + `results_table.csv` from saved artefacts. |
| `make_app_example.py`, `make_app_screens.py` | Worked single-clip example + dashboard-style figure. |
| `make_test_clip.py` | Generates a synthetic `test_slr.mp4` smoke-test clip (no dataset needed). |
| `generate_docx.py` | Renders the revised manuscript `.docx` from the reproduced metrics. |
| `webapp/app.py` | Flask dashboard: live results + "try the model" upload + dataset browser. |
| `requirements.txt` | Core dependencies. |

---

## 2. Setup

Requires **Python 3.12**. From the project root:

```powershell
# Create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install core dependencies
pip install -r requirements.txt
```

> On macOS/Linux use `python3 -m venv .venv` and `source .venv/bin/activate`.

The first model call downloads `yolo11n-pose.pt` / `yolo11m-pose.pt` from
Ultralytics (needs internet); both weights are already present in this folder, so
they will be reused.

### Optional extras (not in `requirements.txt`)

Install only what you need for the corresponding step:

```powershell
pip install flask        # for the web dashboard (webapp/app.py)
pip install mediapipe    # for the MediaPipe baseline (mediapipe_slr.py)
pip install python-docx  # for generating the manuscript (generate_docx.py)
```

`matplotlib` is pulled in automatically with `ultralytics`, so the figure scripts
work out of the box.

> **GPU note:** the default `torch` from PyPI is the CPU build — everything runs
> on CPU (slower, but fine; use the lighter `yolo11n-pose.pt`). For an NVIDIA GPU:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`.
> All scripts auto-detect CUDA and fall back to CPU when it's absent.

---

## 3. Quick start — no dataset required

This proves the whole pipeline end-to-end without the access-restricted dataset.

```powershell
# (a) Make a synthetic smoke-test clip (creates test_slr.mp4)
python make_test_clip.py

# (b) Run the pipeline on it
python uco_slr_pipeline.py --video test_slr.mp4 --side r

# (c) Live webcam — opens an annotated real-time window (skeleton + knee/
#     trunk-thigh angles + fps). Press q or Esc in the window to stop.
python uco_slr_pipeline.py --webcam 0 --side r

# Want the same live window while replaying a video file? add --show
python uco_slr_pipeline.py --video test_slr.mp4 --side r --show
```

Output: a per-session CSV (`slr_session_report.csv`) with latency, fps,
raw-signal σ / CV / RMSSD (jitter), peak hip flexion, and frame/session
conformity.

---

## 4. Real-time live analysis

Run YOLO11-pose per frame on a webcam (or any video) and watch the skeleton and
joint angles update live. This is the same `process_video` loop used everywhere
else, so the numbers match the batch results.

```powershell
# Live webcam: opens an annotated window. Press q or Esc to stop.
python uco_slr_pipeline.py --webcam 0 --side r

# Replay a video file with the live window:
python uco_slr_pipeline.py --video datasets/clips_mp4/1/07/cam2.mp4 --side r --show

# Pick the leg / model / device:
python uco_slr_pipeline.py --webcam 0 --side l --model yolo11n-pose.pt --device 0
```

**On-screen overlay:** the trunk→hip→knee→ankle skeleton, the live **knee** angle,
the true **trunk-to-thigh (hip-flexion)** angle, and a running **fps** counter
(smoothed over the last ~30 frames).

**Controls & flags:**

| Flag | Effect |
|------|--------|
| `--show` | Force the live window on (default for `--video` is off). |
| `--no-show` | Disable the window even for `--webcam` (headless / benchmarking). |
| `--side l\|r` | Which leg to measure. |
| `--device 0\|cpu` | GPU index or CPU (default: auto-detect). |
| `q` or `Esc` | Quit the live window. |

The live window is **on by default for `--webcam`** and **off for `--video`**.
On exit, the session summary still prints and is written to
`slr_session_report.csv`.

> **Performance:** ~10 fps on CPU; **30+ fps on an NVIDIA GPU** (see the GPU note
> in §2). The live window needs GUI-enabled OpenCV — the `opencv-python` wheel in
> `requirements.txt` has it (only `opencv-python-headless` would not).

---

## 5. Get the dataset

**UCO Physical Rehabilitation dataset**
Aguilar-Ortega R, Berral-Soler R, Jiménez-Velasco I, *et al.* "UCO Physical
Rehabilitation: New Dataset and Study of Human Pose Estimation Methods on Physical
Rehabilitation Exercises." *Sensors* 2023; 23(21):8862. doi:10.3390/s23218862

- Repo: https://github.com/AVAuco/ucophyrehab
- Project page: https://www.uco.es/investiga/grupos/ava/portfolio/ucophyrehab/
- **Access:** email `inforeha@uco.es` with your name, affiliation, and research purpose.
- 27 subjects, 5 RGB cameras (1280×720), ~2160 sequences, **OptiTrack 2D/3D
  ground truth** for hip/knee/ankle (lower body).
- Relevant exercises: **03 / 07 = supine "lift the extended leg" (the SLR)**;
  01/05 and 02/06 = seated lower-limb exercises.

### Expected layout

The reproduction scripts read clips from `datasets/clips_mp4/`:

```
datasets/
  clips_mp4/
    dataset_2d.json                # side (left/right) metadata
    0/03/cam2.mp4                  # <subject>/<exercise>/<camera>.mp4
    0/03/cam2_p2d.txt              # OptiTrack 2D ground truth (hip/knee/ankle)
    0/07/cam2.mp4
    ...
```

`uco_slr_pipeline.py` alone accepts any root via `--uco-root`; the
build/train/eval/dashboard scripts assume `datasets/clips_mp4/`.

> The `datasets/` folder is **git-ignored** (it is ~23 GB) — keep it local.

---

## 6. Validate on the dataset

```powershell
# Proposed model (subjects 1-3, both SLR exercises, auto leg resolution)
python uco_slr_pipeline.py --uco-root datasets/clips_mp4 --subjects 1 2 3 --exercises 03 07 --side auto --model yolo11n-pose.pt

# Reference model (accuracy/latency trade-off)
python uco_slr_pipeline.py --uco-root datasets/clips_mp4 --model yolo11m-pose.pt
```

`--side auto` resolves the correct leg per (subject, exercise) from
`dataset_2d.json`. Output: per-session CSV + a Bland–Altman summary (model vs.
UCO ground-truth knee angle).

---

## 7. Full reproduction workflow

Run these in order. Each step writes artefacts the next step (and the article)
consumes.

```powershell
# 1. Build the fine-tuning dataset (subject-disjoint: train 13-26, val 10-12).
#    Writes ft_dataset/ (images, labels, uco_slr.yaml).
python build_finetune_dataset.py

# 2. Fine-tune yolo11n-pose. Writes ft_runs/yolo11n_uco_slr/weights/best.pt
python train_finetune.py

# 3. Three-way evaluation on HELD-OUT test subjects 0-9 (stock vs fine-tuned).
#    Writes uco_report_n_test.csv, uco_report_ft_test.csv, pairs_*.npz, metrics_eval.json
python eval_testset.py

# 4. MediaPipe baseline on the same test subjects (needs: pip install mediapipe).
#    Writes uco_report_mediapipe_test.csv, pairs_mediapipe.npz, updates metrics_eval.json
python mediapipe_slr.py --subjects 0 1 2 3 4 5 6 7 8 9 --out uco_report_mediapipe_test.csv

# 5. Render figures + results table from the saved artefacts -> figures/
python make_figures.py

# 6. (optional) Worked single-clip example + dashboard-style figure
python make_app_example.py
python make_app_screens.py

# 7. (optional) Generate the revised manuscript (needs: pip install python-docx)
python generate_docx.py
```

---

## 8. Web dashboard

```powershell
pip install flask
python webapp/app.py
# then open http://127.0.0.1:5000
```

Pages:
- `/` — live results read from the `uco_report_*.csv` files + `metrics*.json`.
- `/try` — upload an image/clip and run YOLO11-pose, returning the knee and true
  trunk-to-thigh (hip-flexion) angles.
- `/video`, `/api/*` — clip inference and JSON endpoints backing the pages.

The dashboard binds to `127.0.0.1` only (local use).

---

## 9. Outputs reference

| Artefact | Produced by |
|----------|-------------|
| `slr_session_report.csv` | `uco_slr_pipeline.py` (video/webcam/UCO modes) |
| `uco_report_*_test.csv`, `pairs_*.npz`, `metrics_eval.json` | `eval_testset.py`, `mediapipe_slr.py` |
| `ft_dataset/` | `build_finetune_dataset.py` |
| `ft_runs/.../best.pt` | `train_finetune.py` |
| `figures/*.png`, `figures/results_table.csv` | `make_figures.py`, `make_app_*.py` |
| `*.docx` | `generate_docx.py` |

---


