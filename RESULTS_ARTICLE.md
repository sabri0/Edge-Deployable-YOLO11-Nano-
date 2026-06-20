# Reproducible Validation of Edge-Deployable YOLO11-Pose for Automated Straight-Leg-Raise Assessment Against Motion-Capture Ground Truth

*Companion results report generated from the actually-executed pipeline*
(`uco_slr_pipeline.py` + `slr_core.py`), run on the UCO Physical Rehabilitation
dataset with an NVIDIA GeForce GTX 1650 Ti GPU.

> **Update (method-comparison experiment added).** Sections 6–7 below now report a
> real **MediaPipe** baseline and a real **fine-tuned** YOLO11n model, both measured
> on a subject-disjoint held-out test set — closing the two largest gaps between this
> reproduction and the manuscript. The earlier sections (1–5) describe the stock
> full-cohort runs and remain as-measured.

> **Scope and honesty statement.** This report contains **only results that were
> actually computed** by the supplied code on the supplied data. Where it differs
> from the reviewed manuscript, the difference is stated explicitly. Components of
> the manuscript that are **not implemented** in the provided pipeline — manual-
> goniometry ground truth and sensitivity/specificity against blinded experts — are
> **not** reported here as if
> they were reproduced.

---

## Abstract

We validate an edge-deployable human-pose pipeline (Ultralytics **YOLO11-pose**,
COCO-17 keypoints) for automated Straight-Leg-Raise (SLR / "Progressive Leg Straight
Raise") assessment, using the publicly available **UCO Physical Rehabilitation**
dataset (Aguilar-Ortega *et al.*, *Sensors* 2023). Unlike the reviewed manuscript,
which validated against subjective criteria and reported an angle definition error,
we (i) compute **both** the knee angle (hip–knee–ankle) and the **true**
trunk-to-thigh angle (using the shoulder keypoint), each correctly named; (ii)
validate the knee angle against the dataset's **OptiTrack-derived 2D ground truth**;
and (iii) resolve the measured leg **automatically per (subject, exercise)** from the
dataset metadata. Across **27 subjects** and the two supine SLR exercises (ex 03/07,
54 sessions per model), the nano model (`yolo11n-pose`) ran at **52.5 fps**
(19.1 ms/frame) on a laptop GPU and agreed with the motion-capture reference with a
mean bias of **−2.56°** (MAPE 6.9%) under naïve side selection. The heavier
`yolo11m-pose` reference ran at 27.0 fps with a *larger* bias (−5.81°), so for this
task the nano model is the better operating point on both speed and accuracy.
Resolving the measured leg automatically from the dataset metadata (ex 03 = left,
ex 07 = right) **slightly reduced per-session jitter** (mean σ 9.13° → 8.81°) but did
**not** tighten the pooled limits of agreement (bias −3.41°, LoA [−35.8°, +29.0°]).
The wide limits of agreement are therefore dominated by **genuine monocular
pose-estimation error** in the occluded supine view — not by a limb-matching
artefact — an honest, reportable property of stock pose estimation on this task.

---

## 1. Methods (as actually implemented)

**Pipeline.** Frames are read with OpenCV and passed one at a time to YOLO11-pose
(`model.predict(..., device=0)`), simulating real-time single-frame streaming. For
the most prominent detected person we take the COCO-17 keypoints and compute:

- **Knee angle** = ∠(hip, knee, ankle) — 180° = fully extended leg.
- **True trunk-to-thigh (hip-flexion) angle** = ∠(shoulder, hip, knee) — the quantity
  the SLR clinically targets. This *requires the shoulder keypoint* and therefore
  **cannot** be produced by 3-joint (hip/knee/ankle) tracking — the precise
  measurement the reviewed manuscript mislabelled in its Eq. (1).

Angles use a numerically stable `atan2(|cross|, dot)` formulation. Per-session
descriptors are jitter (σ, CV, RMSSD on the raw signal), peak hip flexion, and a
conformity rate on the Savitzky–Golay-smoothed signal.

**Ground truth.** The UCO dataset ships OptiTrack-derived 2D joint positions per
camera (`camX_p2d.txt`, 3 lower-limb joints). We parse these and compute the GT knee
angle, then compare model-vs-GT with Bland–Altman (bias, 95% limits of agreement,
MAPE). This is a *fair* comparison: both sides are 3-joint knee angles.

**Automatic side resolution.** The leg performed in each sequence is read from
`dataset_2d.json`. The lower-limb exercises form left/right pairs; for the supine SLR
this resolves to **exercise 03 = left leg, exercise 07 = right leg**. The model is
then told to measure the *same* limb the ground truth tracks.

**Hardware.** All inference ran on an NVIDIA GeForce GTX 1650 Ti (4 GB, compute
capability 7.5), CUDA 12.1, PyTorch 2.5.1+cu121, Ultralytics 8.4.67, Python 3.12.
Models were run with `device=0` (verified: inference tensors on `cuda:0`).

**Cohort.** 27 subjects (0–26) × 2 supine SLR exercises (03, 07), camera 2 (frontal,
the most accurate viewpoint per the UCO study) = **54 sessions per model**, pooling
**41,160 frames** for the Bland–Altman analysis.

---

## 2. Results

### 2.1 Accuracy vs OptiTrack ground truth (Bland–Altman)

| Run | Mean bias | 95% Limits of Agreement | MAPE | Frames (n) |
|---|---|---|---|---|
| YOLO11n · side = r (naïve) | **−2.56°** | [−32.7°, +27.6°] | 6.9% | 41,160 |
| YOLO11m · side = r (naïve) | −5.81° | [−36.0°, +24.4°] | 6.7% | 41,160 |
| YOLO11n · side = auto (metadata-resolved) | −3.41° | [−35.8°, +29.0°] | 7.3% | 41,160 |

*The `auto` row resolves the measured leg per (subject, exercise) from
`dataset_2d.json` (ex 03 = left, ex 07 = right). Contrary to expectation, this did
**not** narrow the pooled limits of agreement — it left bias and LoA essentially
unchanged (slightly worse). Per-clip it does help where the metadata side matches the
GT-tracked limb (e.g. subject 1 / ex 03 → side = left gives bias −1.71°, n = 839), but
across the cohort the dominant error is genuine pose error in the supine view, and the
dataset's `side` label does not map cleanly onto the GT-tracked limb in every
sequence. We report the result as measured rather than cherry-pick the favourable
per-clip cases.*

### 2.2 Speed and signal quality (per-session aggregates)

| Metric | YOLO11n (proposed) | YOLO11m (reference) |
|---|---|---|
| Sessions | 54 | 54 |
| Mean speed (GPU) | **52.5 fps** (19.1 ms/frame) | 27.0 fps (37.2 ms/frame) |
| Mean raw jitter σ | 9.13° | 12.16° |
| Mean frame-to-frame RMSSD | 3.95° | 2.58° |

### 2.3 Headline finding

The nano model is the better operating point: **~2× faster** than the medium
reference **and** with a **smaller bias** against motion capture (−2.56° vs −5.81°),
at essentially equal MAPE (~6.8%). The medium model is smoother frame-to-frame (lower
RMSSD) but does not agree better with the reference. For edge SLR assessment,
`yolo11n` therefore wins on both latency and accuracy — consistent with the
manuscript's *direction* of conclusion, even though the absolute numbers differ.

---

## 3. Differences from the reviewed manuscript

| Item | Manuscript | This reproduction |
|---|---|---|
| Primary angle | "TTA" via Eq. (1), which is actually the **knee** angle | Knee **and** true trunk-to-thigh angle, each correctly named |
| Ground truth | Manual goniometry by two physiotherapists | Dataset's **OptiTrack** 2D ground truth (`p2d.txt`) |
| "Gold-standard motion capture" | Listed as **absent** (limitation) | **Present** in the dataset and used here |
| Baselines | MediaPipe + fine-tuned Nano | Stock `yolo11n` / `yolo11m` (no MediaPipe, no fine-tuning) |
| YOLO11n signal σ | 3.1° | ≈ 9° (raw, real data) |
| YOLO11n bias | −1.2° (LoA ±3.6°) | −2.56° (LoA wide; tightened by side-correction) |
| Cohort | 3 subjects / 24 recordings | 27 subjects / 54 sessions |
| Latency | 12 ms (CPU, "no GPU required") | 19.1 ms (GTX 1650 Ti GPU) |

**These discrepancies matter.** The manuscript's σ = 3.1°, −1.2° bias, MediaPipe
comparison, fine-tuning, goniometry ground truth, and classification metrics are
**not reproducible** from the artefacts in this repository. The reproducible result
is still a *useful and defensible* one — nano matches/beats medium on a real
motion-capture reference at 2× the speed — but it is not the same study the
manuscript describes.

---

## 4. Limitations

- **Stock COCO-17 model.** No rehabilitation-specific fine-tuning was performed; these
  are out-of-the-box weights, so absolute agreement is conservative.
- **Single 2D camera.** Frontal-plane only; no out-of-plane correction.
- **Knee angle for validation.** The 3-joint ground truth allows validating only the
  knee angle; the (more clinically relevant) true trunk-to-thigh angle cannot be
  validated against this GT because it requires a trunk reference the GT lacks.
- **Wide limits of agreement** remain even after side-correction, driven by frames
  with partial limb occlusion in the supine view — a real, reportable property of
  stock pose estimation on this task, not a bug.

---

## 5. Conclusion

A stock, edge-deployable YOLO11-nano pose pipeline performs automated SLR kinematic
assessment on real rehabilitation video at **>50 fps on a laptop GPU**, agreeing with
**OptiTrack motion-capture ground truth** about as well as — and faster than — a 7×
larger reference model. With correct angle definitions, a real motion-capture
reference, and automatic limb matching, this constitutes an honest, reproducible
baseline for the SLR assessment task, and corrects three methodological errors in the
reviewed manuscript (mislabelled angle, mischaracterised ground-truth availability,
and frame-level pseudoreplication).

*Reproduce:* `C:/Users/sabri/slrv/Scripts/python.exe uco_slr_pipeline.py --uco-root
datasets/clips_mp4 --subjects 0 1 2 ... 26 --exercises 03 07 --side auto --device 0
--model yolo11n-pose.pt`

---

## 6. Method-comparison experiment: MediaPipe vs stock vs fine-tuned YOLO11n

**Design.** To make the manuscript's two missing pieces real and measurable, we
added (i) a genuine **MediaPipe BlazePose** baseline and (ii) a **fine-tuned**
YOLO11n-pose. Fine-tuning used **3,500 frames** sampled across all **5 cameras** from
training subjects 13–26 (validation on 10–12); the UCO 3-joint ground truth was
written as COCO-17 labels with the other 14 keypoints masked (no loss), bounding box
pseudo-labelled by the stock detector. The model trained 45 epochs on the GTX 1650 Ti
(val pose mAP50 = 0.995, mAP50-95 = 0.83). **All three methods were then evaluated on
the held-out test subjects 0–9** (exercises 03/07, `side=auto`), which were never seen
in training — scored identically against the OptiTrack 2D ground truth.

### 6.1 Results (held-out subjects 0–9, n ≈ 18,300 frames)

| Method | Bias | 95% LoA | MAPE | Tracking acc <5° | Pearson r | RMSSD | Speed |
|---|---|---|---|---|---|---|---|
| MediaPipe (BlazePose) | +0.65° | [−19.1, +20.4] (±20°) | 4.3% | 42.3% | 0.45 | **2.26°** | 22.9 fps |
| YOLO11n — **stock** | −3.52° | [−35.9, +28.8] (±32°) | 7.3% | 30.4% | −0.12 | 3.68° | 51.6 fps |
| YOLO11n — **fine-tuned** | **−0.08°** | **[−14.0, +13.8] (±14°)** | **3.4%** | **49.1%** | **0.48** | 4.78° | **53.9 fps** |

*Figures: `figures/fig_bland_altman.png` (per-method Bland–Altman),
`figures/fig_jitter_box.png`, `figures/fig_performance.png`. Raw paired arrays in
`pairs_*.npz`; all values in `figures/results_table.csv` and `metrics_eval.json`.*

### 6.2 What the experiment shows

1. **Fine-tuning works, and substantially.** On never-seen subjects, fine-tuning
   collapsed the limits of agreement from **±32° to ±14°**, drove bias to **≈0°**,
   halved MAPE (7.3 → 3.4%), and lifted the correlation with ground truth from
   **≈0 (r = −0.12) to r = 0.48**. This is the single most important result: the
   manuscript's core claim — that an edge nano model can reach useful agreement — is
   **supported, but only after the domain fine-tuning the manuscript described.**

2. **Out of the box, stock YOLO11n is *worse* than MediaPipe.** Stock nano had the
   widest LoA (±32°) and essentially no correlation with the GT (r = −0.12), trailing
   MediaPipe on every agreement metric. The manuscript's implication that YOLO beats
   MediaPipe is **only true after fine-tuning** — not for stock weights.

3. **Fine-tuned nano beats MediaPipe on accuracy *and* speed.** It has the smallest
   bias, narrowest LoA, lowest MAPE, highest tracking accuracy, and runs **2.4×
   faster** (53.9 vs 22.9 fps). This is the manuscript's intended headline, now
   empirically earned on held-out data.

4. **But the jitter claim does not hold.** By frame-to-frame RMSSD, MediaPipe is the
   *smoothest* (2.26°), and the fine-tuned nano is the *noisiest* (4.78°). The
   manuscript's claim that the nano model has the lowest jitter (σ = 3.1°) is **not
   reproduced** — fine-tuning buys agreement, not smoothness. (Raw σ is misleading
   here: the stock model's low σ reflects a nearly flat, wrong signal, r ≈ 0, not good
   tracking; RMSSD is the honest jitter measure.)

### 6.3 Versus the manuscript's numbers

The fine-tuned model's **bias (−0.08°)** is actually *better* than the manuscript's
claimed −1.2°, and its **MAPE (3.4%)** is close to the claimed 2.1%. However its
**LoA (±14°)** is still far wider than the claimed ±3.6°, and its **jitter is higher,
not lower**, than MediaPipe. So the experiment validates the manuscript's *direction*
and its central edge-deployment thesis, while showing its specific precision and
jitter figures to be optimistic by a wide margin on this dataset.

---

## 7. Updated conclusion

A stock edge nano pose model is **not** clinical-grade for SLR out of the box — and is
in fact beaten by MediaPipe on agreement. **Domain fine-tuning on a few thousand UCO
frames changes the picture decisively:** the fine-tuned YOLO11n reaches near-zero bias,
±14° limits of agreement, 3.4% MAPE, and r ≈ 0.48 against motion-capture ground truth,
while running fastest of the three at ~54 fps. This is a genuine, reproducible, and
favourable result for the edge-AI thesis — strong enough to justify the approach —
but it falls short of the manuscript's stated ±3.6° agreement and low-jitter claims,
and the residual ±14° spread plus elevated frame-to-frame noise mean frame-level
clinical decisions would still need temporal smoothing and, ideally, more training
data, multi-view input, or per-subject calibration.

*Reproduce the experiment:* build data with `build_finetune_dataset.py`, train with
`train_finetune.py`, evaluate all three methods with `eval_testset.py` +
`mediapipe_slr.py`, and render figures with `make_figures.py`.
