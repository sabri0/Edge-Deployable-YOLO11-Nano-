# Novelty & Literature Positioning

This document records the defensible novelty of the study and how it sits against the
literature, using only what was actually measured in this folder.

## 1. The gap in the literature

| Approach | Representative refs | Validation reference used | Reported agreement | Latency | Special hardware | Mocap-validated? | Reproducible? |
|---|---|---|---|---|---|---|---|
| MediaPipe / BlazePose in rehab | Bazarevsky 2020; Chen 2022; Liao 2023 | repetition counts / subjective / self-reference | ~85–90% knee-angle "accuracy" | ~45 ms (CPU) | none | **No** | partial |
| OpenPose | Cao 2021 | — | 95–97% (unconstrained) | 80–200 ms | GPU | No | partial |
| Kinect / depth sensors | Mehrizi 2020 | marker mocap | 88–92% | real-time | **depth sensor** | partial | No |
| Optoelectronic (Vicon/OptiTrack) | gold standard | — | <0.5° | 200+ Hz | **$100k+ rig** | — | No |
| **This work — fine-tuned YOLO11n** | — | **UCO OptiTrack 2-D GT** | **bias −0.08°, LoA ±14°, MAPE 3.4%, r 0.48** | **18.6 ms** | none (4 GB laptop GPU) | **Yes** | **Full (code+weights+arrays)** |

Two structural gaps recur in the rehab-pose literature:
1. **Validation is rarely against motion capture.** Most rehab pose-estimation papers
   validate against subjective scores, repetition counts, or the model's own outputs —
   not a marker-based reference. Numbers like "90% accuracy" are therefore not
   comparable across studies and not anchored to ground truth.
2. **Results are rarely reproducible.** Weights, paired data and evaluation code are
   seldom released, so the field cannot audit claimed agreement.

This study closes both gaps for the SLR task.

## 2. Statement of novelty (defensible claims)

1. **First motion-capture-anchored, three-way benchmark (MediaPipe vs stock vs
   fine-tuned YOLO11-Nano) for the supine SLR on the UCO dataset**, scored against the
   dataset's own OptiTrack-derived 2-D ground truth with identical metrics for all
   methods.
2. **Causal isolation of the fine-tuning effect** via a subject-disjoint, held-out
   design: fine-tuning collapses the limits of agreement ±32° → ±14°, bias → ≈0°, and
   correlation r ≈ 0 → 0.48. The contribution that matters is *domain adaptation*, not
   the nano architecture.
3. **Three corrective (negative) findings** absent from the prior narrative:
   (a) a *stock* nano model is **worse** than MediaPipe; (b) a 7×-larger YOLO11-Medium
   does **not** improve agreement, so capacity — not parameter count — is the limit;
   (c) the common "lightweight ⇒ low jitter" claim is **false** here — MediaPipe is the
   smoothest by RMSSD, and raw σ is a misleading jitter proxy when the signal genuinely
   moves.
4. **Methodological corrections** to the SLR-measurement pipeline: a correctly named
   knee-flexion angle (vs the previously mislabelled "trunk-to-thigh"), automatic
   per-sequence limb-side resolution from dataset metadata, and per-session aggregates
   that prevent frame-level pseudoreplication.
5. **End-to-end reproducibility**: released pipeline, fine-tuned weights, raw paired
   arrays (`pairs_*.npz`), consolidated metrics (`metrics_eval.json`), figures, and an
   interactive dashboard — a standard rarely met in this sub-field.

## 3. Honest framing of the accuracy numbers

Our tracking-accuracy (frames within 5° of mocap) of 49% for the fine-tuned model is
**not** directly comparable to literature "90%+" figures, because (i) ours is a strict
per-frame test against marker-based ground truth, and (ii) the COCO hip keypoint and the
OptiTrack hip marker are not the same anatomical point, which inflates apparent error.
The contribution is the *controlled, mocap-anchored comparison*, not a new
state-of-the-art accuracy headline. Within that fair comparison, the fine-tuned nano is
the best edge-deployable method on agreement and the fastest.
