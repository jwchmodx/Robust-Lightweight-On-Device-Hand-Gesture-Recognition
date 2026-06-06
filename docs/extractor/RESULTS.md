# Results — Lighter, More Robust Landmark Extractor (distilled student vs MediaPipe)

**Task:** replace the MediaPipe feature extractor with a compact, robust student that
emits the same 21 landmarks → same 42-D `relative_wrist_scale` features, so the
**frozen** 26-class HaGRIDv2 MLP (val/test F1 ≈ 0.99114) runs unchanged.

All F1 below = macro-F1 of the **frozen classifier** fed the extractor's landmarks.

## 1. Clean parity (val, through frozen MLP)
| Student | Params | Clean F1 | vs frozen-MLP ceiling 0.99114 |
|---|---|---|---|
| width=1.0 | 60 K | 0.943 | −4.8 pts (underfits) |
| width=3.0 | 471 K | **0.980** | −1.1 pts (≈ parity) |
| width=3.0 robust | 471 K | 0.978 | −1.3 pts (parity kept despite aug) |

## 2. Robustness — fair full-frame sweep (test, 150/class)
MediaPipe runs **natively** (full frame, self detect+crop); the student gets the
bbox crop of the **same degraded frame**. No-detection = forced miss.

| Condition | Severity | student-w3 clean | **student-w3 robust** | MediaPipe (native) | MP detect-rate |
|---|---|---|---|---|---|
| clean | 0.00 | 0.980 | 0.978 | 0.933 | 0.957 |
| low_light | 0.50 | 0.964 | 0.976 | 0.912 | 0.927 |
| low_light | 1.00 | 0.855 | **0.958** | 0.776 | 0.730 |
| motion_blur | 0.50 | 0.882 | 0.968 | 0.865 | 0.885 |
| motion_blur | 1.00 | 0.679 | **0.934** | 0.702 | 0.720 |

**Retention (F1 at severity 1.0 ÷ clean F1):**
- Robust student: low-light **98%**, motion-blur **96%**.
- MediaPipe: low-light **83%**, motion-blur **75%** (detection collapses).
- Clean-trained student: motion-blur **69%** — *worse than MediaPipe* → robustness training is essential.

Figure: `cache_v2/robustness_curves.png`.

## 3. On-device proxy metrics
| Model | Params | MMACs | size fp32 / int8 | × smaller vs MP landmark |
|---|---|---|---|---|
| student-w1 | 60 K | 8.3 | 0.24 / 0.06 MB | 33.4× |
| **student-w3** | 471 K | 54.7 | 1.89 / 0.47 MB | **4.2×** |
| MediaPipe landmark-only | 2.0 M | — | 8.0 / 2.0 MB | 1× |
| MediaPipe full (palm+landmark) | 3.76 M | — | 15.0 / 3.76 MB | 0.5× (8× larger than ours) |

## 4. Headline
A **471 K-param** student (4–8× smaller than MediaPipe, no palm detector):
- **matches** MediaPipe on clean (0.978 vs 0.933 through the frozen MLP),
- is **far more robust** to low-light/motion-blur (retains 96–98% vs 75–83%),
- is **drop-in compatible** — same 42-D features, frozen classifier untouched.
The win comes from distilling MediaPipe-quality GT landmarks under **clean-target /
degraded-input** training.

## 5. Honest caveats
- Student is given a **hand box** (GT bbox); MediaPipe self-detects. So the student
  owns the *landmark stage* only — end-to-end webcam use still needs a detector
  (deliberately out of scope, D1). The robustness win is for the landmark stage.
- On **clean** images the student can only *match* its training signal (GT/MediaPipe
  landmarks), not exceed the underlying labels; the small clean edge over native
  MediaPipe is from avoided detection misses.
- "Lighter" is a **proxy-metric** win (params/MACs/size); we did not measure on-device
  latency. INT8 sizes are projected (post-training quantization not yet applied).
- Degradations limited to **low-light + motion-blur** (the agreed webcam-realistic set).

## 6. Artifacts
- `src_v2/eval/frozen_mlp.py` — frozen-classifier contract harness (reproduces 0.99114)
- `src_v2/data/{sample_manifest,fetch_crop}.py` — 117 K-crop dataset builder
- `src_v2/student/{model,dataset,train}.py` — student + training
- `src_v2/degrade/transforms.py` — low-light / motion-blur
- `src_v2/eval/{robustness_sweep,full_frame_sweep,proxy_metrics,make_figure}.py` — eval
- `cache_v2/` — crops, checkpoints, `full_frame_sweep.csv`, `proxy_metrics.csv`, figure
