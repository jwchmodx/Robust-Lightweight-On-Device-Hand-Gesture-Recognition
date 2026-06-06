# Revised Design Plan — A Lighter, More Robust Landmark Extractor for the HaGRID Static Model

**Scope (narrowed):** We own **only the MediaPipe feature-extraction stage**. The 33-class static MLP gesture classifier in `../hagrid/` is owned by a teammate and is treated as **frozen**. Our deliverable is a replacement landmark extractor that is **(a) lighter** and **(b) more robust to adverse imaging conditions**, while emitting features **byte-for-byte compatible** with the existing classifier so it runs unchanged.

Supersedes `../PLAN.md` (the original two-model static+dynamic proposal). Old exploratory code is archived in `../old/`.

---

## 1. The compatibility contract (non-negotiable)

The frozen MLP consumes a **42-D vector** produced by `relative_wrist_scale` preprocessing of 21 single-hand landmarks. Exact transform (from `../hagrid/src/make_hagrid_npy_rel.py`):

```
arr: (21, 2)            # x, y only — Z IS DISCARDED
wrist     = arr[0]
relative  = arr - wrist
scale     = max_i || relative[i] ||      # distance to FARTHEST landmark (not middle-MCP)
feature   = (relative / (scale + 1e-6)).reshape(42,)
```

- 33 classes (no `no_gesture`; `INCLUDE_NO_GESTURE=False`), **single hand only**.
- The classifier (`relative_wrist_scale_42d_single_hand_only_xlarge_...`) reports **F1 ≈ 0.99** on this representation.
- **Anything we build must output 21 (x,y) landmarks in HaGRID image-normalized coordinates**, after which this exact transform yields the 42-D input. Reproduce the transform verbatim — do **not** substitute the old wrist→middle-MCP scaling.

**Key fact:** the classifier was trained on HaGRID's **official `hand_landmarks` annotations** (themselves MediaPipe-generated). So those annotations are our **ground-truth targets**, and matching them = guaranteed compatibility.

---

## 2. The core reframing: we distill MediaPipe, we don't modify it

MediaPipe Hands is Google's **pre-trained, inference-only** two-stage pipeline — Palm Detector (~1.76M params) + Landmark Model (~2.0M params). No training code is released for the landmark regressor (Model Maker only retrains the *gesture* head). Therefore:

> "Make MediaPipe lighter" and "train MediaPipe on degraded images" are **impossible as literally stated**. Both goals are achieved by **knowledge distillation**: train a compact **student** landmark regressor that mimics the teacher's 21 landmarks. We choose the student's size, and we train it on degraded inputs for robustness.

Because the student emits the same 21 landmarks → same 42-D feature, **the classifier is untouched. Compatibility is automatic by construction.**

What "win" looks like, stated honestly:
- **Clean images:** student **matches** (cannot exceed) the teacher/GT → preserve F1 ≈ 0.99.
- **Adverse images:** student **beats** MediaPipe, because it is supervised with *clean-derived* targets on *degraded* inputs (see §5).
- **Size:** student ≪ teacher in params / MACs (proxy on-device metrics).

---

## 3. Two key decisions (recommended defaults — confirm or override)

**D1. Operating assumption → single-stage landmark regressor on a hand crop (RECOMMENDED).**
Train/eval on hand crops obtained from HaGRID's **bbox annotations**. This isolates the landmark-regression contribution and lets us drop the heavy palm detector from *our* scope. Detection (finding the hand in a full webcam frame) is a **separable concern**: keep MediaPipe's palm detector at deploy time, or treat a lightweight detector as a stretch goal. Rationale: biggest, most-achievable lightweight win; HaGRID is hand-prominent.
- *Alternative:* full-frame regressor (no crop) — heavier, harder, only needed if we must own end-to-end detection.

**D2. Label source → HaGRID official `hand_landmarks` annotations (RECOMMENDED).**
They *are* the contract the MLP was trained on, they're MediaPipe-quality, and using them avoids re-running MediaPipe. Download `annotations.zip` (with landmarks).
- *Alternative / augmentation:* also run MediaPipe to cross-check or to label any images lacking annotations.

---

## 4. Data plan (we deleted the raw images — re-acquire)

| Item | Source | Use |
|---|---|---|
| HaGRID images | re-download (scratch, sharded, delete-after) | **student CNN input** (clean) |
| HaGRID annotations (`annotations_with_landmarks/annotations.zip`) | sbercloud / repo `download.py` | **21-landmark targets + bbox for crops + labels + user IDs** |

- Crops: use bbox to crop each labeled hand, resize to a fixed student input (e.g. 96×96 or 128×128).
- Single-hand filter + drop samples with ≠21 landmarks (mirror the teammate's `make_hagrid_npy_rel.py` filters exactly).
- Persist only the **crop tensors + landmark targets** needed for training; delete raw per the >1 GB / scratch policy.
- **Reuse the teammate's exact train/val/test split** (official HaGRID splits, subject-disjoint). Note: `npy_dataset/y_*_id` is the *class* index, not user id — but the **annotations do carry `user_id`** if subject-aware analysis is ever needed.

**Implemented (Phase 2):**
- Source images = official **HaGRIDv2 512p zip (128 GB)**. No small v2 subset exists; remotezip is too slow (~1.8 s/req). Solution: **custom parallel 1-RTT ranged fetcher** (`src_v2/data/fetch_crop.py`, ~20–25 img/s @ 64 threads) pulling only sampled images.
- `src_v2/data/sample_manifest.py` → manifest of **117,000 single-hand samples** (train 3000/cls, val 500, test 1000 × 26 classes) with bbox + GT landmarks from `annotations.zip`.
- Each sample stored as a **128×128 JPEG crop** (~5 KB; ~0.58 GB total) + `targets.parquet` row: `landmarks_crop`(42, crop-frame) and **`crop_to_img`**(4) affine `[ox,oy,sx,sy]`.
- **Compatibility-critical:** `crop_to_img` maps crop-frame landmarks back to original-image-normalized coords (`x=ox+u·sx, y=oy+v·sy`, `sx≠sy` preserves aspect ratio). Verified: GT landmarks reconstructed via this affine → `relative_wrist_scale` → frozen MLP reproduces ~0.98+ accuracy. **The student predicts crop-frame landmarks; eval applies `crop_to_img` before the frozen MLP.**

---

## 5. Distillation + robustness training (the heart of the project)

**Targets from clean, inputs degraded — the labeling order is everything:**
```
target   = landmarks_GT(clean crop)          # HaGRID annotation (or MediaPipe(clean))
input    = degrade(clean crop)               # synthetic adverse condition
student(input) -> predicted 21 (x,y) landmarks
loss     = || student(degrade(x)) - target ||   (e.g. wing/L1/MSE on landmarks)
```
This is the **only correct setup**: it teaches the student to output correct landmarks on inputs that defeat MediaPipe. ❌ Never degrade-then-MediaPipe-label (the teacher fails → garbage labels).

**Synthetic degradations — FINALIZED to two conditions** (webcam-realistic; via `albumentations`, parameterized severities):
- **Low light** — gamma / brightness reduction (± mild color/contrast shift).
- **Motion blur** — directional blur kernels (varied angle + length).
- Train with a **mix of clean + graded low-light/motion-blur**; reserve held-out severities for the eval sweep.
- *Dropped from scope:* fog/haze (unrealistic indoors), JPEG/low-res/noise (kept only as optional secondary stress tests, not headline).

**Optional robustness boosters:**
- **Consistency loss:** `|| student(degrade(x)) − student(clean(x)) ||` and/or `|| student(degrade(x)) − target ||` — robustness as regularization.
- **Per-landmark confidence/visibility head** so the downstream MLP can down-weight occluded joints.

---

## 6. Student architecture

Compact CNN landmark regressor (image crop → 42 outputs):
- Backbone: tiny MobileNetV3-small / a few depthwise-separable conv blocks; target **~0.2–0.5M params** (vs 2.0M landmark model, 3.76M full pipeline) → **4–18× reduction**.
- Head: global pool → FC → 42 (x,y), sigmoid to [0,1] image-normalized coords.
- Train in PyTorch (env `kmw_gesture`); export proxy metrics with `ptflops`.
- **Compression (stretch):** INT8 post-training quantization + magnitude pruning → headline size/MACs numbers.

---

## 7. Evaluation protocol (report all three together)

1. **Landmark accuracy** — normalized mean per-joint error (NME, normalized by hand size) of student vs GT, on **clean** and at **each degradation severity**. Primary scientific metric.
2. **End-to-end compatibility + robustness** — feed student landmarks → `relative_wrist_scale` → **frozen MLP**; report **F1** on clean (must ≈ 0.99 to prove compatibility) and per-condition (the robustness win vs MediaPipe-fed baseline).
3. **On-device proxy** — params, model size (MB), MACs/FLOPs, input dim; student vs MediaPipe.

- **Baseline to beat = MediaPipe itself** fed into the same frozen MLP, evaluated on the same clean+degraded sets. The story: *equal on clean, better on adverse, smaller in size.*
- Robustness sweep uses the **fixed-denominator** rule (a sample undetectable under degradation counts as a miss, not a drop) so the win isn't inflated.
- Use the teammate's split; optionally 5-fold for final numbers (mean ± std).

---

## 8. Repo layout (under `revised_design_plan/` scope, code in `../src_v2/`)

```
On-Device Hand Gesture Recognition/
├── revised_design_plan/
│   └── revised_design_plan.md        # this file
├── hagrid/                           # teammate's FROZEN classifier (read-only to us)
├── src_v2/
│   ├── data/        # re-download, crop from bbox, build (crop, landmark) pairs
│   ├── degrade/     # albumentations adverse-condition transforms + severities
│   ├── student/     # compact regressor, train loop, distillation+consistency loss
│   ├── compress/    # quantization / pruning
│   └── eval/        # NME, frozen-MLP F1 harness, proxy metrics, severity sweeps
└── old/                             # archived original exploration
```

---

## 9. Sequencing

1. **Contract harness:** load teammate's npy + `make_hagrid_npy_rel.py`; build a frozen-MLP eval wrapper. Reproduce F1 ≈ 0.99 from stored landmarks to lock the baseline. *(no new data needed)*
2. **Re-acquire data:** download HaGRID images + annotations; build (clean crop → GT 21-landmark) pairs on the teammate's split.
3. **Teacher/GT baseline:** feed GT (and MediaPipe) landmarks → frozen MLP; confirm clean F1 and measure MediaPipe under degradation (the bar to beat).
4. **Student v0 (clean only):** train compact regressor to clean-parity NME and F1. *Get parity before robustness.*
5. **Robustness training:** add graded degradations + consistency loss; re-run the severity sweep.
6. **Compression:** quantize/prune; record proxy-metric deltas.
7. **Final eval + figures:** the three-metric table, severity curves, efficiency–accuracy scatter.

---

## 10. Decisions (RESOLVED) & risks

- [x] **D1 — single-stage crop regressor** (drop palm detector from our scope; detection separable).
- [x] **D2 — HaGRID GT annotations** as landmark targets (guarantees compatibility).
- [x] **Degradations — low-light + motion blur** only (headline); others optional stress tests.
- [ ] Student input crop size (96 vs 128) and exact param budget — finalize during Student v0.
- **Risk — parity is the hard part:** landmark *regression* is harder than classification, and small errors hurt look-alike/inverted classes (`peace`/`peace_inverted`, `three`/`three2`/`three3`, `*_inverted`). Budget time to reach clean parity before claiming the robustness gain.
- **Risk — "lighter" may be academic:** if MediaPipe already runs fine on the target webcam, the win is in proxy metrics (params/FLOPs), not perceived latency. Lead the narrative with **robustness**, support with size.
- **Risk — detection out of scope:** a crop-based student assumes a localized hand; end-to-end webcam use still needs a detector (MediaPipe palm or a stretch-goal replacement).
```
