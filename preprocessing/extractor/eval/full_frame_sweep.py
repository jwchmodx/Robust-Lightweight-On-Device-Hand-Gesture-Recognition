"""Fair full-frame robustness sweep.

Degrade the FULL image, then:
  - MediaPipe runs natively on the full frame (its real detect+crop+landmark mode),
  - the student gets the bbox crop of the SAME degraded frame.
Both -> landmarks -> relative_wrist_scale -> frozen MLP -> macro-F1.
MediaPipe no-detection = forced miss (fixed denominator).

Step 1 (--fetch): cache ~per-class test FULL images to scratch.
Step 2 (default):  run the sweep over students + MediaPipe.
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "degrade"))
sys.path.insert(0, str(ROOT / "student"))
sys.path.insert(0, str(ROOT / "data"))
from frozen_mlp import FrozenClassifier                    # noqa: E402
from transforms import degrade, CONDITIONS, EVAL_SEVERITIES  # noqa: E402
from model import HandLandmarkNet                          # noqa: E402
from fetch_crop import fetch_jpg, crop_and_targets, ZIP_URL  # noqa: E402

FULL_DIR = Path.home() / "scratch/gesture_raw/test_full"
_MP = None


def _mp_init():
    global _MP
    import mediapipe as mp
    _MP = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                   min_detection_confidence=0.5)


def _mp_full(bgr):
    """MediaPipe on a full frame -> (21,2) image-normalized landmarks or None."""
    res = _MP.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0].landmark
    return np.array([[p.x, p.y] for p in lm], np.float32)


def subset(per_class):
    df = pd.read_parquet(ROOT / "cache_v2/manifest.parquet")
    df = df[df.split == "test"].groupby("label", group_keys=False).head(per_class)
    return df.reset_index(drop=True)


def do_fetch(df, workers):
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    from remotezip import RemoteZip
    print("[fetch] loading central dir...")
    with RemoteZip(ZIP_URL) as z:
        info = {i.filename: i for i in z.infolist() if i.filename.endswith(".jpg")}

    def grab(row):
        out = FULL_DIR / f"{row.image_id}.jpg"
        if out.exists():
            return True
        inf = info.get(row.zip_path)
        if inf is None:
            return False
        try:
            (out).write_bytes(fetch_jpg(ZIP_URL, inf))
            return True
        except Exception:
            return False
    from tqdm import tqdm
    ok = 0
    with ThreadPoolExecutor(workers) as ex:
        for r in tqdm(ex.map(grab, [r for _, r in df.iterrows()]), total=len(df)):
            ok += bool(r)
    print(f"[fetch] cached {ok}/{len(df)} full images -> {FULL_DIR}")


def load_full(df):
    imgs, bboxes, lms, labels = [], [], [], []
    for _, r in df.iterrows():
        p = FULL_DIR / f"{r.image_id}.jpg"
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        imgs.append(bgr); bboxes.append(np.asarray(r.bbox, np.float32))
        lms.append(np.asarray(r.landmarks, np.float32)); labels.append(r.label)
    return imgs, bboxes, lms, labels


@torch.no_grad()
def student_on_full(model, dimgs, bboxes, lms, device):
    crops, affs = [], []
    for img, bb, lm in zip(dimgs, bboxes, lms):
        res = crop_and_targets(img, bb, lm, 128, 1.3)
        if res is None:
            crops.append(np.zeros((128, 128, 3), np.uint8)); affs.append(np.array([0, 0, 1, 1], np.float32))
        else:
            crops.append(cv2.cvtColor(res[0], cv2.COLOR_BGR2RGB)); affs.append(res[2])
    x = torch.from_numpy(np.stack(crops).astype(np.float32).transpose(0, 3, 1, 2) / 255.).to(device)
    out = []
    for i in range(0, len(x), 512):
        out.append(model(x[i:i+512]).cpu().numpy())
    lm_crop = np.concatenate(out).reshape(-1, 21, 2)
    aff = np.stack(affs)
    img_lm = np.empty_like(lm_crop)
    img_lm[:, :, 0] = aff[:, 0:1] + lm_crop[:, :, 0] * aff[:, 2:3]
    img_lm[:, :, 1] = aff[:, 1:2] + lm_crop[:, :, 1] * aff[:, 3:4]
    return img_lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--students", nargs="*", default=[])
    ap.add_argument("--mediapipe", action="store_true")
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "cache_v2/full_frame_sweep.csv"))
    args = ap.parse_args()

    df = subset(args.per_class)
    if args.fetch:
        do_fetch(df, args.workers)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = FrozenClassifier(device="cpu")
    imgs, bboxes, lms, labels = load_full(df)
    y = np.array([clf.label_to_idx.get(l, -1) for l in labels]); valid = y >= 0
    print(f"[eval] {len(imgs)} full test images")

    students = {}
    for spec in args.students:
        name, ckpt = spec.split("=", 1)
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        m = HandLandmarkNet(width=sd["width"]).to(device).eval()
        m.load_state_dict(sd["state_dict"]); students[name] = m

    pool = None
    if args.mediapipe:
        from multiprocessing import Pool
        pool = Pool(args.workers, initializer=_mp_init)

    rng = np.random.default_rng(0)
    conds = [("clean", 0.0)] + [(c, s) for c in CONDITIONS for s in EVAL_SEVERITIES if s > 0]
    rows = []
    for cond, sev in conds:
        dimgs = [degrade(b, None if cond == "clean" else cond, sev, rng) for b in imgs]
        row = {"condition": cond, "severity": sev}
        for name, m in students.items():
            lm = student_on_full(m, dimgs, bboxes, lms, device)
            pred = clf.predict_from_landmarks(lm)
            row[f"{name}_F1"] = round(f1_score(y[valid], pred[valid], average="macro"), 4)
        if pool is not None:
            res = list(pool.map(_mp_full, dimgs))
            det = np.array([r is not None for r in res])
            lm = np.stack([r if r is not None else np.zeros((21, 2), np.float32) for r in res])
            pred = clf.predict_from_landmarks(lm); pred[~det] = -999
            row["mediapipe_F1"] = round(f1_score(y[valid], pred[valid], average="macro"), 4)
            row["mp_detrate"] = round(det[valid].mean(), 4)
        rows.append(row); print("  ", row)

    if pool:
        pool.close()
    out = pd.DataFrame(rows); out.to_csv(args.out, index=False)
    print(f"\n[done] -> {args.out}\n", out.to_string(index=False))


if __name__ == "__main__":
    main()
