"""Robustness sweep: compare landmark extractors under graded degradation.

For each extractor (student checkpoints and/or the MediaPipe baseline) and each
(condition, severity), degrade the test crops, extract 21 landmarks, map crop->
image via the stored affine, relative_wrist_scale, and score macro-F1 through the
FROZEN classifier. For MediaPipe we also report detection rate (a no-detection
counts as a miss -> fixed denominator, no inflation).

Usage:
  python preprocessing/extractor/eval/robustness_sweep.py --students w3=preprocessing/extractor/cache_v2/student_w3/best_student.pt \
         w3rob=cache_v2/student_w3_robust/best_student.pt --mediapipe --per-class 150
"""
import argparse
import sys
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
from frozen_mlp import FrozenClassifier                 # noqa: E402
from transforms import degrade, CONDITIONS, EVAL_SEVERITIES  # noqa: E402
from model import HandLandmarkNet                       # noqa: E402

_MP = None


def _mp_worker_init():
    global _MP
    import mediapipe as mp
    _MP = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                   min_detection_confidence=0.5)


def _mp_one(bgr):
    """Returns (21,2) crop-frame landmarks or None."""
    res = _MP.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0].landmark
    return np.array([[p.x, p.y] for p in lm], np.float32)


def load_eval(clf, per_class, split="test"):
    df = pd.read_parquet(ROOT / "cache_v2/targets.parquet")
    df = df[df.split == split]
    df = df.groupby("label", group_keys=False).head(per_class).reset_index(drop=True)
    crops_dir = ROOT / "cache_v2/crops"
    imgs, affines, labels = [], [], []
    for _, r in df.iterrows():
        p = crops_dir / r.split / r.label / f"{r.image_id}.jpg"
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        imgs.append(bgr)
        affines.append(np.asarray(r.crop_to_img, np.float32))
        labels.append(clf.label_to_idx.get(r.label, -1))
    return imgs, np.stack(affines), np.array(labels)


def crop_to_img(lm_crop, aff):
    """(N,21,2) crop-frame + (N,4) affine -> (N,21,2) image-normalized."""
    out = np.empty_like(lm_crop)
    out[:, :, 0] = aff[:, 0:1] + lm_crop[:, :, 0] * aff[:, 2:3]
    out[:, :, 1] = aff[:, 1:2] + lm_crop[:, :, 1] * aff[:, 3:4]
    return out


@torch.no_grad()
def student_predict(model, imgs, device):
    arr = np.stack([cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in imgs]).astype(np.float32) / 255.
    x = torch.from_numpy(arr.transpose(0, 3, 1, 2)).to(device)
    out = []
    for i in range(0, len(x), 512):
        out.append(model(x[i:i+512]).cpu().numpy())
    return np.concatenate(out).reshape(-1, 21, 2)


def mp_predict(imgs, pool):
    res = list(pool.map(_mp_one, imgs))
    det = np.array([r is not None for r in res])
    lm = np.stack([r if r is not None else np.zeros((21, 2), np.float32) for r in res])
    return lm, det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", nargs="*", default=[], help="name=ckpt.pt ...")
    ap.add_argument("--mediapipe", action="store_true")
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "cache_v2/robustness_sweep.csv"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = FrozenClassifier(device="cpu")
    imgs, affines, labels = load_eval(clf, args.per_class)
    valid = labels >= 0
    print(f"[eval] {len(imgs)} test crops, {args.per_class}/class")

    students = {}
    for spec in args.students:
        name, ckpt = spec.split("=", 1)
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        m = HandLandmarkNet(width=sd["width"]).to(device).eval()
        m.load_state_dict(sd["state_dict"])
        students[name] = m

    pool = None
    if args.mediapipe:
        from multiprocessing import Pool
        pool = Pool(args.workers, initializer=_mp_worker_init)

    rng = np.random.default_rng(0)
    rows = []
    conditions = [("clean", 0.0)] + [(c, s) for c in CONDITIONS for s in EVAL_SEVERITIES if s > 0]
    for cond, sev in conditions:
        dimgs = [degrade(b, None if cond == "clean" else cond, sev, rng) for b in imgs]
        row = {"condition": cond, "severity": sev}
        for name, m in students.items():
            lm = crop_to_img(student_predict(m, dimgs, device), affines)
            pred = clf.predict_from_landmarks(lm)
            row[f"{name}_F1"] = round(f1_score(labels[valid], pred[valid], average="macro"), 4)
        if pool is not None:
            lm_c, det = mp_predict(dimgs, pool)
            lm = crop_to_img(lm_c, affines)
            pred = clf.predict_from_landmarks(lm)
            pred[~det] = -999  # no-detection = forced miss (fixed denominator)
            row["mediapipe_F1"] = round(f1_score(labels[valid], pred[valid], average="macro"), 4)
            row["mediapipe_detrate"] = round(det[valid].mean(), 4)
        rows.append(row)
        print("  ", row)

    if pool:
        pool.close()
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\n[done] -> {args.out}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
