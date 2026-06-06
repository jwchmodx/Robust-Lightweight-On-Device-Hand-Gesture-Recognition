"""Build a sampling manifest from HaGRIDv2 annotations for the student dataset.

Selects single-hand samples from the 26 frozen-classifier classes, caps per
class/split, and records everything needed to fetch + crop later:
  split, label, image_id, zip_path, bbox(x,y,w,h), landmarks(42).

The 26 classes are read from the frozen model config so they match exactly.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
ANN_DEFAULT = Path.home() / "scratch/gesture_raw/ann/annotations"
MODEL_CFG = (REPO_ROOT / "models/static/hagrid/experiments"
             / "relative_wrist_scale_42d_single_hand_only_xlarge_norm_bn_sch_do0p2_seed1"
             / "seed_1" / "config.json")
ZIP_PREFIX = "HaGRIDv2_dataset_512"
SPLITS = ["train", "val", "test"]
NUM_LANDMARKS = 21


def selected_classes():
    return set(json.loads(MODEL_CFG.read_text())["target_names"])  # 26 single-hand classes


def sample_split(split_dir: Path, classes, cap, rng):
    rows = []
    for cls in sorted(classes):
        jp = split_dir / f"{cls}.json"
        if not jp.exists():
            print(f"[warn] missing {jp}")
            continue
        data = json.loads(jp.read_text())
        cand = []
        for image_id, rec in data.items():
            labels = rec.get("labels", [])
            lms = rec.get("hand_landmarks", [])
            bboxes = rec.get("bboxes", [])
            # single-hand only, correct label, 21 landmarks present
            if len(labels) != 1 or len(lms) != 1 or len(bboxes) != 1:
                continue
            if labels[0] != cls or len(lms[0]) != NUM_LANDMARKS:
                continue
            cand.append((image_id, bboxes[0], lms[0]))
        if cap and len(cand) > cap:
            idx = rng.choice(len(cand), size=cap, replace=False)
            cand = [cand[i] for i in idx]
        for image_id, bbox, lm in cand:
            rows.append({
                "split": split_dir.name,
                "label": cls,
                "image_id": image_id,
                "zip_path": f"{ZIP_PREFIX}/{cls}/{image_id}.jpg",
                "bbox": np.asarray(bbox, np.float32).tolist(),
                "landmarks": np.asarray(lm, np.float32).reshape(-1).tolist(),  # 42
            })
        print(f"  [{split_dir.name}/{cls}] candidates kept: {min(len(cand),cap) if cap else len(cand)}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default=str(ANN_DEFAULT))
    ap.add_argument("--out", default=str(ROOT / "cache_v2/manifest.parquet"))
    ap.add_argument("--cap-train", type=int, default=3000)
    ap.add_argument("--cap-val", type=int, default=500)
    ap.add_argument("--cap-test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    ann = Path(args.ann)
    classes = selected_classes()
    rng = np.random.default_rng(args.seed)
    caps = {"train": args.cap_train, "val": args.cap_val, "test": args.cap_test}
    print(f"[info] {len(classes)} classes, caps={caps}")

    rows = []
    for split in SPLITS:
        print(f"[{split}] sampling (cap={caps[split]})...")
        rows += sample_split(ann / split, classes, caps[split], rng)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"\n[done] manifest: {len(df)} samples -> {out}")
    print(df.groupby("split").size().to_dict())
    print("per-split-class min/max:",
          df.groupby(["split", "label"]).size().groupby("split").agg(["min", "max"]).to_dict())


if __name__ == "__main__":
    main()
