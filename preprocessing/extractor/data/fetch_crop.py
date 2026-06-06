"""Parallel range-fetch HaGRIDv2 512p images from the remote 128GB zip, crop the
hand region (from bbox), and store compact JPEG crops + landmark targets.

Throughput trick: the server has ~1.8s round-trip latency but allows ranged GETs,
so we fan out many concurrent 1-RTT fetches (parse the zip local header + data in
a single ranged request).

Outputs:
  cache_v2/crops/<split>/<label>/<image_id>.jpg   square crops (resized)
  cache_v2/targets.parquet                         image_id, split, label,
                                                   landmarks_crop(42), crop_to_img(4)

crop_to_img = [ox, oy, sx, sy]: maps crop-frame (u,v) in [0,1] back to ORIGINAL
image-normalized coords  ->  x = ox + u*sx,  y = oy + v*sy  (sx != sy preserves
the image aspect ratio the frozen MLP was trained on).
"""
import argparse
import io
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parents[1]
ZIP_URL = ("https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/"
           "hagrid_v2/hagridv2_512.zip")

_tls = threading.local()


def _session():
    if not hasattr(_tls, "s"):
        _tls.s = requests.Session()
    return _tls.s


def fetch_jpg(url, info, margin=300):
    """1-RTT ranged fetch of one zip member -> raw jpg bytes."""
    off = info.header_offset
    want = 30 + len(info.filename.encode()) + margin + info.compress_size
    hdr = {"Range": f"bytes={off}-{off + want - 1}"}
    raw = _session().get(url, headers=hdr, timeout=60).content
    name_len, extra_len = struct.unpack("<HH", raw[26:30])
    start = 30 + name_len + extra_len
    data = raw[start:start + info.compress_size]
    if len(data) < info.compress_size:  # margin too small: exact refetch
        h2 = {"Range": f"bytes={off + start}-{off + start + info.compress_size - 1}"}
        data = _session().get(url, headers=h2, timeout=60).content
    if info.compress_type == 0:
        return data
    return zlib.decompress(data, -15)  # deflate


def crop_and_targets(img_bgr, bbox, lm42, out_size, pad):
    """bbox/lm normalized to image. Returns (crop_bgr, lm_crop(21,2), affine[4])."""
    H, W = img_bgr.shape[:2]
    lm = np.asarray(lm42, np.float32).reshape(21, 2)
    bx, by, bw, bh = bbox
    # square box around bbox center, padded, in pixels
    cx, cy = (bx + bw / 2) * W, (by + bh / 2) * H
    side = max(bw * W, bh * H) * pad
    x0, y0 = cx - side / 2, cy - side / 2
    # crop (clamped); keep intended box for coord math even if clamped
    xi0, yi0 = int(round(x0)), int(round(y0))
    xi1, yi1 = int(round(x0 + side)), int(round(y0 + side))
    cxi0, cyi0 = max(0, xi0), max(0, yi0)
    cxi1, cyi1 = min(W, xi1), min(H, yi1)
    patch = img_bgr[cyi0:cyi1, cxi0:cxi1]
    if patch.size == 0:
        return None
    canvas = np.zeros((yi1 - yi0, xi1 - xi0, 3), np.uint8)  # pad region for OOB
    canvas[cyi0 - yi0:cyi1 - yi0, cxi0 - xi0:cxi1 - xi0] = patch
    crop = cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_AREA)
    # landmarks -> crop frame [0,1]
    lm_px = lm * np.array([W, H], np.float32)
    lm_crop = (lm_px - np.array([xi0, yi0], np.float32)) / side
    # affine crop->image-normalized: x = ox + u*sx ; sx = side/W, sy = side/H
    affine = np.array([xi0 / W, yi0 / H, side / W, side / H], np.float32)
    return crop, lm_crop.astype(np.float32), affine


def process_row(args_tuple):
    url, info, row, out_dir, out_size, pad = args_tuple
    try:
        jpg = fetch_jpg(url, info)
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        res = crop_and_targets(img, row["bbox"], row["landmarks"], out_size, pad)
        if res is None:
            return None
        crop, lm_crop, affine = res
        p = out_dir / "crops" / row["split"] / row["label"]
        p.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p / f"{row['image_id']}.jpg"), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        return {"image_id": row["image_id"], "split": row["split"], "label": row["label"],
                "landmarks_crop": lm_crop.reshape(-1).tolist(), "crop_to_img": affine.tolist()}
    except Exception as e:
        return {"error": str(e), "image_id": row.get("image_id")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "cache_v2/manifest.parquet"))
    ap.add_argument("--out", default=str(ROOT / "cache_v2"))
    ap.add_argument("--zip-url", default=ZIP_URL)
    ap.add_argument("--crop-size", type=int, default=128)
    ap.add_argument("--pad", type=float, default=1.3)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0, help="0 = all (debug small runs)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    df = pd.read_parquet(args.manifest)
    if args.limit:
        df = df.groupby("label", group_keys=False).head(max(1, args.limit // 26))
    print(f"[info] fetching {len(df)} crops, {args.workers} workers")

    print("[info] loading remote zip central directory (~40s)...")
    with RemoteZip(args.zip_url) as z:
        info_map = {i.filename: i for i in z.infolist() if i.filename.endswith(".jpg")}
    print(f"[info] central dir: {len(info_map)} jpgs indexed")

    tasks = []
    for _, row in df.iterrows():
        info = info_map.get(row["zip_path"])
        if info is None:
            continue
        tasks.append((args.zip_url, info, row, out_dir, args.crop_size, args.pad))
    print(f"[info] matched {len(tasks)}/{len(df)} paths in zip")

    import time
    from tqdm import tqdm
    t0 = time.time()
    results, errors = [], 0
    ckpt = out_dir / "targets.partial.parquet"
    with ThreadPoolExecutor(args.workers) as ex:
        for r in tqdm(ex.map(process_row, tasks), total=len(tasks)):
            if r is None or "error" in r:
                errors += 1
            else:
                results.append(r)
            if len(results) % 10000 == 0 and len(results) > 0:
                pd.DataFrame(results).to_parquet(ckpt)   # crash-safe checkpoint
    dt = time.time() - t0

    meta = pd.DataFrame(results)
    meta.to_parquet(out_dir / "targets.parquet")
    print(f"\n[done] {len(results)} crops in {dt/60:.1f} min "
          f"({len(results)/dt:.1f} img/s), errors={errors}")
    print(meta.groupby("split").size().to_dict() if len(meta) else "no results")


if __name__ == "__main__":
    main()
