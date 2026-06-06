"""On-device proxy metrics for the lightweight story: params, FLOPs, model size.

Compares the student variants against the MediaPipe baseline (published figures:
palm detector ~1.76M params, hand-landmark model ~2.0M params).
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student"))
from model import HandLandmarkNet, count_params   # noqa: E402

# MediaPipe Hands published sizes (params)
MP_PALM = 1_760_000
MP_LANDMARK = 2_000_000


def flops(model, size=128):
    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(model, (3, size, size),
                                            as_strings=False, print_per_layer_stat=False,
                                            verbose=False)
        return macs  # MACs
    except Exception as e:
        return None


def main():
    rows = []
    for width in (1.0, 3.0):
        m = HandLandmarkNet(width=width).eval()
        p = count_params(m)
        macs = flops(m)
        size_mb = p * 4 / 1e6  # fp32
        rows.append({"model": f"student_w{width:g}", "params": p,
                     "MMACs": round(macs / 1e6, 1) if macs else None,
                     "size_fp32_MB": round(size_mb, 2),
                     "size_int8_MB": round(p / 1e6, 2)})
    # MediaPipe references
    rows.append({"model": "MediaPipe landmark-only", "params": MP_LANDMARK,
                 "MMACs": None, "size_fp32_MB": round(MP_LANDMARK * 4 / 1e6, 2), "size_int8_MB": round(MP_LANDMARK/1e6,2)})
    rows.append({"model": "MediaPipe full (palm+landmark)", "params": MP_PALM + MP_LANDMARK,
                 "MMACs": None, "size_fp32_MB": round((MP_PALM+MP_LANDMARK)*4/1e6, 2), "size_int8_MB": round((MP_PALM+MP_LANDMARK)/1e6,2)})

    import pandas as pd
    df = pd.DataFrame(rows)
    df["params_vs_MP_landmark"] = (MP_LANDMARK / df["params"]).round(1).astype(str) + "x"
    out = ROOT / "cache_v2/proxy_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\n[done] -> {out}")
    print(f"\nstudent_w3 is {MP_LANDMARK/count_params(HandLandmarkNet(width=3.0)):.1f}x smaller than "
          f"MediaPipe's landmark model, {(MP_PALM+MP_LANDMARK)/count_params(HandLandmarkNet(width=3.0)):.1f}x "
          f"smaller than the full pipeline.")


if __name__ == "__main__":
    main()
