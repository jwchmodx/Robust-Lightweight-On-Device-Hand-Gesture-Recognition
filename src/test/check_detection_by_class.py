from pathlib import Path
import csv
from collections import defaultdict

import numpy as np


ANNOTATION_PATH = Path("data/annotations/jester-v1-validation.csv")
PROCESSED_DIR = Path("data/processed/validation")

TARGET_LABELS = [
    "No gesture",
    "Doing other things",
    "Swiping Up",
    "Swiping Down",
    "Swiping Left",
    "Swiping Right",
    "Stop Sign",
]


def read_annotations(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        sample = f.readline()
        delimiter = ";" if ";" in sample else ","
        f.seek(0)

        reader = csv.reader(f, delimiter=delimiter)

        for row in reader:
            if len(row) >= 2:
                rows.append((row[0].strip(), row[1].strip()))

    return rows


def main():
    rows = read_annotations(ANNOTATION_PATH)

    stats = defaultdict(lambda: {
        "samples": 0,
        "all_zero": 0,
        "ratios": [],
    })

    for video_id, label in rows:
        if label not in TARGET_LABELS:
            continue

        npy_path = PROCESSED_DIR / f"{video_id}.npy"

        if not npy_path.exists():
            print("missing:", npy_path)
            continue

        x = np.load(npy_path)

        detected_frames = int((np.abs(x).sum(axis=1) > 0).sum())
        detected_ratio = detected_frames / x.shape[0]

        stats[label]["samples"] += 1
        stats[label]["ratios"].append(detected_ratio)

        if detected_frames == 0:
            stats[label]["all_zero"] += 1

    print(
        f"{'Label':<22}"
        f"{'Samples':>9}"
        f"{'All Zero':>11}"
        f"{'Zero Ratio':>13}"
        f"{'Avg Detect':>13}"
        f"{'Min':>9}"
        f"{'Max':>9}"
    )

    print("-" * 86)

    for label in TARGET_LABELS:
        s = stats[label]

        ratios = np.array(s["ratios"], dtype=np.float32)
        samples = s["samples"]
        all_zero = s["all_zero"]

        zero_ratio = all_zero / samples if samples else 0.0

        print(
            f"{label:<22}"
            f"{samples:>9}"
            f"{all_zero:>11}"
            f"{zero_ratio:>13.4f}"
            f"{ratios.mean():>13.4f}"
            f"{ratios.min():>9.4f}"
            f"{ratios.max():>9.4f}"
        )


if __name__ == "__main__":
    main()