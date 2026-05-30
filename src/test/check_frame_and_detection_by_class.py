from pathlib import Path
import csv
from collections import defaultdict

import numpy as np


ANNOTATION_PATH = Path("data/annotations/jester-v1-validation.csv")

# 새 전처리 데이터를 검사할 경로
PROCESSED_DIR = Path("data/processed_tracking_test/validation")

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
        "lengths": [],
        "detected": [],
        "ratios": [],
    })

    for video_id, label in rows:
        if label not in TARGET_LABELS:
            continue

        npy_path = PROCESSED_DIR / f"{video_id}.npy"

        if not npy_path.exists():
            continue

        x = np.load(npy_path)

        total_frames = x.shape[0]
        detected_frames = int((np.abs(x).sum(axis=1) > 0).sum())
        ratio = detected_frames / total_frames

        stats[label]["lengths"].append(total_frames)
        stats[label]["detected"].append(detected_frames)
        stats[label]["ratios"].append(ratio)

    print(
        f"{'Label':<22}"
        f"{'Samples':>9}"
        f"{'Avg Frames':>12}"
        f"{'Min F':>8}"
        f"{'Max F':>8}"
        f"{'Avg Detected':>14}"
        f"{'Avg Ratio':>12}"
    )
    print("-" * 85)

    for label in TARGET_LABELS:
        lengths = np.array(stats[label]["lengths"])
        detected = np.array(stats[label]["detected"])
        ratios = np.array(stats[label]["ratios"])

        if len(lengths) == 0:
            print(f"{label:<22}{'no data':>9}")
            continue

        print(
            f"{label:<22}"
            f"{len(lengths):>9}"
            f"{lengths.mean():>12.2f}"
            f"{lengths.min():>8}"
            f"{lengths.max():>8}"
            f"{detected.mean():>14.2f}"
            f"{ratios.mean():>12.4f}"
        )


if __name__ == "__main__":
    main()