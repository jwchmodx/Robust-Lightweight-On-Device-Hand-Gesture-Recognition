from pathlib import Path
import csv
from collections import defaultdict

import numpy as np


ANNOTATION_PATH = Path("data/annotations/jester-v1-validation.csv")
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

MIN_DETECTED_FRAMES = 6
MIN_DETECTION_RATIO = 0.20
MIN_LONGEST_RUN = 4


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


def longest_run(mask: np.ndarray) -> int:
    best = 0
    current = 0

    for value in mask:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return best


def main():
    rows = read_annotations(ANNOTATION_PATH)

    stats = defaultdict(lambda: {
        "total": 0,
        "usable": 0,
        "detected_counts": [],
        "ratios": [],
        "runs": [],
    })

    for video_id, label in rows:
        if label not in TARGET_LABELS:
            continue

        path = PROCESSED_DIR / f"{video_id}.npy"

        if not path.exists():
            continue

        x = np.load(path)
        detected = np.abs(x).sum(axis=1) > 0

        detected_count = int(detected.sum())
        ratio = detected_count / len(detected)
        run = longest_run(detected)

        usable = (
            detected_count >= MIN_DETECTED_FRAMES
            and ratio >= MIN_DETECTION_RATIO
            and run >= MIN_LONGEST_RUN
        )

        stats[label]["total"] += 1
        stats[label]["usable"] += int(usable)
        stats[label]["detected_counts"].append(detected_count)
        stats[label]["ratios"].append(ratio)
        stats[label]["runs"].append(run)

    print(
        f"{'Label':<22}"
        f"{'Total':>8}"
        f"{'Usable':>9}"
        f"{'Keep Rate':>12}"
        f"{'Avg Det':>10}"
        f"{'Avg Ratio':>12}"
        f"{'Avg Run':>10}"
    )
    print("-" * 83)

    for label in TARGET_LABELS:
        s = stats[label]

        if s["total"] == 0:
            continue

        keep_rate = s["usable"] / s["total"]

        print(
            f"{label:<22}"
            f"{s['total']:>8}"
            f"{s['usable']:>9}"
            f"{keep_rate:>12.4f}"
            f"{np.mean(s['detected_counts']):>10.2f}"
            f"{np.mean(s['ratios']):>12.4f}"
            f"{np.mean(s['runs']):>10.2f}"
        )


if __name__ == "__main__":
    main()