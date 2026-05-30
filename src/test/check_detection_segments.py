from pathlib import Path
import csv
import random
import numpy as np


ANNOTATION_PATH = Path("data/annotations/jester-v1-validation.csv")
PROCESSED_DIR = Path("data/processed_tracking_test/validation")

TARGET_LABELS = [
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
    random.seed(42)
    rows = read_annotations(ANNOTATION_PATH)

    for label in TARGET_LABELS:
        candidates = [
            video_id for video_id, row_label in rows
            if row_label == label
        ]

        sampled = random.sample(candidates, min(5, len(candidates)))

        print(f"\n=== {label} ===")

        for video_id in sampled:
            x = np.load(PROCESSED_DIR / f"{video_id}.npy")
            detected = np.abs(x).sum(axis=1) > 0

            pattern = "".join("■" if value else "·" for value in detected)

            detected_indices = np.where(detected)[0]

            if len(detected_indices) > 0:
                start = int(detected_indices[0])
                end = int(detected_indices[-1])
                span = end - start + 1
            else:
                start = -1
                end = -1
                span = 0

            print(
                f"{video_id:<8} "
                f"det={detected.sum():>2}/{len(detected):<2} "
                f"span={span:<2} "
                f"longest={longest_run(detected):<2} "
                f"{pattern}"
            )


if __name__ == "__main__":
    main()