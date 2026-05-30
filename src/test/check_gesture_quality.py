from pathlib import Path
import csv
import numpy as np


ANNOTATION_PATH = Path("data/annotations/jester-v1-validation.csv")
PROCESSED_DIR = Path("data/processed/validation")
TARGET_LABEL = "Doing other things"


def read_rows(path: Path):
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
    rows = read_rows(ANNOTATION_PATH)

    total_samples = 0
    all_zero_samples = 0
    frame_ratios = []

    for video_id, label in rows:
        if label != TARGET_LABEL:
            continue

        path = PROCESSED_DIR / f"{video_id}.npy"

        if not path.exists():
            continue

        x = np.load(path)

        detected_frames = int((np.abs(x).sum(axis=1) > 0).sum())
        ratio = detected_frames / x.shape[0]

        total_samples += 1
        frame_ratios.append(ratio)

        if detected_frames == 0:
            all_zero_samples += 1

    print("label:", TARGET_LABEL)
    print("samples:", total_samples)
    print("all-zero samples:", all_zero_samples)

    if total_samples > 0:
        print("all-zero sample ratio:", all_zero_samples / total_samples)
        print("average detected-frame ratio:", float(np.mean(frame_ratios)))
        print("minimum detected-frame ratio:", float(np.min(frame_ratios)))
        print("maximum detected-frame ratio:", float(np.max(frame_ratios)))


if __name__ == "__main__":
    main()