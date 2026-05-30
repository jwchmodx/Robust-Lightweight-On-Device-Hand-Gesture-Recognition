from pathlib import Path
import csv
import numpy as np
from tqdm import tqdm


TARGET_LEN = 32

DATA_DIR = Path("data/processed/train")
ANNOTATION_PATH = Path("data/annotations/jester-v1-train.csv")
OUT_X = DATA_DIR / "X_train.npy"
OUT_Y = DATA_DIR / "y_train.npy"


def read_annotations(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        sample = f.readline()
        delimiter = ";" if ";" in sample else ","
        f.seek(0)

        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if len(row) >= 2:
                rows.append((row[0].strip(), row[1].strip()))

    labels = sorted(set(label for _, label in rows))
    label_map = {label: i for i, label in enumerate(labels)}
    return rows, label_map


def normalize_sequence(x, target_len=32):
    T = x.shape[0]

    if T == target_len:
        return x

    if T > target_len:
        indices = np.linspace(0, T - 1, target_len).astype(int)
        return x[indices]

    pad = np.zeros((target_len - T, x.shape[1]), dtype=x.dtype)
    return np.vstack([x, pad])


def main():
    rows, label_map = read_annotations(ANNOTATION_PATH)

    X = []
    y = []
    skipped = 0

    for video_id, label in tqdm(rows):
        npy_path = DATA_DIR / f"{video_id}.npy"

        if not npy_path.exists():
            skipped += 1
            continue

        x = np.load(npy_path)

        if x.shape[0] < 10:
            skipped += 1
            continue

        x = normalize_sequence(x, TARGET_LEN)

        X.append(x)
        y.append(label_map[label])

    if len(X) == 0:
        raise RuntimeError("No samples found. Check DATA_DIR and annotation file paths.")

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)

    np.save(OUT_X, X)
    np.save(OUT_Y, y)

    print("X:", X.shape)
    print("y:", y.shape)
    print("skipped:", skipped)
    print("saved:", OUT_X)
    print("saved:", OUT_Y)


if __name__ == "__main__":
    main()