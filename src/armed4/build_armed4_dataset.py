from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


TARGET_LABELS = [
    "Doing other things",
    "Swiping Up",
    "Swiping Down",
    "Stop Sign",
]

LABEL_MAP = {
    "Doing other things": 0,
    "Swiping Up": 1,
    "Swiping Down": 2,
    "Stop Sign": 3,
}


def read_annotations(csv_path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []

    with csv_path.open("r", encoding="utf-8") as f:
        sample = f.readline()
        delimiter = ";" if ";" in sample else ","
        f.seek(0)

        reader = csv.reader(f, delimiter=delimiter)

        for row in reader:
            if len(row) < 2:
                continue

            video_id = row[0].strip()
            label = row[1].strip()

            if label in TARGET_LABELS:
                rows.append((video_id, label))

    return rows


def longest_run(mask: np.ndarray) -> int:
    best = 0
    current = 0

    for detected in mask:
        if detected:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return best


def interpolate_missing_frames(sequence: np.ndarray, detected_mask: np.ndarray) -> np.ndarray:
    """
    crop된 구간 내부에서 MediaPipe가 놓친 프레임을
    앞뒤 검출 landmark로 선형 보간한다.

    sequence: (T, 63)
    detected_mask: (T,)
    """
    detected_indices = np.where(detected_mask)[0]

    if len(detected_indices) < 2:
        return sequence.astype(np.float32)

    frame_indices = np.arange(sequence.shape[0])
    filled = sequence.copy().astype(np.float32)

    for feature_index in range(sequence.shape[1]):
        filled[:, feature_index] = np.interp(
            frame_indices,
            detected_indices,
            sequence[detected_indices, feature_index],
        )

    return filled


def temporal_resample(sequence: np.ndarray, target_len: int) -> np.ndarray:
    """
    가변 길이 landmark sequence를 target_len 길이로 시간축 리샘플링한다.
    """
    original_len, feature_dim = sequence.shape

    if original_len == target_len:
        return sequence.astype(np.float32)

    if original_len == 1:
        return np.repeat(sequence, target_len, axis=0).astype(np.float32)

    old_t = np.linspace(0.0, 1.0, original_len)
    new_t = np.linspace(0.0, 1.0, target_len)

    resized = np.zeros((target_len, feature_dim), dtype=np.float32)

    for feature_index in range(feature_dim):
        resized[:, feature_index] = np.interp(
            new_t,
            old_t,
            sequence[:, feature_index],
        )

    return resized


def process_sequence(
    sequence: np.ndarray,
    target_len: int,
    min_detected_frames: int,
    min_longest_run: int,
    min_active_density: float,
) -> tuple[np.ndarray | None, dict]:
    """
    하나의 landmark sequence를 armed interaction용 입력으로 변환한다.
    """
    if sequence.ndim != 2 or sequence.shape[1] != 63:
        return None, {
            "reason": "invalid_shape",
            "shape": list(sequence.shape),
        }

    detected_mask = np.abs(sequence).sum(axis=1) > 0
    detected_indices = np.where(detected_mask)[0]

    detected_frames = int(detected_mask.sum())

    if detected_frames == 0:
        return None, {
            "reason": "no_detected_frames",
            "detected_frames": 0,
        }

    start = int(detected_indices[0])
    end = int(detected_indices[-1])

    active_sequence = sequence[start:end + 1]
    active_mask = detected_mask[start:end + 1]

    active_span = int(len(active_sequence))
    active_density = detected_frames / active_span
    run = longest_run(active_mask)

    stats = {
        "original_frames": int(sequence.shape[0]),
        "detected_frames": detected_frames,
        "active_start": start,
        "active_end": end,
        "active_span": active_span,
        "active_density": float(active_density),
        "longest_run": run,
    }

    if detected_frames < min_detected_frames:
        stats["reason"] = "too_few_detected_frames"
        return None, stats

    if run < min_longest_run:
        stats["reason"] = "too_short_continuous_run"
        return None, stats

    if active_density < min_active_density:
        stats["reason"] = "active_density_too_low"
        return None, stats

    # active 구간 내부의 누락 frame만 보간
    filled_sequence = interpolate_missing_frames(
        active_sequence,
        active_mask,
    )

    # 실제 동작 구간만 고정 길이로 변환
    normalized_sequence = temporal_resample(
        filled_sequence,
        target_len,
    )

    stats["reason"] = "kept"

    return normalized_sequence, stats


def build_split(
    split_name: str,
    annotation_path: Path,
    processed_dir: Path,
    output_dir: Path,
    target_len: int,
    min_detected_frames: int,
    min_longest_run: int,
    min_active_density: float,
) -> dict:
    rows = read_annotations(annotation_path)

    X: list[np.ndarray] = []
    y: list[int] = []

    kept_metadata = []
    skipped_metadata = []

    original_counts = Counter(label for _, label in rows)
    kept_counts = Counter()

    for video_id, label in tqdm(rows, desc=f"build {split_name}"):
        npy_path = processed_dir / f"{video_id}.npy"

        if not npy_path.exists():
            skipped_metadata.append({
                "video_id": video_id,
                "label": label,
                "reason": "file_not_found",
                "path": str(npy_path),
            })
            continue

        sequence = np.load(npy_path)

        normalized, stats = process_sequence(
            sequence=sequence,
            target_len=target_len,
            min_detected_frames=min_detected_frames,
            min_longest_run=min_longest_run,
            min_active_density=min_active_density,
        )

        item = {
            "video_id": video_id,
            "label": label,
            "label_id": LABEL_MAP[label],
            "source_path": str(npy_path),
            **stats,
        }

        if normalized is None:
            skipped_metadata.append(item)
            continue

        X.append(normalized)
        y.append(LABEL_MAP[label])

        kept_metadata.append(item)
        kept_counts[label] += 1

    if not X:
        raise RuntimeError(
            f"No usable samples found for {split_name}. "
            f"Check processed_dir: {processed_dir}"
        )

    X_array = np.stack(X).astype(np.float32)
    y_array = np.array(y, dtype=np.int64)

    output_dir.mkdir(parents=True, exist_ok=True)

    x_path = output_dir / f"X_{split_name}.npy"
    y_path = output_dir / f"y_{split_name}.npy"

    np.save(x_path, X_array)
    np.save(y_path, y_array)

    with (output_dir / f"{split_name}_kept_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(kept_metadata, f, ensure_ascii=False, indent=2)

    with (output_dir / f"{split_name}_skipped_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(skipped_metadata, f, ensure_ascii=False, indent=2)

    summary = {
        "split": split_name,
        "X_shape": list(X_array.shape),
        "y_shape": list(y_array.shape),
        "original_counts": dict(original_counts),
        "kept_counts": dict(kept_counts),
        "skipped_count": len(skipped_metadata),
        "target_len": target_len,
        "filters": {
            "min_detected_frames": min_detected_frames,
            "min_longest_run": min_longest_run,
            "min_active_density": min_active_density,
        },
    }

    print(f"\n=== {split_name.upper()} ===")
    print("X shape:", X_array.shape)
    print("y shape:", y_array.shape)

    for label in TARGET_LABELS:
        original = original_counts[label]
        kept = kept_counts[label]
        keep_rate = kept / original if original else 0.0

        print(
            f"{label:<22} "
            f"original={original:>6} "
            f"kept={kept:>6} "
            f"rate={keep_rate:.4f}"
        )

    print("skipped:", len(skipped_metadata))
    print("saved:", x_path)
    print("saved:", y_path)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-processed",
        default="data/processed_tracking/train",
        help="Tracking 설정으로 생성한 train 개별 npy 폴더",
    )

    parser.add_argument(
        "--val-processed",
        default="data/processed_tracking/validation",
        help="Tracking 설정으로 생성한 validation 개별 npy 폴더",
    )

    parser.add_argument(
        "--train-annotation",
        default="data/annotations/jester-v1-train.csv",
    )

    parser.add_argument(
        "--val-annotation",
        default="data/annotations/jester-v1-validation.csv",
    )

    parser.add_argument(
        "--out-root",
        default="data/model_ready/armed4_24f",
    )

    parser.add_argument(
        "--target-len",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--min-detected-frames",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--min-longest-run",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--min-active-density",
        type=float,
        default=0.40,
    )

    args = parser.parse_args()

    output_dir = Path(args.out_root)

    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, ensure_ascii=False, indent=2)

    train_summary = build_split(
        split_name="train",
        annotation_path=Path(args.train_annotation),
        processed_dir=Path(args.train_processed),
        output_dir=output_dir,
        target_len=args.target_len,
        min_detected_frames=args.min_detected_frames,
        min_longest_run=args.min_longest_run,
        min_active_density=args.min_active_density,
    )

    val_summary = build_split(
        split_name="val",
        annotation_path=Path(args.val_annotation),
        processed_dir=Path(args.val_processed),
        output_dir=output_dir,
        target_len=args.target_len,
        min_detected_frames=args.min_detected_frames,
        min_longest_run=args.min_longest_run,
        min_active_density=args.min_active_density,
    )

    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": TARGET_LABELS,
                "label_map": LABEL_MAP,
                "train": train_summary,
                "val": val_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== DATASET FINISHED ===")
    print("output directory:", output_dir)
    print("labels:", LABEL_MAP)


if __name__ == "__main__":
    main()