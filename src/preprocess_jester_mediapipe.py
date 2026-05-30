
"""
python src/preprocess_jester_mediapipe.py \
  --data-root data/20bn-jester-v1 \
  --annotation data/annotations/jester-v1-validation.csv \
  --out-root data/processed_tracking_test/validation \
  --workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ============================================================
# Worker process global objects
# ============================================================
WORKER_HANDS = None
WORKER_MP_HANDS = None

WORKER_MODEL_COMPLEXITY = 1
WORKER_MIN_DETECTION_CONFIDENCE = 0.3
WORKER_MIN_TRACKING_CONFIDENCE = 0.3
WORKER_RESIZE_SCALE = 2.0


def create_hands_model():
    """
    현재 worker 설정으로 MediaPipe Hands 객체를 생성한다.
    static_image_mode=False이므로 영상 내 프레임 추적을 사용한다.
    """
    return WORKER_MP_HANDS.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=WORKER_MODEL_COMPLEXITY,
        min_detection_confidence=WORKER_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=WORKER_MIN_TRACKING_CONFIDENCE,
    )


def initialize_worker(
    model_complexity: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    resize_scale: float,
) -> None:
    """
    각 multiprocessing worker가 시작될 때 한 번만 실행된다.

    - MediaPipe 로그는 worker 내부 stderr만 차단해서 숨긴다.
    - tqdm은 main process에서 출력하므로 그대로 유지된다.
    - Hands 모델은 worker당 한 번만 로드한다.
    """
    global WORKER_HANDS
    global WORKER_MP_HANDS
    global WORKER_MODEL_COMPLEXITY
    global WORKER_MIN_DETECTION_CONFIDENCE
    global WORKER_MIN_TRACKING_CONFIDENCE
    global WORKER_RESIZE_SCALE

    # MediaPipe / TFLite 로그 억제
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

    # worker에서 발생하는 native stderr 로그만 숨김
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)

    # stderr 차단 이후 import해야 초기화 로그도 화면에 나오지 않음
    import mediapipe as mp

    WORKER_MP_HANDS = mp.solutions.hands
    WORKER_MODEL_COMPLEXITY = model_complexity
    WORKER_MIN_DETECTION_CONFIDENCE = min_detection_confidence
    WORKER_MIN_TRACKING_CONFIDENCE = min_tracking_confidence
    WORKER_RESIZE_SCALE = resize_scale

    WORKER_HANDS = create_hands_model()


# ============================================================
# Annotation
# ============================================================
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
            rows.append((video_id, label))

    return rows


# ============================================================
# Frame / sequence processing
# ============================================================
def get_frame_paths(video_dir: Path) -> list[Path]:
    frame_paths = (
        list(video_dir.glob("*.jpg"))
        + list(video_dir.glob("*.jpeg"))
        + list(video_dir.glob("*.png"))
    )

    # Jester 프레임 이름이 숫자일 때 1, 2, 10 순으로 안전하게 정렬
    def sort_key(path: Path):
        try:
            return int(path.stem)
        except ValueError:
            return path.stem

    return sorted(frame_paths, key=sort_key)


def reset_worker_hands_model() -> None:
    """
    static_image_mode=False에서는 이전 영상의 tracking state가
    다음 영상으로 넘어가면 안 되므로 영상마다 reset한다.
    """
    global WORKER_HANDS

    if WORKER_HANDS is None:
        WORKER_HANDS = create_hands_model()
        return

    if hasattr(WORKER_HANDS, "reset"):
        WORKER_HANDS.reset()
    else:
        # reset()이 없는 버전에 대한 안전한 fallback
        WORKER_HANDS.close()
        WORKER_HANDS = create_hands_model()


def process_one_folder(video_dir: Path, out_path: Path) -> dict:
    """
    영상 폴더 하나를 읽어서 하나의 npy 파일로 저장한다.

    input:
        video_dir/1.jpg, 2.jpg, ...

    output:
        out_path -> shape (T, 63)
    """
    global WORKER_HANDS
    global WORKER_RESIZE_SCALE

    frame_paths = get_frame_paths(video_dir)

    if not frame_paths:
        return {
            "status": "error",
            "error": "no_frames",
            "video_dir": str(video_dir),
        }

    reset_worker_hands_model()

    sequence: list[np.ndarray] = []

    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path))

        if image is None:
            sequence.append(np.zeros(63, dtype=np.float32))
            continue

        if WORKER_RESIZE_SCALE != 1.0:
            image = cv2.resize(
                image,
                None,
                fx=WORKER_RESIZE_SCALE,
                fy=WORKER_RESIZE_SCALE,
                interpolation=cv2.INTER_CUBIC,
            )

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = WORKER_HANDS.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            landmarks = np.array(
                [
                    coordinate
                    for lm in hand_landmarks.landmark
                    for coordinate in (lm.x, lm.y, lm.z)
                ],
                dtype=np.float32,
            )

            sequence.append(landmarks)

        else:
            sequence.append(np.zeros(63, dtype=np.float32))

    sequence_array = np.stack(sequence).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, sequence_array)

    detected_frames = int(
        (np.abs(sequence_array).sum(axis=1) > 0).sum()
    )

    return {
        "status": "ok",
        "path": str(out_path),
        "frames": int(sequence_array.shape[0]),
        "detected_frames": detected_frames,
        "detection_ratio": detected_frames / sequence_array.shape[0],
    }


# ============================================================
# Multiprocessing worker
# ============================================================
def worker(task: tuple[str, str, Path, Path, bool]) -> dict:
    video_id, label, data_root, out_root, overwrite = task

    video_dir = data_root / video_id
    out_path = out_root / f"{video_id}.npy"

    if not video_dir.exists():
        return {
            "status": "error",
            "video_id": video_id,
            "label": label,
            "error": "folder_not_found",
            "video_dir": str(video_dir),
        }

    if out_path.exists() and not overwrite:
        try:
            sequence = np.load(out_path, mmap_mode="r")

            detected_frames = int(
                (np.abs(sequence).sum(axis=1) > 0).sum()
            )

            return {
                "status": "skipped_exists",
                "video_id": video_id,
                "label": label,
                "path": str(out_path),
                "frames": int(sequence.shape[0]),
                "detected_frames": detected_frames,
                "detection_ratio": detected_frames / sequence.shape[0],
            }

        except Exception as e:
            return {
                "status": "error",
                "video_id": video_id,
                "label": label,
                "error": f"failed_to_read_existing_file: {e}",
                "path": str(out_path),
            }

    try:
        result = process_one_folder(video_dir, out_path)

        result["video_id"] = video_id
        result["label"] = label

        return result

    except Exception as e:
        return {
            "status": "error",
            "video_id": video_id,
            "label": label,
            "error": str(e),
            "video_dir": str(video_dir),
        }


# ============================================================
# Main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=str,
        default="data/20bn-jester-v1",
    )

    parser.add_argument(
        "--annotation",
        type=str,
        default="data/annotations/jester-v1-train.csv",
    )

    parser.add_argument(
        "--out-root",
        type=str,
        default="data/processed/train",
    )

    parser.add_argument(
        "--max-num",
        type=int,
        default=None,
        help="테스트용 처리 개수. 예: --max-num 100",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 존재하는 npy 파일도 다시 생성",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, cpu_count() - 2)),
    )

    parser.add_argument(
        "--model-complexity",
        type=int,
        default=1,
        choices=[0, 1],
    )

    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--resize-scale",
        type=float,
        default=2.0,
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    annotation_path = Path(args.annotation)
    out_root = Path(args.out_root)

    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")

    if not annotation_path.exists():
        raise FileNotFoundError(f"annotation not found: {annotation_path}")

    rows = read_annotations(annotation_path)

    if args.max_num is not None:
        rows = rows[: args.max_num]

    label_names = sorted({label for _, label in rows})
    label_map = {
        label: index for index, label in enumerate(label_names)
    }

    out_root.mkdir(parents=True, exist_ok=True)

    with (out_root / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print("data_root:", data_root.resolve())
    print("annotation:", annotation_path.resolve())
    print("out_root:", out_root.resolve())
    print("total:", len(rows))
    print("workers:", args.workers)
    print("static_image_mode: False")
    print("model_complexity:", args.model_complexity)
    print("min_detection_confidence:", args.min_detection_confidence)
    print("min_tracking_confidence:", args.min_tracking_confidence)
    print("resize_scale:", args.resize_scale)

    tasks = [
        (video_id, label, data_root, out_root, args.overwrite)
        for video_id, label in rows
    ]

    metadata: list[dict] = []
    errors: list[dict] = []

    with Pool(
        processes=args.workers,
        initializer=initialize_worker,
        initargs=(
            args.model_complexity,
            args.min_detection_confidence,
            args.min_tracking_confidence,
            args.resize_scale,
        ),
    ) as pool:
        iterator = pool.imap_unordered(worker, tasks, chunksize=4)

        for result in tqdm(
            iterator,
            total=len(tasks),
            desc="preprocess",
            file=sys.stdout,
        ):
            if result["status"] in {"ok", "skipped_exists"}:
                result["label_id"] = label_map[result["label"]]
                metadata.append(result)
            else:
                errors.append(result)

    # 순서가 흔들리지 않게 video id 숫자 기준 정렬
    def metadata_sort_key(item: dict):
        try:
            return int(item["video_id"])
        except ValueError:
            return item["video_id"]

    metadata.sort(key=metadata_sort_key)
    errors.sort(key=metadata_sort_key)

    with (out_root / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with (out_root / "errors.json").open("w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    processed_count = sum(
        1 for item in metadata if item["status"] == "ok"
    )

    skipped_count = sum(
        1 for item in metadata if item["status"] == "skipped_exists"
    )

    print("\n=== Finished ===")
    print("processed:", processed_count)
    print("skipped existing:", skipped_count)
    print("errors:", len(errors))
    print("metadata:", out_root / "metadata.json")
    print("errors file:", out_root / "errors.json")


if __name__ == "__main__":
    main()