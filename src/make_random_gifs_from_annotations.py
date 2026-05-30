""" 
python src/make_random_gifs_from_annotations.py \
  --data-root data/20bn-jester-v1 \
  --annotation data/annotations/jester-v1-train.csv \
  --out-root data/random_gifs/train \
  --num-per-label 3 \
  --seed 42
"""


from pathlib import Path
import csv
import argparse
import random
import imageio.v2 as imageio
from PIL import Image, ImageDraw


def read_annotations(csv_path: Path):
    rows = []

    with open(csv_path, "r", encoding="utf-8") as f:
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


def safe_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def get_frame_paths(video_dir: Path):
    frame_paths = (
        list(video_dir.glob("*.jpg")) +
        list(video_dir.glob("*.jpeg")) +
        list(video_dir.glob("*.png"))
    )

    def sort_key(path: Path):
        try:
            return int(path.stem)
        except ValueError:
            return path.stem

    return sorted(frame_paths, key=sort_key)


def make_gif(video_dir: Path, out_path: Path, label: str, video_id: str, fps: int = 8, max_width: int = 360):
    frame_paths = get_frame_paths(video_dir)

    if len(frame_paths) == 0:
        print(f"[WARN] no frames: {video_dir}")
        return False

    frames = []

    for idx, frame_path in enumerate(frame_paths, start=1):
        img = Image.open(frame_path).convert("RGB")

        if img.width > max_width:
            ratio = max_width / img.width
            new_w = max_width
            new_h = int(img.height * ratio)
            img = img.resize((new_w, new_h))

        draw = ImageDraw.Draw(img)

        # 상단 검은 배경
        banner_h = 36
        draw.rectangle([0, 0, img.width, banner_h], fill=(0, 0, 0))

        text = f"{label} | id={video_id} | frame {idx}/{len(frame_paths)}"
        draw.text((8, 10), text, fill=(255, 255, 255))

        frames.append(img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=str,
        default="data/20bn-jester-v1"
    )

    parser.add_argument(
        "--annotation",
        type=str,
        default="data/annotations/jester-v1-train.csv"
    )

    parser.add_argument(
        "--out-root",
        type=str,
        default="data/random_gifs"
    )

    parser.add_argument(
        "--num-per-label",
        type=int,
        default=3,
        help="각 label마다 랜덤으로 몇 개 뽑을지"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=8
    )

    parser.add_argument(
        "--max-width",
        type=int,
        default=360
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help='특정 라벨만 보고 싶으면 쉼표로 구분. 예: "Swiping Left,Swiping Right,Stop Sign"'
    )

    args = parser.parse_args()

    random.seed(args.seed)

    data_root = Path(args.data_root)
    annotation_path = Path(args.annotation)
    out_root = Path(args.out_root)

    rows = read_annotations(annotation_path)

    # label별로 video_id 묶기
    label_to_ids = {}
    for video_id, label in rows:
        label_to_ids.setdefault(label, []).append(video_id)

    # 특정 label만 선택할 수도 있게
    if args.labels is not None:
        selected_labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    else:
        selected_labels = sorted(label_to_ids.keys())

    print("annotation:", annotation_path.resolve())
    print("data_root:", data_root.resolve())
    print("out_root:", out_root.resolve())
    print("num_per_label:", args.num_per_label)
    print("seed:", args.seed)
    print("labels:", selected_labels)
    print()

    total_saved = 0

    for label in selected_labels:
        if label not in label_to_ids:
            print(f"[WARN] label not found in annotation: {label}")
            continue

        candidates = label_to_ids[label]

        sample_count = min(args.num_per_label, len(candidates))
        sampled_ids = random.sample(candidates, sample_count)

        print(f"[{label}] total={len(candidates)}, sampled={sample_count}")

        label_dir = out_root / safe_name(label)

        for video_id in sampled_ids:
            video_dir = data_root / video_id
            out_path = label_dir / f"{safe_name(label)}_{video_id}.gif"

            ok = make_gif(
                video_dir=video_dir,
                out_path=out_path,
                label=label,
                video_id=video_id,
                fps=args.fps,
                max_width=args.max_width,
            )

            if ok:
                total_saved += 1
                print("  saved:", out_path)
            else:
                print("  failed:", video_dir)

    print()
    print("done")
    print("total saved:", total_saved)


if __name__ == "__main__":
    main()