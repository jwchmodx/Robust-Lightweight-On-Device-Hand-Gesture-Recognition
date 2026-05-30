from pathlib import Path
import csv
import argparse
import imageio.v2 as imageio
from PIL import Image, ImageDraw


def read_csv(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        sample = f.readline()
        delimiter = ";" if ";" in sample else ","
        f.seek(0)

        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if len(row) >= 2:
                rows.append((row[0].strip(), row[1].strip()))
    return rows


def read_labels(label_path):
    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            labels.append(line.strip())
    return labels


def make_gif(video_dir, out_path, label, fps=8):
    frame_paths = sorted(
        list(video_dir.glob("*.jpg")) +
        list(video_dir.glob("*.jpeg")) +
        list(video_dir.glob("*.png"))
    )

    if not frame_paths:
        return False

    frames = []

    for p in frame_paths:
        img = Image.open(p).convert("RGB")

        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, img.width, 30], fill=(0, 0, 0))
        draw.text((10, 8), label, fill=(255, 255, 255))

        frames.append(img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", default="data/20bn-jester-v1")
    parser.add_argument("--annotation", default="data/annotations/jester-v1-train.csv")
    parser.add_argument("--labels", default="data/annotations/jester-v1-labels.csv")
    parser.add_argument("--out-root", default="data/gif_all")

    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    rows = read_csv(Path(args.annotation))
    labels = read_labels(Path(args.labels))

    # 라벨별 하나씩 선택
    picked = {}

    for video_id, label in rows:
        if label not in picked:
            picked[label] = video_id

        if len(picked) == len(labels):
            break

    print("total labels:", len(labels))
    print("picked:", len(picked))

    # GIF 생성
    for label, video_id in picked.items():
        video_dir = data_root / video_id
        safe_label = label.replace(" ", "_").replace("/", "_")

        out_path = out_root / f"{safe_label}_{video_id}.gif"

        ok = make_gif(video_dir, out_path, label)

        if ok:
            print("saved:", out_path)
        else:
            print("failed:", label, video_id)


if __name__ == "__main__":
    main()