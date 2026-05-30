from pathlib import Path
import csv
import argparse
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont


TARGET_LABELS = [
    "No gesture",
    "Doing other things",
    "Swiping Up",
    "Swiping Down",
    "Swiping Left",
    "Swiping Right",
    "Stop Sign",
]


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


def make_gif(video_dir, out_path, label, fps=8, max_width=360):
    frame_paths = sorted(
        list(video_dir.glob("*.jpg")) +
        list(video_dir.glob("*.jpeg")) +
        list(video_dir.glob("*.png"))
    )

    if not frame_paths:
        print("No frames:", video_dir)
        return False

    frames = []

    for p in frame_paths:
        img = Image.open(p).convert("RGB")

        if img.width > max_width:
            ratio = max_width / img.width
            img = img.resize((max_width, int(img.height * ratio)))

        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, img.width, 28], fill=(0, 0, 0))
        draw.text((8, 6), label, fill=(255, 255, 255))

        frames.append(img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", default="data/20bn-jester-v1")
    parser.add_argument("--annotation", default="data/annotations/jester-v1-train.csv")
    parser.add_argument("--out-root", default="data/gif_check")
    parser.add_argument("--fps", type=int, default=8)

    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    rows = read_csv(Path(args.annotation))

    picked = {}

    for video_id, label in rows:
        if label in TARGET_LABELS and label not in picked:
            picked[label] = video_id

        if len(picked) == len(TARGET_LABELS):
            break

    print("picked samples:")
    for label, video_id in picked.items():
        print(label, "->", video_id)

    for label, video_id in picked.items():
        video_dir = data_root / video_id
        safe_label = label.replace(" ", "_").replace("/", "_")
        out_path = out_root / f"{safe_label}_{video_id}.gif"

        ok = make_gif(video_dir, out_path, label, fps=args.fps)

        if ok:
            print("saved:", out_path)


if __name__ == "__main__":
    main()