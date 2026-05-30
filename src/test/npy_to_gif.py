from pathlib import Path
import argparse
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw


# MediaPipe hand connections
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def draw_frame(points, size=512):
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    xy = points.reshape(21, 3)[:, :2]

    # 전부 0이면 빈 프레임
    if np.allclose(xy, 0):
        return img

    # MediaPipe 좌표는 보통 0~1 정규화 좌표
    xy = np.clip(xy, 0, 1)
    xy[:, 0] *= size
    xy[:, 1] *= size

    for a, b in CONNECTIONS:
        draw.line([tuple(xy[a]), tuple(xy[b])], width=4, fill="black")

    for x, y in xy:
        r = 5
        draw.ellipse([x-r, y-r, x+r, y+r], fill="red")

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="예: data/processed/train/1.npy")
    parser.add_argument("--output", required=True, help="예: data/gif_from_npy/1.gif")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    x = np.load(args.input)

    if x.ndim != 2 or x.shape[1] != 63:
        raise ValueError(f"Expected shape (T, 63), got {x.shape}")

    frames = [draw_frame(frame, size=args.size) for frame in x]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    imageio.mimsave(out_path, frames, fps=args.fps, loop=0)

    print("saved:", out_path)
    print("shape:", x.shape)


if __name__ == "__main__":
    main()