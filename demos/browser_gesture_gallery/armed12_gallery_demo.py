"""
Webcam gallery demo for the Armed GRU gesture checkpoint.

Left two-finger slide  -> previous image
Right two-finger slide -> next image
Up two-finger slide    -> image above in the gallery grid
Down two-finger slide  -> image below in the gallery grid

Run:
  python3 armed12_gallery_demo.py

Optional real image folder:
  python3 tools/armed12_gallery_demo.py --images /path/to/gallery

Self-test without webcam/MediaPipe:
  python3 tools/armed12_gallery_demo.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import torch
from torch import nn

try:
    import cv2
except ImportError:
    cv2 = None


DEMO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = DEMO_ROOT / "models" / "gru_armed12_24f_best.pt"
DEFAULT_STATIC_MODEL = DEMO_ROOT / "static" / "best_mlp.pt"
DEFAULT_STATIC_LABELS = DEMO_ROOT / "static" / "label_to_id.json"
DEFAULT_STATIC_MEAN = DEMO_ROOT / "static" / "mean.npy"
DEFAULT_STATIC_STD = DEMO_ROOT / "static" / "std.npy"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LEFT_CLASS = "Sliding Two Fingers Left"
RIGHT_CLASS = "Sliding Two Fingers Right"
UP_CLASS = "Sliding Two Fingers Up"
DOWN_CLASS = "Sliding Two Fingers Down"
FIST_CLASS = "fist"
LIKE_CLASS = "like"
DISLIKE_CLASS = "dislike"
PALM_CLASS = "palm"


class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.classifier(hidden[-1])


class MLPGestureClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Prediction:
    label: str = "-"
    confidence: float = 0.0
    accepted: bool = False


@dataclass
class StaticPrediction:
    label: str = "-"
    confidence: float = 0.0
    armed: bool = False


def load_model(model_path: Path, device: torch.device):
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    required = [
        "model_state_dict",
        "input_size",
        "sequence_length",
        "hidden_size",
        "num_layers",
        "dropout",
        "class_names",
    ]
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint missing keys: {missing}")

    class_names = list(checkpoint["class_names"])
    model = GRUClassifier(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        num_classes=len(class_names),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint, class_names


def load_static_label_mapping(label_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    if not label_path.exists():
        raise FileNotFoundError(f"Static label mapping not found: {label_path}")
    with open(label_path, "r", encoding="utf-8") as f:
        label_to_id = json.load(f)
    id_to_label = {int(v): str(k) for k, v in label_to_id.items()}
    return label_to_id, id_to_label


def load_static_model(
    model_path: Path,
    label_path: Path,
    mean_path: Path,
    std_path: Path,
    device: torch.device,
):
    if not model_path.exists():
        raise FileNotFoundError(f"Static model checkpoint not found: {model_path}")
    if not mean_path.exists():
        raise FileNotFoundError(f"Static mean.npy not found: {mean_path}")
    if not std_path.exists():
        raise FileNotFoundError(f"Static std.npy not found: {std_path}")

    label_to_id, id_to_label = load_static_label_mapping(label_path)
    if FIST_CLASS not in label_to_id:
        raise ValueError(f"Static label mapping must include {FIST_CLASS!r}")

    model = MLPGestureClassifier(input_dim=42, num_classes=len(label_to_id)).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError("Static checkpoint missing key: model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = np.load(mean_path).astype(np.float32)
    std = np.load(std_path).astype(np.float32)
    if mean.shape != (1, 42):
        raise ValueError(f"Expected static mean shape (1, 42), got {mean.shape}")
    if std.shape != (1, 42):
        raise ValueError(f"Expected static std shape (1, 42), got {std.shape}")

    return model, id_to_label, mean, std


def landmarks_to_vector(hand_landmarks) -> np.ndarray:
    return np.array(
        [
            value
            for landmark in hand_landmarks.landmark
            for value in (landmark.x, landmark.y, landmark.z)
        ],
        dtype=np.float32,
    )


def landmarks_to_vector_xy(hand_landmarks) -> np.ndarray:
    return np.array(
        [
            value
            for landmark in hand_landmarks.landmark
            for value in (landmark.x, landmark.y)
        ],
        dtype=np.float32,
    )


def detected_mask(sequence: np.ndarray) -> np.ndarray:
    return np.abs(sequence).sum(axis=1) > 0


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


def interpolate_missing_frames(sequence: np.ndarray, mask: np.ndarray) -> np.ndarray:
    detected_indices = np.where(mask)[0]
    if len(detected_indices) < 2:
        return sequence.astype(np.float32)

    positions = np.arange(sequence.shape[0])
    filled = sequence.copy().astype(np.float32)
    for feature_index in range(sequence.shape[1]):
        filled[:, feature_index] = np.interp(
            positions,
            detected_indices,
            sequence[detected_indices, feature_index],
        )
    return filled


def temporal_resample(sequence: np.ndarray, target_len: int) -> np.ndarray:
    original_len, feature_dim = sequence.shape
    if original_len == target_len:
        return sequence.astype(np.float32)
    if original_len == 1:
        return np.repeat(sequence, target_len, axis=0).astype(np.float32)

    original_t = np.linspace(0.0, 1.0, original_len)
    target_t = np.linspace(0.0, 1.0, target_len)
    resized = np.zeros((target_len, feature_dim), dtype=np.float32)
    for feature_index in range(feature_dim):
        resized[:, feature_index] = np.interp(
            target_t,
            original_t,
            sequence[:, feature_index],
        )
    return resized


def prepare_recorded_sequence(
    sequence: np.ndarray,
    target_len: int,
    min_detected_frames: int,
    min_longest_run: int,
    min_active_density: float,
):
    mask = detected_mask(sequence)
    indices = np.where(mask)[0]
    detected_count = int(mask.sum())

    if detected_count == 0:
        return None, "No hand landmarks recorded"

    start = int(indices[0])
    end = int(indices[-1])
    active_sequence = sequence[start : end + 1]
    active_mask = mask[start : end + 1]

    active_span = len(active_sequence)
    density = detected_count / active_span
    run = longest_run(active_mask)

    if detected_count < min_detected_frames:
        return None, "Too few detected frames"
    if run < min_longest_run:
        return None, "Hand detection was too unstable"
    if density < min_active_density:
        return None, "Active segment density too low"

    filled = interpolate_missing_frames(active_sequence, active_mask)
    return temporal_resample(filled, target_len), "Accepted"


@torch.inference_mode()
def predict(model: nn.Module, sequence: np.ndarray, device: torch.device, class_names: list[str]):
    x = torch.from_numpy(sequence).unsqueeze(0).to(device)
    probabilities = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    class_id = int(np.argmax(probabilities))
    return class_names[class_id], float(probabilities[class_id])


@torch.inference_mode()
def predict_static(
    model: nn.Module,
    vector: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    id_to_label: dict[int, str],
):
    x = vector.reshape(1, 42).astype(np.float32)
    x = (x - mean) / std
    logits = model(torch.from_numpy(x).float().to(device))
    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    class_id = int(np.argmax(probabilities))
    return id_to_label[class_id], float(probabilities[class_id])


def load_gallery_images(image_dir: Path | None, width: int, height: int) -> list[np.ndarray]:
    if image_dir is not None:
        paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        images = []
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                images.append(fit_to_canvas(image, width, height))
        if images:
            return images

    return [make_placeholder_image(i + 1, width, height) for i in range(4)]


def fit_to_canvas(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), (18, 18, 18), dtype=np.uint8)
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, int(source_w * scale))
    resized_h = max(1, int(source_h * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    x = (width - resized_w) // 2
    y = (height - resized_h) // 2
    canvas[y : y + resized_h, x : x + resized_w] = resized
    return canvas


def paste_perspective(canvas: np.ndarray, image: np.ndarray, quad: np.ndarray) -> None:
    height, width = canvas.shape[:2]
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(source, quad.astype(np.float32))
    warped = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255, cv2.LINE_AA)
    shadow = cv2.GaussianBlur(mask, (31, 31), 0)
    canvas[:] = np.where(shadow[..., None] > 0, (canvas * 0.82).astype(np.uint8), canvas)
    canvas[mask > 0] = warped[mask > 0]


def transition_quad(
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    scale: float,
    direction: str,
    depth: float,
) -> np.ndarray:
    card_w = width * scale
    card_h = height * scale
    x0 = center_x - card_w / 2
    x1 = center_x + card_w / 2
    y0 = center_y - card_h / 2
    y1 = center_y + card_h / 2
    pinch = depth * min(width, height) * 0.10

    if direction in {"left", "right"}:
        sign = 1 if direction == "right" else -1
        return np.float32(
            [
                [x0 + sign * pinch, y0 + pinch],
                [x1 + sign * pinch, y0 - pinch],
                [x1 - sign * pinch, y1 + pinch],
                [x0 - sign * pinch, y1 - pinch],
            ]
        )

    sign = 1 if direction == "down" else -1
    return np.float32(
        [
            [x0 + pinch, y0 + sign * pinch],
            [x1 - pinch, y0 - sign * pinch],
            [x1 + pinch, y1 - sign * pinch],
            [x0 - pinch, y1 + sign * pinch],
        ]
    )


def render_3d_transition(
    old_image: np.ndarray,
    new_image: np.ndarray,
    progress: float,
    direction: str,
) -> np.ndarray:
    progress = max(0.0, min(1.0, progress))
    eased = 1 - (1 - progress) ** 3
    height, width = old_image.shape[:2]
    canvas = np.full_like(old_image, (12, 12, 12))
    distance = width if direction in {"left", "right"} else height
    sign = 1 if direction in {"right", "down"} else -1

    old_center_x = width / 2 + (sign * distance * 0.62 * eased if direction in {"left", "right"} else 0)
    old_center_y = height / 2 + (sign * distance * 0.62 * eased if direction in {"up", "down"} else 0)
    new_center_x = width / 2 - (sign * distance * 0.62 * (1 - eased) if direction in {"left", "right"} else 0)
    new_center_y = height / 2 - (sign * distance * 0.62 * (1 - eased) if direction in {"up", "down"} else 0)

    old_quad = transition_quad(width, height, old_center_x, old_center_y, 1.0 - 0.20 * eased, direction, eased)
    new_quad = transition_quad(width, height, new_center_x, new_center_y, 0.80 + 0.20 * eased, direction, 1 - eased)

    paste_perspective(canvas, old_image, old_quad)
    paste_perspective(canvas, new_image, new_quad)
    return canvas


def draw_reaction_burst(
    frame: np.ndarray,
    started_at: float,
    now: float,
    emoji: str,
    direction: str = "up",
    duration: float = 1.0,
) -> None:
    age = now - started_at
    if age < 0 or age > duration:
        return
    height, width = frame.shape[:2]
    progress = age / duration
    if direction == "down":
        bubbles = [
            (0.18, 0.36, 90, 0.00),
            (0.32, 0.30, 132, 0.12),
            (0.50, 0.34, 104, 0.24),
            (0.66, 0.29, 142, 0.36),
            (0.82, 0.36, 96, 0.48),
        ]
        headline_y = int(height * 0.28 + 120 * progress)
    else:
        bubbles = [
            (0.18, 0.78, -118, 0.00),
            (0.32, 0.72, -164, 0.12),
            (0.50, 0.76, -136, 0.24),
            (0.66, 0.70, -176, 0.36),
            (0.82, 0.78, -124, 0.48),
        ]
        headline_y = int(height * 0.34 - 120 * progress)
    alpha = max(0.0, 1.0 - progress)
    for x_frac, y_frac, lift, delay in bubbles:
        local = max(0.0, min(1.0, (progress - delay * 0.35) / 0.8))
        x = int(width * x_frac + np.sin((local + delay) * np.pi * 2) * 18)
        y = int(height * y_frac + lift * local)
        radius = int(26 + 10 * local)
        color = (255, 165, 70)
        cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), radius, (255, 255, 255), 2, cv2.LINE_AA)
        draw_unicode_text(frame, emoji, (x - 18, y + 18), size=36)
    if alpha > 0:
        draw_unicode_text(frame, emoji, (width // 2 - 44, headline_y), size=72)


def draw_like_burst(frame: np.ndarray, started_at: float, now: float, duration: float = 1.0) -> None:
    draw_reaction_burst(frame, started_at, now, "👍", "up", duration)


def draw_dislike_burst(frame: np.ndarray, started_at: float, now: float, duration: float = 1.0) -> None:
    draw_reaction_burst(frame, started_at, now, "👎", "down", duration)


def make_placeholder_image(index: int, width: int, height: int) -> np.ndarray:
    palettes = [
        ((24, 72, 122), (230, 190, 80)),
        ((42, 110, 82), (220, 112, 80)),
        ((96, 60, 130), (88, 185, 210)),
        ((130, 72, 56), (105, 168, 96)),
    ]
    base, accent = palettes[(index - 1) % len(palettes)]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    mix = (0.65 * x + 0.35 * y)[..., None]
    image = (np.array(base, dtype=np.float32) * (1 - mix) + np.array(accent, dtype=np.float32) * mix)
    image = image.astype(np.uint8)

    center = (width // 2, height // 2)
    cv2.circle(image, center, min(width, height) // 5, accent, -1, cv2.LINE_AA)
    cv2.circle(image, center, min(width, height) // 5, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(
        image,
        f"PHOTO {index}",
        (width // 2 - 150, height // 2 + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )
    return image


def draw_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    scale: float = 0.7,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> None:
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_unicode_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    size: int = 48,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        draw_text(frame, text, position, max(0.6, size / 48), color, 2)
        return

    font_candidates = [
        "/System/Library/Fonts/Apple Color Emoji.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "C:/Windows/Fonts/seguiemj.ttf",
        "C:/Windows/Fonts/seguisym.ttf",
    ]
    font = None
    for font_path in font_candidates:
        try:
            font = ImageFont.truetype(font_path, size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    frame[:] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def draw_overlay(
    frame: np.ndarray,
    image_index: int,
    image_count: int,
    state: str,
    prediction: Prediction,
    static_prediction: StaticPrediction,
    status: str,
    auto_mode: bool,
    progress: float,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 104), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    title = "Swipe, then make a fist"
    if state == "IDLE":
        title = "Make a fist to start"
    elif state == "ARMED":
        title = "Ready - swipe now"
    elif state == "RECORDING":
        title = "Swipe now"
    elif static_prediction.armed:
        title = "Fist detected"

    title_color = (80, 235, 120) if state in {"ARMED", "RECORDING"} else (255, 255, 255)
    draw_text(frame, title, (24, 42), 0.92, title_color, 2)
    draw_text(frame, status, (24, 78), 0.62, (220, 235, 255), 2)
    draw_text(frame, f"{image_index + 1}/{image_count}", (frame.shape[1] - 92, 42), 0.86)

    if state in {"ARMED", "RECORDING"}:
        badge_x = frame.shape[1] - 290
        badge_y = 20
        cv2.rectangle(frame, (badge_x, badge_y), (badge_x + 170, badge_y + 40), (28, 135, 64), -1)
        cv2.rectangle(frame, (badge_x, badge_y), (badge_x + 170, badge_y + 40), (130, 255, 170), 2)
        draw_text(frame, "CONTROL ON", (badge_x + 16, badge_y + 28), 0.58, (255, 255, 255), 2)

    if state == "RECORDING":
        bar_x, bar_y, bar_w, bar_h = frame.shape[1] - 344, 70, 260, 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), (0, 210, 255), -1)

    bottom_y = frame.shape[0] - 54
    debug_color = (200, 200, 200)
    draw_text(
        frame,
        f"static {static_prediction.label} {static_prediction.confidence:.2f} | swipe {prediction.label} {prediction.confidence:.2f}",
        (24, bottom_y),
        0.42,
        debug_color,
        1,
    )
    draw_text(
        frame,
        f"state {state} | SPACE record | R reset | Q quit",
        (24, bottom_y + 24),
        0.42,
        debug_color,
        1,
    )


def paste_webcam_preview(frame: np.ndarray, webcam_frame: np.ndarray) -> None:
    preview_w = min(320, frame.shape[1] // 4)
    scale = preview_w / webcam_frame.shape[1]
    preview_h = int(webcam_frame.shape[0] * scale)
    preview = cv2.resize(webcam_frame, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
    x = frame.shape[1] - preview_w - 24
    y = frame.shape[0] - preview_h - 56
    frame[y : y + preview_h, x : x + preview_w] = preview
    cv2.rectangle(frame, (x, y), (x + preview_w, y + preview_h), (255, 255, 255), 2)


def run(args: argparse.Namespace) -> None:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit(
            "mediapipe is not installed in this Python environment. "
            "Install it with: python3 -m pip install mediapipe"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, class_names = load_model(args.model, device)
    static_model, static_id_to_label, static_mean, static_std = load_static_model(
        args.static_model,
        args.static_labels,
        args.static_mean,
        args.static_std,
        device,
    )
    target_len = int(checkpoint["sequence_length"])

    if int(checkpoint["input_size"]) != 63:
        raise ValueError(f"This demo expects input_size=63, got {checkpoint['input_size']}")
    required_gestures = {LEFT_CLASS, RIGHT_CLASS, UP_CLASS, DOWN_CLASS}
    missing_gestures = sorted(required_gestures - set(class_names))
    if missing_gestures:
        raise ValueError(f"Checkpoint classes missing gallery gestures: {missing_gestures}")

    images = load_gallery_images(args.images, args.width, args.height)
    image_index = 0

    if args.backend == "avfoundation":
        cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    elif args.backend == "any":
        cap = cv2.VideoCapture(args.camera)
    else:
        cap = cv2.VideoCapture(args.camera, int(args.backend))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    state = "IDLE"
    auto_mode = bool(args.auto)
    fist_count = 0
    ready_since: float | None = None
    last_sample_time = 0.0
    last_action_time = 0.0
    last_active_time = 0.0
    recording_started = 0.0
    record_buffer: list[np.ndarray] = []
    prediction = Prediction()
    static_prediction = StaticPrediction()
    status = "Make a fist to turn on gesture control"
    sample_interval = 1.0 / args.sample_fps
    required_samples = max(1, int(round(args.sample_fps * args.record_seconds)))
    progress = 0.0
    recent_labels = deque(maxlen=4)
    consecutive_read_failures = 0
    transition_from: int | None = None
    transition_to: int | None = None
    transition_direction = "right"
    transition_started = 0.0
    like_started = -999.0
    last_like_time = -999.0

    print("=== Armed Gallery Gesture Demo ===")
    print("model:", args.model)
    print("static model:", args.static_model)
    print("device:", device)
    print("classes:", class_names)
    print("images:", len(images))
    print("fist gesture: turn gesture control on")
    print("left gesture: next image")
    print("right gesture: previous image")
    print(f"up gesture: move up by {args.columns} image(s)")
    print(f"down gesture: move down by {args.columns} image(s)")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hands:
        while True:
            ok, webcam_frame = cap.read()
            if not ok:
                consecutive_read_failures += 1
                if consecutive_read_failures >= args.max_read_failures:
                    print("Failed to read camera frame.")
                    break
                time.sleep(0.05)
                continue
            consecutive_read_failures = 0

            if not args.no_mirror:
                webcam_frame = cv2.flip(webcam_frame, 1)

            rgb = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            now = time.monotonic()

            hand_detected = bool(results.multi_hand_landmarks)
            current_vector = np.zeros(63, dtype=np.float32)
            is_fist = False
            if hand_detected:
                hand_landmarks = results.multi_hand_landmarks[0]
                current_vector = landmarks_to_vector(hand_landmarks)
                current_vector_xy = landmarks_to_vector_xy(hand_landmarks)
                static_label, static_confidence = predict_static(
                    static_model,
                    current_vector_xy,
                    static_mean,
                    static_std,
                    device,
                    static_id_to_label,
                )
                is_fist = static_label == FIST_CLASS and static_confidence >= args.fist_confidence_threshold
                if is_fist:
                    fist_count = min(args.fist_frames, fist_count + 1)
                else:
                    fist_count = 0
                static_prediction = StaticPrediction(
                    static_label,
                    static_confidence,
                    fist_count >= args.fist_frames,
                )
                if (
                    static_label == LIKE_CLASS
                    and static_confidence >= args.like_confidence_threshold
                    and now - last_like_time >= args.like_cooldown_seconds
                ):
                    like_started = now
                    last_like_time = now
                mp_drawing.draw_landmarks(webcam_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                fist_count = 0
                static_prediction = StaticPrediction()

            if state == "IDLE":
                progress = 0.0
                if static_prediction.armed:
                    state = "ARMED"
                    last_active_time = now
                    record_buffer = []
                    status = "Release fist to start recording"
                elif hand_detected:
                    status = "Make a fist to turn on gesture control"
                else:
                    status = "Show one hand and make a fist"

            elif state == "ARMED":
                progress = 0.0
                if now - last_active_time >= args.active_timeout:
                    state = "IDLE"
                    status = "Make a fist again to start"
                elif static_prediction.armed:
                    status = "Release fist to start recording"
                elif hand_detected:
                    state = "RECORDING"
                    record_buffer = []
                    last_sample_time = 0.0
                    recording_started = now
                    status = "Swipe now"
                else:
                    status = "Swipe when your hand appears"

            elif state == "RECORDING":
                if not hand_detected:
                    status = "Keep swiping"
                    if now - recording_started >= args.recording_timeout:
                        state = "IDLE"
                        record_buffer = []
                        progress = 0.0
                        status = "Make a fist again to start"
                elif now - last_sample_time >= sample_interval and not is_fist:
                    record_buffer.append(current_vector.copy())
                    last_sample_time = now

                progress = min(1.0, len(record_buffer) / required_samples)
                if len(record_buffer) >= required_samples:
                    if len(record_buffer) < args.min_detected_frames:
                        prediction = Prediction("-", 0.0, False)
                        status = "Swipe longer"
                        state = "IDLE"
                        record_buffer = []
                        progress = 0.0
                        continue

                    raw_sequence = np.stack(record_buffer).astype(np.float32)
                    model_input, reason = prepare_recorded_sequence(
                        raw_sequence,
                        target_len=target_len,
                        min_detected_frames=args.min_detected_frames,
                        min_longest_run=args.min_longest_run,
                        min_active_density=args.min_active_density,
                    )

                    if model_input is None:
                        prediction = Prediction("-", 0.0, False)
                        status = reason
                    else:
                        label, confidence = predict(model, model_input, device, class_names)
                        is_target = label in {LEFT_CLASS, RIGHT_CLASS, UP_CLASS, DOWN_CLASS}
                        accepted = is_target and confidence >= args.confidence_threshold
                        prediction = Prediction(label, confidence, accepted)
                        recent_labels.append(label)

                        if accepted and label == LEFT_CLASS:
                            old_index = image_index
                            image_index = (image_index + 1) % len(images)
                            transition_from = old_index
                            transition_to = image_index
                            transition_direction = "right"
                            transition_started = now
                            last_action_time = now
                            status = "Moved right"
                        elif accepted and label == RIGHT_CLASS:
                            old_index = image_index
                            image_index = (image_index - 1) % len(images)
                            transition_from = old_index
                            transition_to = image_index
                            transition_direction = "left"
                            transition_started = now
                            last_action_time = now
                            status = "Moved left"
                        elif accepted and label == UP_CLASS:
                            old_index = image_index
                            image_index = (image_index - args.columns) % len(images)
                            transition_from = old_index
                            transition_to = image_index
                            transition_direction = "up"
                            transition_started = now
                            last_action_time = now
                            status = "Moved up"
                        elif accepted and label == DOWN_CLASS:
                            old_index = image_index
                            image_index = (image_index + args.columns) % len(images)
                            transition_from = old_index
                            transition_to = image_index
                            transition_direction = "down"
                            transition_started = now
                            last_action_time = now
                            status = "Moved down"
                        elif is_target:
                            status = "Low confidence - no move"
                        else:
                            status = "Not a gallery slide"

                    state = "IDLE"
                    ready_since = None
                    record_buffer = []
                    progress = 0.0
                    status = f"{status} - make a fist for next swipe"

            if transition_from is not None and transition_to is not None:
                transition_progress = (now - transition_started) / args.transition_seconds
                if transition_progress >= 1.0:
                    transition_from = None
                    transition_to = None
                    display = images[image_index].copy()
                else:
                    display = render_3d_transition(
                        images[transition_from],
                        images[transition_to],
                        transition_progress,
                        transition_direction,
                    )
            else:
                display = images[image_index].copy()
            draw_like_burst(display, like_started, now, args.like_burst_seconds)
            paste_webcam_preview(display, webcam_frame)
            draw_overlay(
                display,
                image_index=image_index,
                image_count=len(images),
                state=state,
                prediction=prediction,
                static_prediction=static_prediction,
                status=status,
                auto_mode=auto_mode,
                progress=progress,
            )
            cv2.imshow("Armed Gesture Gallery", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("a"):
                auto_mode = not auto_mode
                ready_since = now
                status = f"Auto mode {'ON' if auto_mode else 'OFF'}"
            elif key == ord("r"):
                state = "IDLE"
                fist_count = 0
                ready_since = None
                record_buffer = []
                prediction = Prediction()
                static_prediction = StaticPrediction()
                transition_from = None
                transition_to = None
                status = "Reset - make a fist to start"
            elif key == ord(" ") and state in {"IDLE", "ARMED"}:
                state = "RECORDING"
                record_buffer = []
                last_sample_time = 0.0
                recording_started = now
                status = "Swipe now"

    cap.release()
    cv2.destroyAllWindows()


def self_test(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, class_names = load_model(args.model, device)
    static_model, static_id_to_label, static_mean, static_std = load_static_model(
        args.static_model,
        args.static_labels,
        args.static_mean,
        args.static_std,
        device,
    )
    images = load_gallery_images(args.images, args.width, args.height)
    dummy = np.zeros((int(checkpoint["sequence_length"]), int(checkpoint["input_size"])), dtype=np.float32)
    label, confidence = predict(model, dummy, device, class_names)
    static_label, static_confidence = predict_static(
        static_model,
        np.zeros(42, dtype=np.float32),
        static_mean,
        static_std,
        device,
        static_id_to_label,
    )
    print("self-test: ok")
    print("model:", args.model)
    print("static_model:", args.static_model)
    print("sequence_length:", checkpoint["sequence_length"])
    print("input_size:", checkpoint["input_size"])
    print("images:", len(images))
    print("dummy_prediction:", label, f"{confidence:.3f}")
    print("dummy_static_prediction:", static_label, f"{static_confidence:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gesture-controlled gallery demo for Armed GRU model.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--static-model", type=Path, default=DEFAULT_STATIC_MODEL)
    parser.add_argument("--static-labels", type=Path, default=DEFAULT_STATIC_LABELS)
    parser.add_argument("--static-mean", type=Path, default=DEFAULT_STATIC_MEAN)
    parser.add_argument("--static-std", type=Path, default=DEFAULT_STATIC_STD)
    parser.add_argument("--images", type=Path, default=None, help="Folder of jpg/png/webp images.")
    parser.add_argument("--camera", type=int, default=0)
    default_backend = "avfoundation" if platform.system() == "Darwin" else "any"
    parser.add_argument(
        "--backend",
        default=default_backend,
        help="OpenCV camera backend: avfoundation, any, or numeric backend id.",
    )
    parser.add_argument("--max-read-failures", type=int, default=80)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-width", type=int, default=960)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--sample-fps", type=float, default=12.0)
    parser.add_argument("--record-seconds", type=float, default=0.65)
    parser.add_argument("--recording-timeout", type=float, default=1.4)
    parser.add_argument("--ready-frames", type=int, default=8)
    parser.add_argument("--auto-delay", type=float, default=0.35)
    parser.add_argument("--cooldown-seconds", type=float, default=0.75)
    parser.add_argument("--active-timeout", type=float, default=6.0)
    parser.add_argument("--transition-seconds", type=float, default=0.55)
    parser.add_argument("--columns", type=int, default=2, help="Gallery grid width used by up gesture.")
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--fist-confidence-threshold", type=float, default=0.65)
    parser.add_argument("--fist-frames", type=int, default=3)
    parser.add_argument("--like-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--like-burst-seconds", type=float, default=1.0)
    parser.add_argument("--like-cooldown-seconds", type=float, default=1.2)
    parser.add_argument("--min-detected-frames", type=int, default=3)
    parser.add_argument("--min-longest-run", type=int, default=2)
    parser.add_argument("--min-active-density", type=float, default=0.30)
    parser.add_argument("--min-detection-confidence", type=float, default=0.3)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.3)
    parser.add_argument("--auto", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
