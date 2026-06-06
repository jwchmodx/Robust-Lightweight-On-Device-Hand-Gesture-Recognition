from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROBUST_ROOT = Path(__file__).resolve().parents[1]


CONTROL_LABELS = [
    "No gesture",
    "Doing other things",
    "Swiping Up",
    "Swiping Down",
    "Swiping Left",
    "Swiping Right",
    "Stop Sign",
]


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


class ArrayDataset(Dataset):
    def __init__(self, x_path: Path, y_path: Path) -> None:
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        if self.x.ndim != 3:
            raise ValueError(f"Expected X shape (N,T,D), got {self.x.shape}")
        if self.y.ndim != 1:
            raise ValueError(f"Expected y shape (N,), got {self.y.shape}")
        if len(self.x) != len(self.y):
            raise ValueError(f"X/y length mismatch: {len(self.x)} vs {len(self.y)}")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.array(self.x[index], dtype=np.float32, copy=True))
        y = torch.tensor(int(self.y[index]), dtype=torch.long)
        return x, y


class SequenceFolderDataset(Dataset):
    def __init__(
        self,
        sequence_dir: Path,
        annotation_path: Path,
        class_names: list[str],
        original_label_names: list[str],
        sequence_length: int,
    ) -> None:
        self.sequence_dir = sequence_dir
        self.class_names = class_names
        self.sequence_length = sequence_length
        target_label_map = {label: idx for idx, label in enumerate(class_names)}

        rows = read_annotation_rows(annotation_path)
        self.samples: list[tuple[Path, int]] = []
        skipped_missing = 0
        skipped_label = 0

        for video_id, label in rows:
            if label not in target_label_map:
                skipped_label += 1
                continue
            path = sequence_dir / f"{video_id}.npy"
            if not path.exists():
                skipped_missing += 1
                continue
            self.samples.append((path, target_label_map[label]))

        if not self.samples:
            raise RuntimeError(
                f"No usable samples in {sequence_dir} for labels {class_names}"
            )

        self.original_label_map = {
            label: idx for idx, label in enumerate(original_label_names)
        }
        self.skipped_missing = skipped_missing
        self.skipped_label = skipped_label

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label_id = self.samples[index]
        x = normalize_sequence(np.load(path), self.sequence_length)
        x = torch.from_numpy(np.array(x, dtype=np.float32, copy=True))
        y = torch.tensor(label_id, dtype=torch.long)
        return x, y


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROBUST_ROOT / path


def read_annotation_rows(csv_path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        sample = f.readline()
        delimiter = ";" if ";" in sample else ","
        f.seek(0)
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if len(row) >= 2:
                rows.append((row[0].strip(), row[1].strip()))
    return rows


def normalize_sequence(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.ndim != 2 or x.shape[1] != 63:
        raise ValueError(f"Expected sequence shape (T,63), got {x.shape}")
    if x.shape[0] == target_len:
        return x.astype(np.float32)
    if x.shape[0] > target_len:
        indices = np.linspace(0, x.shape[0] - 1, target_len).astype(int)
        return x[indices].astype(np.float32)
    pad = np.zeros((target_len - x.shape[0], x.shape[1]), dtype=x.dtype)
    return np.vstack([x, pad]).astype(np.float32)


def read_label_map(label_map_path: Path) -> list[str]:
    with label_map_path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    return [
        label
        for label, _ in sorted(label_map.items(), key=lambda item: item[1])
    ]


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def build_model(checkpoint: dict, device: torch.device) -> nn.Module:
    class_names = checkpoint["class_names"]
    model = GRUClassifier(
        input_size=int(checkpoint.get("input_size", 63)),
        hidden_size=int(checkpoint.get("hidden_size", 128)),
        num_classes=len(class_names),
        num_layers=int(checkpoint.get("num_layers", 1)),
        dropout=float(checkpoint.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    num_classes: int,
) -> tuple[float, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total_correct = 0
    total_count = 0

    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        total_correct += int((pred == y).sum().item())
        total_count += int(y.numel())
        for true_label, pred_label in zip(y.cpu(), pred.cpu()):
            confusion[int(true_label), int(pred_label)] += 1

    return total_correct / total_count, confusion.numpy()


def metrics_from_confusion(confusion: np.ndarray) -> dict:
    per_class = []
    f1_values = []

    for class_id in range(confusion.shape[0]):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)
        support = int(confusion[class_id, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class.append({
            "class_id": class_id,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
        f1_values.append(f1)

    accuracy = float(np.trace(confusion) / confusion.sum())
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "weighted_f1": float(
            sum(row["f1"] * row["support"] for row in per_class)
            / max(1, sum(row["support"] for row in per_class))
        ),
        "per_class": per_class,
    }


def default_data_root(checkpoint: dict) -> str | None:
    if "data_root" in checkpoint:
        return checkpoint["data_root"]
    if checkpoint.get("subset") == "control":
        return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved dynamic GRU checkpoints and save macro-F1 metrics."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-root")
    parser.add_argument("--sequence-dir")
    parser.add_argument(
        "--annotation",
        default="data/annotations/jester-v1-validation.csv",
    )
    parser.add_argument(
        "--all-labels-annotation",
        default="data/annotations/jester-v1-train.csv",
    )
    args = parser.parse_args()

    checkpoint_path = resolve_path(args.checkpoint)
    output_path = resolve_path(args.out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = list(checkpoint["class_names"])
    sequence_length = int(checkpoint.get("sequence_length", 32))
    model = build_model(checkpoint, device)

    if args.sequence_dir:
        train_rows = read_annotation_rows(resolve_path(args.all_labels_annotation))
        original_label_names = sorted({label for _, label in train_rows})
        dataset = SequenceFolderDataset(
            sequence_dir=resolve_path(args.sequence_dir),
            annotation_path=resolve_path(args.annotation),
            class_names=class_names,
            original_label_names=original_label_names,
            sequence_length=sequence_length,
        )
        data_source = {
            "type": "sequence_folder",
            "sequence_dir": str(resolve_path(args.sequence_dir)),
            "annotation": str(resolve_path(args.annotation)),
            "skipped_missing": dataset.skipped_missing,
            "skipped_non_target_label": dataset.skipped_label,
        }
    else:
        data_root_arg = args.data_root or default_data_root(checkpoint)
        if not data_root_arg:
            raise ValueError("Provide --data-root or --sequence-dir for this checkpoint.")
        data_root = resolve_path(data_root_arg)
        dataset = ArrayDataset(data_root / "X_val.npy", data_root / "y_val.npy")
        label_map_path = data_root / "label_map.json"
        if label_map_path.exists():
            data_class_names = read_label_map(label_map_path)
            if data_class_names != class_names:
                raise ValueError(
                    f"label_map classes differ from checkpoint classes: "
                    f"{data_class_names} vs {class_names}"
                )
        data_source = {
            "type": "array_dataset",
            "data_root": str(data_root),
        }

    accuracy, confusion = evaluate(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        device=device,
        num_classes=len(class_names),
    )
    metrics = metrics_from_confusion(confusion)
    metrics["accuracy"] = accuracy

    result = {
        "checkpoint": str(checkpoint_path),
        "class_names": class_names,
        "num_classes": len(class_names),
        "num_samples": len(dataset),
        "sequence_length": sequence_length,
        "device": str(device),
        "data_source": data_source,
        "checkpoint_best_val_accuracy": checkpoint.get("best_val_accuracy"),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "per_class": metrics["per_class"],
        "confusion_matrix": confusion.tolist(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"checkpoint: {checkpoint_path.name}")
    print(f"samples: {len(dataset):,}")
    print(f"accuracy: {result['accuracy']:.6f}")
    print(f"macro_f1: {result['macro_f1']:.6f}")
    print(f"weighted_f1: {result['weighted_f1']:.6f}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
