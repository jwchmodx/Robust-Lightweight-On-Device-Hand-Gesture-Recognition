"""
Armed-4 GRU 학습 실행 예시:

python src/armed4/train_gru_armed4.py \
  --data-root data/model_ready/armed4_24f \
  --epochs 30 \
  --batch-size 256 \
  --hidden-size 128

클래스 불균형 보정까지 적용할 경우:

python src/armed4/train_gru_armed4.py \
  --data-root data/model_ready/armed4_24f \
  --epochs 30 \
  --batch-size 256 \
  --hidden-size 128 \
  --use-class-weights
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROBUST_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROBUST_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_label_map(label_map_path: Path) -> list[str]:
    """label_map.json의 label id 순서대로 클래스 이름 목록을 반환한다."""
    if not label_map_path.exists():
        raise FileNotFoundError(f"Missing file: {label_map_path}")

    with label_map_path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)

    class_names = [
        label for label, label_id in sorted(label_map.items(), key=lambda item: item[1])
    ]

    expected_ids = list(range(len(class_names)))
    actual_ids = sorted(label_map.values())

    if actual_ids != expected_ids:
        raise ValueError(
            f"label_map ids must be continuous from 0. Got: {label_map}"
        )

    return class_names


class GestureDataset(Dataset):
    """
    Armed-4용 통합 npy 파일을 직접 읽는다.

    X shape: (N, sequence_length, 63)
    y shape: (N,)
    """

    def __init__(self, x_path: Path, y_path: Path) -> None:
        if not x_path.exists():
            raise FileNotFoundError(f"Missing file: {x_path}")
        if not y_path.exists():
            raise FileNotFoundError(f"Missing file: {y_path}")

        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")

        if self.x.ndim != 3 or self.x.shape[2] != 63:
            raise ValueError(
                f"Expected X shape (N, T, 63), got {self.x.shape} from {x_path}"
            )

        if self.y.ndim != 1:
            raise ValueError(
                f"Expected y shape (N,), got {self.y.shape} from {y_path}"
            )

        if len(self.x) != len(self.y):
            raise ValueError(
                f"X/y sample count mismatch: {len(self.x)} vs {len(self.y)}"
            )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.array(self.x[index], dtype=np.float32, copy=True))
        y = torch.tensor(int(self.y[index]), dtype=torch.long)
        return x, y


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
        last_hidden = hidden[-1]
        return self.classifier(last_hidden)


def print_dataset_distribution(
    dataset: GestureDataset,
    class_names: list[str],
    split_name: str,
) -> np.ndarray:
    labels = np.asarray(dataset.y, dtype=np.int64)
    counts = np.bincount(labels, minlength=len(class_names))

    invalid = labels[(labels < 0) | (labels >= len(class_names))]
    if len(invalid) > 0:
        raise ValueError(
            f"{split_name}: y contains label ids not present in label_map: "
            f"{np.unique(invalid).tolist()}"
        )

    print(f"\n[{split_name}] samples: {len(dataset):,}")
    for class_id, class_name in enumerate(class_names):
        print(f"  {class_id:>2} | {class_name:<22} | {counts[class_id]:,}")

    return counts


def make_class_weights(class_counts: np.ndarray) -> torch.Tensor:
    if np.any(class_counts == 0):
        raise ValueError(
            f"Cannot calculate class weights because a class has zero samples: "
            f"{class_counts.tolist()}"
        )

    total = class_counts.sum()
    num_classes = len(class_counts)
    weights = total / (num_classes * class_counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    progress = tqdm(loader, desc="train", leave=False)

    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        predictions = logits.argmax(dim=1)

        total_loss += loss.item() * batch_size
        total_correct += int((predictions == y).sum().item())
        total_count += batch_size

        progress.set_postfix(
            loss=f"{total_loss / total_count:.4f}",
            acc=f"{total_correct / total_count:.4f}",
        )

    return total_loss / total_count, total_correct / total_count


@torch.inference_mode()
def run_validation(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, torch.Tensor]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)
        predictions = logits.argmax(dim=1)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += int((predictions == y).sum().item())
        total_count += batch_size

        for true_label, pred_label in zip(y.cpu(), predictions.cpu()):
            confusion[true_label, pred_label] += 1

    return total_loss / total_count, total_correct / total_count, confusion


def print_per_class_accuracy(
    confusion: torch.Tensor,
    class_names: list[str],
) -> None:
    print("\n=== Validation Accuracy by Class ===")

    for class_id, class_name in enumerate(class_names):
        total = int(confusion[class_id].sum().item())
        correct = int(confusion[class_id, class_id].item())
        accuracy = correct / total if total > 0 else 0.0

        print(
            f"{class_id:>2} | {class_name:<22} | "
            f"{correct:>5}/{total:<5} | {accuracy:.4f}"
        )


def print_confusion_matrix(
    confusion: torch.Tensor,
    class_names: list[str],
) -> None:
    confusion_np = confusion.numpy()

    print("\n=== Confusion Matrix ===")
    print("rows = true label, columns = predicted label")

    header = "true\\pred".ljust(26)
    for i in range(len(class_names)):
        header += f"{i:>8}"
    print(header)

    for i, name in enumerate(class_names):
        row = f"{i} {name[:22]:<22}"
        for value in confusion_np[i]:
            row += f"{value:>8}"
        print(row)

    print("\n=== Class Index ===")
    for i, name in enumerate(class_names):
        print(f"{i}: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GRU on model-ready Armed-4 gesture dataset."
    )

    parser.add_argument(
        "--data-root",
        default="data/model_ready/armed4_24f",
        help="Directory containing X_train.npy, y_train.npy, X_val.npy, y_val.npy, label_map.json",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use inverse-frequency class weighting in CrossEntropyLoss.",
    )
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--output-dir", default="outputs")

    args = parser.parse_args()
    set_seed(args.seed)

    data_root = resolve_path(args.data_root)

    train_dataset = GestureDataset(
        x_path=data_root / "X_train.npy",
        y_path=data_root / "y_train.npy",
    )
    val_dataset = GestureDataset(
        x_path=data_root / "X_val.npy",
        y_path=data_root / "y_val.npy",
    )

    if train_dataset.x.shape[1:] != val_dataset.x.shape[1:]:
        raise ValueError(
            f"Train/val input shape mismatch: "
            f"{train_dataset.x.shape[1:]} vs {val_dataset.x.shape[1:]}"
        )

    class_names = read_label_map(data_root / "label_map.json")
    num_classes = len(class_names)
    sequence_length = int(train_dataset.x.shape[1])
    input_size = int(train_dataset.x.shape[2])

    train_counts = print_dataset_distribution(train_dataset, class_names, "train")
    print_dataset_distribution(val_dataset, class_names, "validation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=== Dataset / Device ===")
    print("data root:", data_root.resolve())
    print("X train shape:", train_dataset.x.shape)
    print("X val shape:", val_dataset.x.shape)
    print("sequence length:", sequence_length)
    print("input size:", input_size)
    print("classes:", class_names)
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    model = GRUClassifier(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_classes=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"trainable parameters: {parameter_count:,}")

    if args.use_class_weights:
        class_weights = make_class_weights(train_counts).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("class weights:", class_weights.cpu().numpy().round(4).tolist())
    else:
        class_weights = None
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    model_dir = resolve_path(args.model_dir)
    output_dir = resolve_path(args.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"gru_armed4_{sequence_length}f"
    if args.use_class_weights:
        run_name += "_weighted"

    checkpoint_path = model_dir / f"{run_name}_best.pt"
    history_path = output_dir / f"{run_name}_history.json"
    metrics_path = output_dir / f"{run_name}_metrics.json"

    history: list[dict[str, float | int]] = []
    best_val_accuracy = -1.0
    best_confusion: torch.Tensor | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    if args.epochs <= 0:
        verify_path = output_dir / "verify_armed4_train_gru.json"
        verify = {
            "mode": "verify_only",
            "status": "ok",
            "data_root": str(data_root),
            "num_classes": num_classes,
            "class_names": class_names,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "sequence_length": sequence_length,
            "input_size": input_size,
            "parameter_count": parameter_count,
        }
        with verify_path.open("w", encoding="utf-8") as f:
            json.dump(verify, f, indent=2, ensure_ascii=False)
        print("\n=== Verify only ===")
        print("epochs <= 0, skipped training and validation.")
        print(f"Verification summary saved to: {verify_path}")
        return

    print("\n=== Training ===")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy, confusion = run_validation(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )

        result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(result)

        print(
            f"Epoch {epoch:>2}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f} acc {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_confusion = confusion.clone()
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "input_size": input_size,
                "sequence_length": sequence_length,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "class_names": class_names,
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_accuracy,
                "data_root": str(data_root),
                "use_class_weights": args.use_class_weights,
                "class_weights": (
                    class_weights.detach().cpu().tolist()
                    if class_weights is not None
                    else None
                ),
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  saved best model: {checkpoint_path}")

        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print(
                    f"\nEarly stopping: validation accuracy가 "
                    f"{args.patience} epoch 동안 개선되지 않았습니다."
                )
                break

    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    metrics = {
        "run_name": run_name,
        "task": "Armed-state 4-class gesture classifier",
        "data_root": str(data_root),
        "class_names": class_names,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "confusion_matrix": (
            best_confusion.tolist() if best_confusion is not None else None
        ),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n=== Finished ===")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Model saved to: {checkpoint_path}")
    print(f"History saved to: {history_path}")
    print(f"Metrics saved to: {metrics_path}")

    if best_confusion is not None:
        print_per_class_accuracy(
            confusion=best_confusion,
            class_names=class_names,
        )
        print_confusion_matrix(
            confusion=best_confusion,
            class_names=class_names,
        )


if __name__ == "__main__":
    main()
