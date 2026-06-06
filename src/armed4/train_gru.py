""" 
python src/train_gru.py \
  --subset control \
  --epochs 30 \
  --batch-size 256 \
  --hidden-size 128 
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


CONTROL_LABELS = [
    "No gesture",
    "Doing other things",
    "Swiping Up",
    "Swiping Down",
    "Swiping Left",
    "Swiping Right",
    "Stop Sign",
]

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


def read_annotation_labels(annotation_path: Path) -> list[str]:
    labels: list[str] = []

    with annotation_path.open("r", encoding="utf-8") as f:
        sample = f.readline()
        delimiter = ";" if ";" in sample else ","
        f.seek(0)

        reader = csv.reader(f, delimiter=delimiter)

        for row in reader:
            if len(row) < 2:
                continue

            label = row[1].strip()
            labels.append(label)

    return labels


def verify_saved_labels(
    saved_y: np.ndarray,
    annotation_labels: list[str],
    original_label_map: dict[str, int],
    split_name: str,
) -> None:
    """
    X/y 생성 당시의 라벨 매핑이 annotation 기준 정렬 매핑과 같은지 확인한다.
    build_train_dataset.py에서 sorted(set(labels))를 사용했다는 전제와 맞는다.
    """
    expected = np.array(
        [original_label_map[label] for label in annotation_labels],
        dtype=np.int64,
    )

    if len(saved_y) != len(expected):
        raise ValueError(
            f"{split_name}: y 길이({len(saved_y)})와 "
            f"annotation 길이({len(expected)})가 다릅니다."
        )

    if not np.array_equal(np.asarray(saved_y), expected):
        raise ValueError(
            f"{split_name}: y 라벨 매핑이 annotation 기반 매핑과 일치하지 않습니다. "
            "데이터셋 생성 시 사용한 label_map 방식을 확인하세요."
        )


class GestureDataset(Dataset):
    def __init__(
        self,
        x_path: Path,
        y_path: Path,
        indices: np.ndarray,
        original_to_target: dict[int, int],
    ) -> None:
        # 큰 npy를 통째로 RAM에 복사하지 않고 필요한 샘플만 읽음
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")

        self.indices = indices
        self.original_to_target = original_to_target

        if self.x.ndim != 3 or self.x.shape[1:] != (32, 63):
            raise ValueError(
                f"Expected X shape (N, 32, 63), got {self.x.shape} from {x_path}"
            )

        if len(self.x) != len(self.y):
            raise ValueError(
                f"X/y sample count mismatch: {len(self.x)} vs {len(self.y)}"
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source_index = int(self.indices[index])

        # mmap 배열을 그대로 Tensor로 감싸면 읽기 전용 경고가 날 수 있어 copy 사용
        x = torch.from_numpy(
            np.array(self.x[source_index], dtype=np.float32, copy=True)
        )

        original_label = int(self.y[source_index])
        target_label = self.original_to_target[original_label]
        y = torch.tensor(target_label, dtype=torch.long)

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

        # 마지막 GRU layer의 최종 hidden state: (batch, hidden_size)
        last_hidden = hidden[-1]

        logits = self.classifier(last_hidden)
        return logits


def prepare_split(
    x_path: Path,
    y_path: Path,
    annotation_path: Path,
    original_label_map: dict[str, int],
    target_labels: list[str],
    split_name: str,
) -> GestureDataset:
    if not x_path.exists():
        raise FileNotFoundError(f"Missing file: {x_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"Missing file: {y_path}")

    annotation_labels = read_annotation_labels(annotation_path)
    saved_y = np.load(y_path, mmap_mode="r")

    verify_saved_labels(
        saved_y=saved_y,
        annotation_labels=annotation_labels,
        original_label_map=original_label_map,
        split_name=split_name,
    )

    target_label_map = {
        label: index for index, label in enumerate(target_labels)
    }

    original_to_target = {
        original_label_map[label]: target_label_map[label]
        for label in target_labels
    }

    indices = np.array(
        [
            index
            for index, label in enumerate(annotation_labels)
            if label in target_label_map
        ],
        dtype=np.int64,
    )

    print(f"\n[{split_name}] selected samples: {len(indices):,}")

    counts = Counter(
        annotation_labels[index] for index in indices
    )

    for label in target_labels:
        print(f"  {target_label_map[label]:>2} | {label:<20} | {counts[label]:,}")

    return GestureDataset(
        x_path=x_path,
        y_path=y_path,
        indices=indices,
        original_to_target=original_to_target,
    )


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
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        predictions = logits.argmax(dim=1)

        total_loss += loss.item() * batch_size
        total_correct += (predictions == y).sum().item()
        total_count += batch_size

        progress.set_postfix(
            loss=f"{total_loss / total_count:.4f}",
            acc=f"{total_correct / total_count:.4f}",
        )

    epoch_loss = total_loss / total_count
    epoch_accuracy = total_correct / total_count

    return epoch_loss, epoch_accuracy


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

    confusion = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
    )

    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        predictions = logits.argmax(dim=1)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == y).sum().item()
        total_count += batch_size

        true_cpu = y.cpu()
        pred_cpu = predictions.cpu()

        for true_label, pred_label in zip(true_cpu, pred_cpu):
            confusion[true_label, pred_label] += 1

    epoch_loss = total_loss / total_count
    epoch_accuracy = total_correct / total_count

    return epoch_loss, epoch_accuracy, confusion


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
            f"{class_id:>2} | {class_name:<20} | "
            f"{correct:>5}/{total:<5} | {accuracy:.4f}"
        )

def print_confusion_matrix(confusion, class_names):
    confusion_np = confusion.numpy()

    print("\n=== Confusion Matrix ===")
    print("rows = true label, columns = predicted label")

    header = "true\\pred".ljust(22)
    for i in range(len(class_names)):
        header += f"{i:>7}"
    print(header)

    for i, name in enumerate(class_names):
        row = f"{i} {name[:18]:<18}"
        for value in confusion_np[i]:
            row += f"{value:>7}"
        print(row)

    print("\n=== Class Index ===")
    for i, name in enumerate(class_names):
        print(f"{i}: {name}")

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-x",
        default="data/processed/train/X_train.npy",
    )
    parser.add_argument(
        "--train-y",
        default="data/processed/train/y_train.npy",
    )
    parser.add_argument(
        "--val-x",
        default="data/processed/validation/X_val.npy",
    )
    parser.add_argument(
        "--val-y",
        default="data/processed/validation/y_val.npy",
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
        "--subset",
        choices=["control", "all"],
        default="control",
        help="control: 제어용 7개 제스처만 학습, all: 전체 라벨 학습",
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
        "--model-dir",
        default="models",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    train_annotation_path = resolve_path(args.train_annotation)
    val_annotation_path = resolve_path(args.val_annotation)

    train_annotation_labels = read_annotation_labels(train_annotation_path)
    val_annotation_labels = read_annotation_labels(val_annotation_path)

    original_label_names = sorted(set(train_annotation_labels))

    if sorted(set(val_annotation_labels)) != original_label_names:
        raise ValueError(
            "Train과 validation의 전체 label 집합이 다릅니다."
        )

    original_label_map = {
        label: index for index, label in enumerate(original_label_names)
    }

    if args.subset == "control":
        target_labels = CONTROL_LABELS
    else:
        target_labels = original_label_names

    missing_labels = [
        label for label in target_labels
        if label not in original_label_map
    ]

    if missing_labels:
        raise ValueError(f"Annotation에 존재하지 않는 라벨: {missing_labels}")

    train_dataset = prepare_split(
        x_path=resolve_path(args.train_x),
        y_path=resolve_path(args.train_y),
        annotation_path=train_annotation_path,
        original_label_map=original_label_map,
        target_labels=target_labels,
        split_name="train",
    )

    val_dataset = prepare_split(
        x_path=resolve_path(args.val_x),
        y_path=resolve_path(args.val_y),
        annotation_path=val_annotation_path,
        original_label_map=original_label_map,
        target_labels=target_labels,
        split_name="validation",
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)
    print("Classes:", len(target_labels))
    print("Class names:", target_labels)

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
        input_size=63,
        hidden_size=args.hidden_size,
        num_classes=len(target_labels),
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Trainable parameters: {parameter_count:,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    model_dir = resolve_path(args.model_dir)
    output_dir = resolve_path(args.output_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"gru_{args.subset}"
    checkpoint_path = model_dir / f"{run_name}_best.pt"
    history_path = output_dir / f"{run_name}_history.json"

    history: list[dict[str, float | int]] = []

    best_val_accuracy = -1.0
    best_confusion = None
    epochs_without_improvement = 0

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
            num_classes=len(target_labels),
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
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "input_size": 63,
                "sequence_length": 32,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "class_names": target_labels,
                "subset": args.subset,
                "best_val_accuracy": best_val_accuracy,
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

    print("\n=== Finished ===")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Model saved to: {checkpoint_path}")
    print(f"History saved to: {history_path}")

    if best_confusion is not None:
        print_per_class_accuracy(
            confusion=best_confusion,
            class_names=target_labels,
        )

        print_confusion_matrix(
            confusion=best_confusion,
            class_names=target_labels,
        )


if __name__ == "__main__":
    main()
