"""Train the student landmark regressor.

Phase 3 (clean parity):  --robust off  -> match MediaPipe/GT on clean crops.
Phase 4 (robustness):    --robust on   -> TrainAugmentor (low-light + motion blur).

Each epoch reports val landmark NME (crop units) AND end-to-end macro-F1 through
the FROZEN classifier (reconstruct crop->image landmarks via affine, then
relative_wrist_scale). End-to-end F1 is the metric we actually care about.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "degrade"))
sys.path.insert(0, str(ROOT / "student"))
from frozen_mlp import FrozenClassifier, relative_wrist_scale   # noqa: E402
from transforms import TrainAugmentor                            # noqa: E402
from model import HandLandmarkNet, count_params                  # noqa: E402
from dataset import HandCropDataset                              # noqa: E402


def nme(pred, target):
    """Mean per-landmark L2 in crop units (pred,target: (B,42))."""
    p = pred.reshape(-1, 21, 2); t = target.reshape(-1, 21, 2)
    return torch.norm(p - t, dim=2).mean().item()


@torch.no_grad()
def evaluate(model, loader, clf, device):
    model.eval()
    nmes, preds_lm, affines, labels = [], [], [], []
    for img, target, affine, label_idx in loader:
        img = img.to(device)
        pred = model(img).cpu()
        nmes.append(nme(pred, target) * len(img))
        preds_lm.append(pred.numpy()); affines.append(affine.numpy())
        labels.append(np.asarray(label_idx))
    n = len(loader.dataset)
    lm = np.concatenate(preds_lm).reshape(-1, 21, 2)
    aff = np.concatenate(affines); y = np.concatenate(labels)
    # reconstruct original-image-normalized landmarks via affine
    lm_img = np.empty_like(lm)
    lm_img[:, :, 0] = aff[:, 0:1] + lm[:, :, 0] * aff[:, 2:3]
    lm_img[:, :, 1] = aff[:, 1:2] + lm[:, :, 1] * aff[:, 3:4]
    pred_cls = clf.predict_from_landmarks(lm_img)
    from sklearn.metrics import f1_score, accuracy_score
    mask = y >= 0
    f1 = f1_score(y[mask], pred_cls[mask], average="macro")
    acc = accuracy_score(y[mask], pred_cls[mask])
    return sum(nmes) / n, f1, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--robust", action="store_true", help="enable degradation augmentation")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "cache_v2/student"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = FrozenClassifier(device="cpu")
    l2i = clf.label_to_idx

    aug = TrainAugmentor(seed=0) if args.robust else None
    train_ds = HandCropDataset("train", label_to_idx=l2i, augmentor=aug)
    val_ds = HandCropDataset(args.val_split, label_to_idx=l2i)
    if args.limit:
        train_ds.df = train_ds.df.head(args.limit).reset_index(drop=True)
        val_ds.df = val_ds.df.head(max(256, args.limit // 4)).reset_index(drop=True)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} robust={args.robust}")

    train_dl = DataLoader(train_ds, args.bs, shuffle=True, num_workers=args.workers,
                          pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, args.bs, shuffle=False, num_workers=args.workers)

    model = HandLandmarkNet(width=args.width).to(device)
    print(f"[model] width={args.width} params={count_params(model):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossfn = nn.SmoothL1Loss(beta=0.01)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    best_f1 = -1
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for img, target, _, _ in train_dl:
            img, target = img.to(device), target.to(device)
            opt.zero_grad()
            loss = lossfn(model(img), target)
            loss.backward(); opt.step()
            tot += loss.item() * len(img)
        sched.step()
        v_nme, v_f1, v_acc = evaluate(model, val_dl, clf, device)
        tag = ""
        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save({"state_dict": model.state_dict(), "width": args.width,
                        "val_f1": v_f1}, out / "best_student.pt")
            tag = " *"
        print(f"ep{ep:02d} train_loss={tot/len(train_ds):.5f} "
              f"val_nme={v_nme:.4f} val_F1={v_f1:.4f} val_acc={v_acc:.4f}{tag}")
    print(f"[done] best val_F1={best_f1:.4f}  (frozen-MLP target=0.99114)")


if __name__ == "__main__":
    main()
