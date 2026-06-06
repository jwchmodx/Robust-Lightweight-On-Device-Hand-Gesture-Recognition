"""Contract harness for the frozen HaGRID static MLP classifier.

This is the interface our landmark extractor must satisfy. It:
  - reproduces the teammate's `relative_wrist_scale` feature transform EXACTLY,
  - loads the frozen MLP (relative_wrist_scale_42d_single_hand_only_xlarge),
  - exposes `predict_from_landmarks(raw_landmarks)` so any extractor (MediaPipe
    or our future student) can be scored end-to-end through the real classifier.

Run directly to verify it reproduces the reported test macro-F1 = 0.99114.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# --- paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
HAGRID = REPO_ROOT / "models/static/hagrid"
MODEL_DIR = (HAGRID / "experiments"
             / "relative_wrist_scale_42d_single_hand_only_xlarge_norm_bn_sch_do0p2_seed1"
             / "seed_1")
NPY = HAGRID / "npy_dataset"

NUM_LANDMARKS, POINT_DIM = 21, 2


# --- feature contract (verbatim from make_hagrid_npy_rel.py) ------------------
def relative_wrist_scale(arr):
    """arr: (..., 21, 2) raw image-coords -> (..., 42) relative_wrist_scale.

    wrist-relative, divided by distance to the farthest landmark. 2D only.
    """
    arr = np.asarray(arr, dtype=np.float32)
    wrist = arr[..., 0:1, :]
    relative = arr - wrist
    scale = np.linalg.norm(relative, axis=-1).max(axis=-1, keepdims=True)  # (...,1)
    processed = relative / (scale[..., None] + 1e-6)
    return processed.reshape(*arr.shape[:-2], NUM_LANDMARKS * POINT_DIM)


# --- frozen model ------------------------------------------------------------
class MLPGestureClassifier(nn.Module):
    """Replicates hagrid/src/train_landmark_experiment.py:MLPGestureClassifier."""
    def __init__(self, input_dim, hidden_dims, num_classes, dropout, use_bn):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FrozenClassifier:
    def __init__(self, model_dir=MODEL_DIR, device="cpu"):
        cfg = json.loads((model_dir / "config.json").read_text())
        self.target_names = cfg["target_names"]
        self.label_to_idx = {n: i for i, n in enumerate(self.target_names)}
        self.mean = np.load(model_dir / "mean.npy").astype(np.float32)
        self.std = np.load(model_dir / "std.npy").astype(np.float32)
        self.device = device
        self.model = MLPGestureClassifier(
            cfg["input_dim"], cfg["hidden_dims"], cfg["num_classes"],
            cfg["dropout"], use_bn=not cfg.get("no_batchnorm", False),
        ).to(device)
        # trusted local checkpoint (teammate's); it embeds numpy objects so weights_only=False
        sd = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
        if isinstance(sd, dict):
            for k in ("model_state_dict", "state_dict", "model"):
                if k in sd:
                    sd = sd[k]
                    break
        self.model.load_state_dict(sd)
        self.model.eval()

    def predict_from_features(self, feats42):
        x = (np.asarray(feats42, np.float32) - self.mean) / self.std
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x).to(self.device))
        return logits.argmax(1).cpu().numpy()

    def predict_from_landmarks(self, raw_landmarks):
        """raw_landmarks: (N, 21, 2) image-coords -> predicted class indices.

        THE interface our extractor must feed: it just needs accurate 21 (x,y).
        """
        return self.predict_from_features(relative_wrist_scale(raw_landmarks))


# --- contract test -----------------------------------------------------------
def main():
    from sklearn.metrics import f1_score, accuracy_score
    clf = FrozenClassifier()
    X = np.load(NPY / "X_test.npy").reshape(-1, NUM_LANDMARKS, POINT_DIM)  # raw coords
    y = np.load(NPY / "y_test.npy")                                        # str labels
    keep = np.array([lbl in clf.label_to_idx for lbl in y])
    X, y = X[keep], y[keep]
    y_idx = np.array([clf.label_to_idx[lbl] for lbl in y])
    print(f"[contract] test samples (26-class): {len(y_idx)}")

    pred = clf.predict_from_landmarks(X)   # raw landmarks -> OUR transform -> frozen MLP
    f1 = f1_score(y_idx, pred, average="macro")
    acc = accuracy_score(y_idx, pred)
    print(f"[contract] reproduced macro-F1 = {f1:.5f}  acc = {acc:.5f}")
    print(f"[contract] target               = 0.99114")
    ok = abs(f1 - 0.99114) < 1e-3
    print(f"[contract] {'PASS' if ok else 'MISMATCH'} "
          f"-> relative_wrist_scale transform is {'exact' if ok else 'WRONG'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
