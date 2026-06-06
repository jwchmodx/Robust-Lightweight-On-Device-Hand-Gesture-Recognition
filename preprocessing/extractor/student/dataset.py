"""Dataset of JPEG hand crops + landmark targets, with optional degradation.

Returns per item:
  img        float32 tensor (3,128,128), /255, RGB
  target     float32 (42,)   crop-frame landmarks in [0,1]
  affine     float32 (4,)    crop_to_img [ox,oy,sx,sy] (for end-to-end eval)
  label_idx  int             frozen-classifier class index
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]


class HandCropDataset(Dataset):
    def __init__(self, split, cache_dir=ROOT / "cache_v2", label_to_idx=None,
                 augmentor=None, degrade_fn=None):
        self.cache = Path(cache_dir)
        self.crops = self.cache / "crops"
        df = pd.read_parquet(self.cache / "targets.parquet")
        self.df = df[df.split == split].reset_index(drop=True)
        self.label_to_idx = label_to_idx or {}
        self.augmentor = augmentor      # TrainAugmentor (train only)
        self.degrade_fn = degrade_fn    # fixed eval degradation: img->img

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        path = self.crops / r.split / r.label / f"{r.image_id}.jpg"
        bgr = cv2.imread(str(path))
        if bgr is None:
            bgr = np.zeros((128, 128, 3), np.uint8)
        if self.augmentor is not None:
            bgr = self.augmentor(bgr)
        if self.degrade_fn is not None:
            bgr = self.degrade_fn(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = torch.from_numpy(rgb.transpose(2, 0, 1))
        target = torch.tensor(np.asarray(r.landmarks_crop, np.float32))
        affine = torch.tensor(np.asarray(r.crop_to_img, np.float32))
        label_idx = self.label_to_idx.get(r.label, -1)
        return img, target, affine, label_idx
