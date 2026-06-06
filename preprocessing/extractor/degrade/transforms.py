"""Adverse-condition degradations: low-light and motion-blur (FINALIZED set).

Each is severity-parameterized in [0,1] so the SAME code serves:
  - train-time augmentation (random condition + random severity), and
  - eval-time graded robustness sweeps (fixed severities).

Operates on uint8 BGR images (OpenCV). Geometry is untouched, so landmark
targets stay valid (only pixels are degraded — the core robustness setup).
"""
import cv2
import numpy as np


def low_light(img, severity, rng=None):
    """Darken via gamma>1 + gain<1, plus mild sensor noise (low light is noisy)."""
    rng = rng or np.random.default_rng()
    s = float(np.clip(severity, 0, 1))
    gamma = 1.0 + 2.2 * s          # 1.0 -> 3.2
    gain = 1.0 - 0.55 * s          # 1.0 -> 0.45
    x = (img.astype(np.float32) / 255.0) ** gamma
    x = x * gain
    noise_sigma = 0.02 * s
    if noise_sigma > 0:
        x = x + rng.normal(0, noise_sigma, x.shape).astype(np.float32)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def motion_blur(img, severity, rng=None):
    """Directional blur; kernel length grows with severity, random angle."""
    rng = rng or np.random.default_rng()
    s = float(np.clip(severity, 0, 1))
    k = int(round(3 + 18 * s))     # 3 -> 21 px
    if k <= 2:
        return img.copy()
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    angle = float(rng.uniform(0, 180))
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    ssum = kernel.sum()
    kernel = kernel / ssum if ssum > 1e-6 else kernel
    return cv2.filter2D(img, -1, kernel)


CONDITIONS = {"low_light": low_light, "motion_blur": motion_blur}
EVAL_SEVERITIES = [0.0, 0.25, 0.5, 0.75, 1.0]   # 0.0 == clean


class TrainAugmentor:
    """Random degradation for training the robust student.

    With prob `p_clean` returns the image unchanged; otherwise applies a random
    subset of conditions at random severity. Mixing clean + degraded keeps clean
    parity while building robustness.
    """
    def __init__(self, p_clean=0.3, p_each=0.6, sev_range=(0.2, 1.0), seed=None):
        self.p_clean = p_clean
        self.p_each = p_each
        self.sev_range = sev_range
        self.rng = np.random.default_rng(seed)

    def __call__(self, img):
        if self.rng.random() < self.p_clean:
            return img
        out = img
        for fn in CONDITIONS.values():
            if self.rng.random() < self.p_each:
                sev = self.rng.uniform(*self.sev_range)
                out = fn(out, sev, self.rng)
        return out


def degrade(img, condition, severity, rng=None):
    """Eval-time single-condition degradation at a fixed severity."""
    if severity == 0.0 or condition is None:
        return img
    return CONDITIONS[condition](img, severity, rng)
