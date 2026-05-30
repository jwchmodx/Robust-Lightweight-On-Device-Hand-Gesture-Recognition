import numpy as np

x = np.load("data/processed/train/60572.npy")

print("shape:", x.shape)
print("first frame:", x[10][:10])
print("non-zero frames:", (x.sum(axis=1) != 0).sum())