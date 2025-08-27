# prep_ut_har.py
# Preprocess UT_HAR into train/val/test arrays ready for DataLoader.

import os, numpy as np
import torch

ROOT = r"C:\Users\Arun AM\WiFi-CSI-Sensing-Benchmark\Benchmark\Data\UT_HAR"  # <-- change this

def load_split(name):
    X = np.load(os.path.join(ROOT, "data", f"X_{name}.csv"))   # NumPy binary despite .csv
    y = np.load(os.path.join(ROOT, "label", f"Y_{name}.csv"))
    X = X.reshape(len(X), 1, 250, 90).astype(np.float32)       # (N,1,250,90)
    return X, y.astype(np.int64)

print("[*] Loading splits...")
Xtr, ytr = load_split("train")
Xva, yva = load_split("val")
Xte, yte = load_split("test")

# ---- Normalization (standardization) using TRAIN ONLY ----
# flatten spatial dims to compute mean/std per-channel (here channel=1)
mean = Xtr.mean(axis=(0,2,3), keepdims=True)   # shape (1,1,1,1)
std  = Xtr.std(axis=(0,2,3), keepdims=True) + 1e-6
print(f"Train mean={float(mean.squeeze()):.4f}, std={float(std.squeeze()):.4f}")

def standardize(X):
    return (X - mean) / std

Xtr = standardize(Xtr)
Xva = standardize(Xva)
Xte = standardize(Xte)

# ---- (Optional) Min-Max to [0,1] instead of standardization ----
# Uncomment if you prefer:
# lo, hi = Xtr.min(), Xtr.max()
# Xtr = (Xtr - lo) / (hi - lo + 1e-6)
# Xva = (Xva - lo) / (hi - lo + 1e-6)
# Xte = (Xte - lo) / (hi - lo + 1e-6)

# ---- Save preprocessed arrays ----
OUT = os.path.join(ROOT, "_preprocessed")
os.makedirs(OUT, exist_ok=True)

np.save(os.path.join(OUT, "X_train.npy"), Xtr)
np.save(os.path.join(OUT, "y_train.npy"), ytr)
np.save(os.path.join(OUT, "X_val.npy"),   Xva)
np.save(os.path.join(OUT, "y_val.npy"),   yva)
np.save(os.path.join(OUT, "X_test.npy"),  Xte)
np.save(os.path.join(OUT, "y_test.npy"),  yte)

# Also save torch tensors if you prefer .pt
torch.save(torch.from_numpy(Xtr), os.path.join(OUT, "X_train.pt"))
torch.save(torch.from_numpy(Xva), os.path.join(OUT, "X_val.pt"))
torch.save(torch.from_numpy(Xte), os.path.join(OUT, "X_test.pt"))
torch.save(torch.from_numpy(ytr), os.path.join(OUT, "y_train.pt"))
torch.save(torch.from_numpy(yva), os.path.join(OUT, "y_val.pt"))
torch.save(torch.from_numpy(yte), os.path.join(OUT, "y_test.pt"))

print(f"[*] Saved to {OUT}")
