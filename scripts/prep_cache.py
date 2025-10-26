# scripts/prep_cache.py
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from src.data.mmfi_wrap import build_mmfi_dataset, infer_batch_keys
from src.data.spectrogram import (
    SpectrogramCache, to_amp_phase, make_csi_spectrogram, norm_spec
)

def extract_csi_numpy(item, data_key):
    """
    Normalize MM-Fi WiFi-CSI shapes to [T, L, S].
    Handles 3D and 4D (e.g., [Frames, Links, Subcarriers, Packets=10]) inputs.
    """
    data = item[data_key]
    arr = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)

    # If 4D, reduce packet/snapshot axis via mean
    if arr.ndim == 4:
        sizes = arr.shape
        pkt_axes = [i for i, s in enumerate(sizes) if s == 10]
        pkt_axis = pkt_axes[-1] if pkt_axes else 3  # usually last
        arr = arr.mean(axis=pkt_axis, keepdims=False)  # -> 3D

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D CSI after reduction, got shape={arr.shape}")

    # Reorder to [T, L, S]:
    dims = arr.shape
    t_axis = int(np.argmax(dims))  # time is the largest
    order_by_size = sorted(range(3), key=lambda i: dims[i])
    l_axis = order_by_size[0] if order_by_size[0] != t_axis else order_by_size[1]  # links usually smallest
    s_axis = [i for i in range(3) if i not in (t_axis, l_axis)][0]
    arr = np.transpose(arr, (t_axis, l_axis, s_axis)).astype(np.float32)
    return arr

def main(args):
    cfg = yaml.safe_load(open(args.config))

    # 0) Validate dataset root
    dataset_root = Path(cfg["dataset_root"])
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

    print(f"[prep] Using dataset_root = {dataset_root}", flush=True)

    # 1) Build dataset via toolbox
    train_ds, _ = build_mmfi_dataset(str(dataset_root), cfg)
    N = len(train_ds)
    if N == 0:
        raise RuntimeError("[prep] MM-Fi dataset returned 0 sequences. "
                           "Check mmfi.manual_split.train_dataset.subjects/actions in your YAML.")
    first = train_ds[0]
    data_key, label_key, subj_key = infer_batch_keys(first)
    print(f"[prep] Found keys: data_key='{data_key}', label_key='{label_key}', subj_key='{subj_key}'", flush=True)

    # 2) Prepare cache
    cache_root = Path(cfg["cache"]["root"])
    cache_root.mkdir(parents=True, exist_ok=True)
    fmt = cfg["cache"]["format"].lower()
    overwrite = bool(cfg["cache"]["overwrite"])
    cache = SpectrogramCache(cache_root, overwrite=overwrite, format=fmt)
    print(f"[prep] Cache dir = {cache_root} (format={fmt}, overwrite={overwrite})", flush=True)

    sp_cfg = cfg["spectrogram"]

    # 3) Iterate and cache with progress bar
    wrote = 0
    example_shape = None
    print(f"[prep] Caching {N} sequences to spectrograms ...", flush=True)
    for i in tqdm(range(N), ncols=80, ascii=True):
        item = train_ds[i]
        key = f"E01_{i:06d}"
        out_path = cache._path(key)
        if out_path.exists() and not overwrite:
            continue  # keep existing
        # CSI -> [T,L,S]
        amp, _ = to_amp_phase(extract_csi_numpy(item, data_key))
        # Spectrogram -> [C,F,T']
        spec = make_csi_spectrogram(amp, sp_cfg)
        # Normalize instance-wise
        spec = norm_spec(spec, cfg["spectrogram"]["normalize"])
        cache.save(key, spec)
        wrote += 1
        if example_shape is None:
            example_shape = spec.shape

    print(f"[prep] Done. New files written: {wrote} / {N}", flush=True)
    if example_shape is not None:
        print(f"[prep] Example spectrogram shape [C,F,T']: {example_shape}", flush=True)
    else:
        print(f"[prep] Nothing new written (files already existed).", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args())