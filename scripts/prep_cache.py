# scripts/prep_cache.py
import argparse
import yaml
import numpy as np
import torch
import os
from pathlib import Path
from tqdm import tqdm

from src.data.mmfi_wrap import build_mmfi_dataset, infer_batch_keys
from src.data.spectrogram import (
    SpectrogramCache, to_amp_phase, make_csi_spectrogram, norm_spec
)

def extract_csi_numpy(item, data_key, merge_packets_to_time: bool = True):
    """
    Normalize MM-Fi WiFi-CSI to [T, L, S].
    If input is 4D [Frames, Links, Subcarriers, Packets], either:
      - merge_packets_to_time=True: reshape to [Frames*Packets, Links, Subcarriers]
      - merge_packets_to_time=False: average over Packets -> [Frames, Links, Subcarriers]
    Then reorder axes to [T, L, S] if needed.
    """
    data = item[data_key]
    arr = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)

    if arr.ndim == 4:
        sizes = arr.shape  # e.g., (297, 3, 114, 10)
        # Heuristics to identify axes
        f_axis = int(np.argmax(sizes))  # time (frames) is usually the largest
        pkt_axes = [i for i, s in enumerate(sizes) if s == 10]  # packets per frame
        pkt_axis = pkt_axes[-1] if pkt_axes else 3  # default to last
        # link = smallest axis not equal to time or packet
        order_by_size = sorted(range(4), key=lambda i: sizes[i])
        l_axis = order_by_size[0]
        if l_axis in (f_axis, pkt_axis):
            l_axis = order_by_size[1]
        # remaining is subcarrier axis
        s_axis = [i for i in range(4) if i not in (f_axis, l_axis, pkt_axis)][0]

        # Reorder to [F, L, S, P]
        arr = np.transpose(arr, (f_axis, l_axis, s_axis, pkt_axis))  # [F,L,S,P]
        F, L, S, P = arr.shape
        if merge_packets_to_time:
            arr = arr.reshape(F * P, L, S).astype(np.float32)  # [T, L, S] with T=F*P
        else:
            arr = arr.mean(axis=-1, keepdims=False).astype(np.float32)  # [F, L, S]

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D CSI after reduction, got shape={arr.shape}")

    # Reorder to [T, L, S] if needed
    dims = arr.shape
    t_axis = int(np.argmax(dims))
    order_by_size = sorted(range(3), key=lambda i: dims[i])
    l_axis = order_by_size[0] if order_by_size[0] != t_axis else order_by_size[1]
    s_axis = [i for i in range(3) if i not in (t_axis, l_axis)][0]
    arr = np.transpose(arr, (t_axis, l_axis, s_axis)).astype(np.float32)  # [T,L,S]
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
                           "Check mmfi.manual_split subjects/actions in your YAML.")
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
    merge_packets_to_time = bool(sp_cfg.get("merge_packets_to_time", True))

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
        amp, _ = to_amp_phase(extract_csi_numpy(item, data_key, merge_packets_to_time=merge_packets_to_time))
        # Spectrogram -> [C,F,T']
        spec = make_csi_spectrogram(amp, sp_cfg)
        # Normalize instance-wise
        spec = norm_spec(spec, sp_cfg.get("normalize", "instance"))
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