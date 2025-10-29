import yaml
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
from typing import Dict, Any, Tuple, List
from pathlib import Path
from src.data.mmfi import make_dataset, make_dataloader  # official toolbox  ✅

def set_seed(seed: int):
    import os, random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); os.environ["PYTHONHASHSEED"]=str(seed)

def build_mmfi_dataset(dataset_root: str, cfg: Dict[str, Any]):
    dataset_root = Path(dataset_root)
    m = cfg["mmfi"]

    # Ensure required nested keys exist for manual_split
    def _ensure_actions():
        return [f"A{i:02d}" for i in range(1, 28)]
    def _ensure_subjects_e01():
        return [f"S{i:02d}" for i in range(1, 11)]

    manual = m.get("manual_split", {})
    train_ds = manual.get("train_dataset", {})
    val_ds   = manual.get("val_dataset", {})
    train_ds.setdefault("subjects", _ensure_subjects_e01())
    train_ds.setdefault("actions",  _ensure_actions())
    val_ds.setdefault("subjects",   [])
    val_ds.setdefault("actions",    _ensure_actions())
    manual["train_dataset"] = train_ds
    manual["val_dataset"]   = val_ds

    mmfi_cfg = {
        "modality": m.get("modality", "wifi-csi"),
        "data_unit": m.get("data_unit", "sequence"),
        "protocol": m.get("protocol", "protocol3"),
        "split_to_use": m.get("split_to_use", "manual_split"),
        "manual_split": manual,
    }

    # Your toolbox signature is (dataset_root, config)
    train_ds, val_ds = make_dataset(str(dataset_root), mmfi_cfg)
    return train_ds, val_ds

def infer_batch_keys(sample) -> Tuple[str, str, str]:
    """
    Detect the data key (wifi-csi), label key (action), and subject key.
    Works with dict-style items from the MM-Fi toolbox.
    """
    if isinstance(sample, dict):
        keys = list(sample.keys())
        # Heuristics
        data_key = next((k for k in keys if "wifi" in k or "csi" in k), None)
        label_key = next((k for k in keys if "label" in k or "action" in k), None)
        subj_key = next((k for k in keys if "subject" in k.lower() or k.lower()=="sid"), None)
        if data_key is None:
            raise KeyError(f"Cannot find CSI data key in sample keys={keys}")
        if label_key is None:
            raise KeyError(f"Cannot find label key in sample keys={keys}")
        if subj_key is None:
            # Fallback: try to parse from a path string if provided
            subj_key = next((k for k in keys if "path" in k.lower()), None)
        return data_key, label_key, subj_key
    raise TypeError("Unexpected sample type; expected dict from MM-Fi toolbox.")

def make_random_split_indices(N: int, test_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_test = max(1, int(N * test_ratio))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return sorted(train_idx.tolist()), sorted(test_idx.tolist())

def loso_subject_lists():
    # E01 subjects S01..S10
    return [f"S{i:02d}" for i in range(1, 11)]

def filter_by_subject(dataset, subj_key: str, subj_list: List[str]):
    indices = []
    for i in range(len(dataset)):
        item = dataset[i]
        sid = str(item[subj_key]) if subj_key in item else str(item.get("subject", ""))
        for s in subj_list:
            if s in sid:
                indices.append(i); break
    return indices

def make_loaders_from_indices(dataset, train_idx, test_idx, batch_size, num_workers=2, pin_memory=True):
    train_ds = Subset(dataset, train_idx)
    test_ds  = Subset(dataset, test_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader