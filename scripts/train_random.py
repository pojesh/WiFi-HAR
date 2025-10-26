import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from src.data.mmfi_wrap import build_mmfi_dataset, infer_batch_keys
from src.models.vit_csi import ViTCSI
from src.utils.metrics import compute_metrics
from src.utils.common import save_json


def set_seed(seed: int):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_cached(path: str):
    # Returns numpy array [C, F, T] without loading full file into RAM for .npy
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    elif path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as f:
            return f["spec"]
    else:
        raise ValueError(f"Unknown cache file type: {path}")


def make_random_split_indices(N: int, test_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_test = max(1, int(N * test_ratio))
    test_idx = np.sort(idx[:n_test]).tolist()
    train_idx = np.sort(idx[n_test:]).tolist()
    return train_idx, test_idx


def batchify(paths, idxs, labels, subj_ids, batch_size):
    # Yields torch tensors without loading all data in RAM
    for start in range(0, len(idxs), batch_size):
        sel = idxs[start:start + batch_size]
        xs = [load_cached(paths[i]) for i in sel]  # list of numpy arrays [C,F,T]
        x = torch.from_numpy(np.stack(xs, axis=0)).float()  # [B,C,F,T]
        y = torch.tensor([labels[i] for i in sel], dtype=torch.long)
        s = torch.tensor([subj_ids[i] for i in sel], dtype=torch.long)
        yield {"x": x, "y": y, "subj": s}


def main(args):
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 42))
    torch.backends.cudnn.benchmark = True

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # 1) Build MM-Fi dataset (toolbox) and infer keys
    train_ds, _ = build_mmfi_dataset(cfg["dataset_root"], cfg)
    if len(train_ds) == 0:
        raise RuntimeError("MM-Fi train dataset is empty. Check your config subjects/actions.")

    sample0 = train_ds[0]
    data_key, label_key, subj_key = infer_batch_keys(sample0)

    # 2) Build cache paths and metadata
    cache_root = Path(cfg["cache"]["root"])
    ext = ".npy" if cfg["cache"]["format"].lower() == "npy" else ".npz"

    actions_map = {f"A{i:02d}": i - 1 for i in range(1, 28)}   # A01->0 ... A27->26
    subj_map = {f"S{i:02d}": i - 1 for i in range(1, 11)}      # S01->0 ... S10->9 (E01)

    paths, labels, subj_ids = [], [], []
    for i in range(len(train_ds)):
        key = f"E01_{i:06d}"
        p = cache_root / f"{key}{ext}"
        if not p.exists():
            raise FileNotFoundError(f"Cached spectrogram missing: {p}. Run prep_cache.py first.")
        paths.append(str(p))

        item = train_ds[i]
        # Label from action string -> int class
        act = str(item[label_key])  # e.g., 'A05'
        if act not in actions_map:
            # tolerate formats like 'A5'
            act = f"A{int(''.join([c for c in act if c.isdigit()])):02d}"
        labels.append(actions_map[act])

        # Subject ID for adversarial head
        sid = str(item.get(subj_key, item.get("subject", "")))
        sid = next((k for k in subj_map.keys() if k in sid), None)
        if sid is None:
            raise KeyError(f"Cannot parse subject from item field '{subj_key}': {item.get(subj_key)}")
        subj_ids.append(subj_map[sid])

    # 3) Random split at sequence level
    tr_idx, te_idx = make_random_split_indices(len(paths), test_ratio=0.2, seed=cfg.get("seed", 42))

    # 4) Build model with input shape inferred from one cached file
    sample_arr = load_cached(paths[0])
    in_chans, F, Tspec = sample_arr.shape
    model = ViTCSI(
        in_chans=in_chans,
        num_classes=27,
        embed_dim=cfg["model"]["embed_dim"],
        depth=cfg["model"]["depth"],
        num_heads=cfg["model"]["num_heads"],
        mlp_ratio=cfg["model"]["mlp_ratio"],
        patch_f=cfg["model"]["patch"]["freq"],
        patch_t=cfg["model"]["patch"]["time"],
        cls_token=cfg["model"]["cls_token"],
        adv_subject_classes=(cfg["adv"]["subject_classes"] if cfg["adv"]["enable"] else 0),
    ).to(device)

    # 5) Optim and AMP
    optim = AdamW(model.parameters(),
                  lr=cfg["optim"]["lr"],
                  weight_decay=cfg["optim"]["weight_decay"],
                  betas=tuple(cfg["optim"]["betas"]))
    scaler = GradScaler(enabled=cfg["optim"]["amp"])
    batch_size = int(cfg["optim"]["batch_size"])
    grad_clip = float(cfg["optim"]["grad_clip_norm"])
    epochs = int(cfg["optim"]["epochs"])

    out_dir = Path(cfg["output_root"]) / cfg["experiment_name"]
    (out_dir / "ckpts").mkdir(parents=True, exist_ok=True)

    best = {"epoch": -1, "accuracy": 0.0, "macro_f1": 0.0}

    # 6) Train/eval loop
    for epoch in range(epochs):
        model.train()
        total_loss, seen = 0.0, 0

        for batch in batchify(paths, tr_idx, labels, subj_ids, batch_size):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            s = batch["subj"].to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["optim"]["amp"]):
                logits, adv_logits = model(x, (cfg["adv"]["lambda"] if cfg["adv"]["enable"] else None))
                # Classification + optional adversarial loss
                from src.losses.adversarial import dann_loss
                loss, _ = dann_loss(
                    logits, y,
                    adv_logits=adv_logits, subj_labels=s,
                    lam=(cfg["adv"]["lambda"] if cfg["adv"]["enable"] else 0.0)
                )
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()
            total_loss += float(loss.item()) * x.size(0)
            seen += x.size(0)

        # Evaluation on test split
        model.eval()
        ys, ps = [], []
        with torch.inference_mode(), autocast(enabled=cfg["optim"]["amp"]):
            for batch in batchify(paths, te_idx, labels, subj_ids, batch_size):
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                logits, _ = model(x, None)
                pred = logits.argmax(dim=1)
                ys.append(y.cpu())
                ps.append(pred.cpu())
        ys = torch.cat(ys).numpy()
        ps = torch.cat(ps).numpy()
        m = compute_metrics(ys, ps, num_classes=27)

        avg_loss = total_loss / max(1, seen)
        print(f"[Random][Epoch {epoch:03d}] loss={avg_loss:.4f}  acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")

        if m["accuracy"] > best["accuracy"]:
            best.update({"epoch": epoch, "accuracy": float(m["accuracy"]), "macro_f1": float(m["macro_f1"])})
            torch.save(
                {"model": model.state_dict(), "cfg": cfg},
                out_dir / "ckpts" / "best_random.pt"
            )

    save_json({"best_random": best}, out_dir / "results_random.json")
    print("Random-split training complete. Best:", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args())