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
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    elif path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as f:
            return f["spec"]
    else:
        raise ValueError(f"Unknown cache file type: {path}")


def batchify(paths, idxs, labels, subj_ids, batch_size):
    for start in range(0, len(idxs), batch_size):
        sel = idxs[start:start + batch_size]
        xs = [load_cached(paths[i]) for i in sel]
        x = torch.from_numpy(np.stack(xs, axis=0)).float()
        y = torch.tensor([labels[i] for i in sel], dtype=torch.long)
        s = torch.tensor([subj_ids[i] for i in sel], dtype=torch.long)
        yield {"x": x, "y": y, "subj": s}


def main(args):
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 42))
    torch.backends.cudnn.benchmark = True

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # 1) Dataset and keys
    ds, _ = build_mmfi_dataset(cfg["dataset_root"], cfg)
    if len(ds) == 0:
        raise RuntimeError("MM-Fi dataset is empty. Check your config subjects/actions.")
    first = ds[0]
    data_key, label_key, subj_key = infer_batch_keys(first)

    # 2) Build cache path list and metadata
    cache_root = Path(cfg["cache"]["root"])
    ext = ".npy" if cfg["cache"]["format"].lower() == "npy" else ".npz"

    actions_map = {f"A{i:02d}": i - 1 for i in range(1, 28)}   # 27 classes
    subj_map = {f"S{i:02d}": i - 1 for i in range(1, 11)}      # 10 subjects for E01

    paths, labels, subj_ids = [], [], []
    subj_strs = []  # exact subject strings for LOSO split
    for i in range(len(ds)):
        key = f"E01_{i:06d}"
        p = cache_root / f"{key}{ext}"
        if not p.exists():
            raise FileNotFoundError(f"Cached spectrogram missing: {p}. Run prep_cache.py first.")
        paths.append(str(p))

        item = ds[i]
        act = str(item[label_key])
        if act not in actions_map:
            act = f"A{int(''.join([c for c in act if c.isdigit()])):02d}"
        labels.append(actions_map[act])

        sid = str(item.get(subj_key, item.get("subject", "")))
        subj_strs.append(sid)
        sid_key = next((k for k in subj_map.keys() if k in sid), None)
        if sid_key is None:
            raise KeyError(f"Cannot parse subject from field '{subj_key}': {item.get(subj_key)}")
        subj_ids.append(subj_map[sid_key])

    # 3) Determine input shape
    sample_arr = load_cached(paths[0])
    in_chans, F, Tspec = sample_arr.shape

    # 4) LOSO folds
    subjects = [f"S{i:02d}" for i in range(1, 11)]  # E01 subjects
    batch_size = int(cfg["optim"]["batch_size"])
    grad_clip = float(cfg["optim"]["grad_clip_norm"])
    epochs = int(cfg["optim"]["epochs"])

    out_dir = Path(cfg["output_root"]) / cfg["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []

    for held in subjects:
        # Build fold indices
        test_idx = [i for i in range(len(paths)) if held in subj_strs[i]]
        train_idx = [i for i in range(len(paths)) if i not in test_idx]
        if len(test_idx) == 0:
            print(f"[LOSO] Warning: no samples found for held-out subject {held}; skipping.")
            continue

        # Rebuild model for each fold (fresh weights)
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

        optim = AdamW(model.parameters(),
                      lr=cfg["optim"]["lr"],
                      weight_decay=cfg["optim"]["weight_decay"],
                      betas=tuple(cfg["optim"]["betas"]))
        scaler = GradScaler(enabled=cfg["optim"]["amp"])

        best = {"accuracy": 0.0, "macro_f1": 0.0}
        for epoch in range(epochs):
            # Train
            model.train()
            total_loss, seen = 0.0, 0
            for batch in batchify(paths, train_idx, labels, subj_ids, batch_size):
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                s = batch["subj"].to(device, non_blocking=True)
                optim.zero_grad(set_to_none=True)
                with autocast(enabled=cfg["optim"]["amp"]):
                    logits, adv_logits = model(x, (cfg["adv"]["lambda"] if cfg["adv"]["enable"] else None))
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

            # Eval on held-out subject
            model.eval()
            ys, ps = [], []
            with torch.inference_mode(), autocast(enabled=cfg["optim"]["amp"]):
                for batch in batchify(paths, test_idx, labels, subj_ids, batch_size):
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
            print(f"[LOSO {held}][Epoch {epoch:03d}] loss={avg_loss:.4f}  acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")

            if m["accuracy"] > best["accuracy"]:
                best = {"accuracy": float(m["accuracy"]), "macro_f1": float(m["macro_f1"])}

        fold_results.append({"subject": held, **best})
        print(f"[LOSO {held}] best_acc={best['accuracy']:.4f}  best_f1={best['macro_f1']:.4f}")

    # Aggregate and save
    if len(fold_results) > 0:
        accs = [r["accuracy"] for r in fold_results]
        f1s = [r["macro_f1"] for r in fold_results]
        summary = {
            "per_subject": fold_results,
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)),
            "worst_acc": float(np.min(accs)),
        }
    else:
        summary = {"per_subject": [], "mean_acc": 0.0, "std_acc": 0.0, "mean_f1": 0.0, "std_f1": 0.0, "worst_acc": 0.0}

    save_json(summary, out_dir / "results_loso.json")
    print("LOSO complete. Summary:", summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args())