import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.optim import AdamW
#from torch.cuda.amp import GradScaler, autocast
from torch.amp import GradScaler, autocast

from src.data.mmfi_wrap import build_mmfi_dataset  # no need to call __getitem__
from src.models.vit_csi import ViTCSI
from src.utils.metrics import compute_metrics
from src.utils.common import save_json


def set_seed(seed: int):
    import random, os
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_cached(path: str):
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    elif path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as f:
            return f["spec"]
    else:
        raise ValueError(f"Unknown cache file type: {path}")


def make_random_split_indices(N: int, test_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(N); rng.shuffle(idx)
    n_test = max(1, int(N * test_ratio))
    return np.sort(idx[n_test:]).tolist(), np.sort(idx[:n_test]).tolist()


def batchify(paths, idxs, labels, subj_ids, batch_size):
    for start in range(0, len(idxs), batch_size):
        sel = idxs[start:start + batch_size]
        xs = [load_cached(paths[i]) for i in sel]
        x = torch.from_numpy(np.stack(xs, axis=0)).float()  # [B,C,F,T]
        y = torch.tensor([labels[i] for i in sel], dtype=torch.long)
        s = torch.tensor([subj_ids[i] for i in sel], dtype=torch.long)
        yield {"x": x, "y": y, "subj": s}


def main(args):
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[train_random] device={device}")

    # 1) Build dataset (metadata only; do NOT call ds[i])
    ds, _ = build_mmfi_dataset(cfg["dataset_root"], cfg)
    N = len(ds)
    if N == 0:
        raise RuntimeError("Dataset empty. Check YAML manual_split subjects/actions.")
    print(f"[train_random] sequences found: {N}")

    # 2) Build cache paths and labels/subjects from metadata only
    cache_root = Path(cfg["cache"]["root"])
    ext = ".npy" if cfg["cache"]["format"].lower() == "npy" else ".npz"

    actions_map = {f"A{i:02d}": i - 1 for i in range(1, 28)}
    subj_map = {f"S{i:02d}": i - 1 for i in range(1, 11)}  # E01

    paths, labels, subj_ids = [], [], []
    for i, meta in enumerate(ds.data_list):  # use metadata; avoid heavy __getitem__
        key = f"E01_{i:06d}"
        p = cache_root / f"{key}{ext}"
        if not p.exists():
            raise FileNotFoundError(f"Missing cache file: {p}. Run prep_cache.py first.")
        paths.append(str(p))

        act = str(meta["action"])            # e.g., 'A05'
        if act not in actions_map:
            # tolerate 'A5'
            digits = "".join([c for c in act if c.isdigit()])
            act = f"A{int(digits):02d}"
        labels.append(actions_map[act])

        sid = str(meta["subject"])           # e.g., 'S01'
        sid_key = next((k for k in subj_map.keys() if k in sid), None)
        if sid_key is None:
            raise KeyError(f"Cannot parse subject from '{sid}'")
        subj_ids.append(subj_map[sid_key])

    # 3) Random split
    tr_idx, te_idx = make_random_split_indices(len(paths), test_ratio=0.2, seed=cfg.get("seed", 42))
    print(f"[train_random] train={len(tr_idx)}  test={len(te_idx)}")

    # 4) Build model (infer input shape from one cached file)
    sample_arr = load_cached(paths[0])
    in_chans, F, Tspec = sample_arr.shape
    print(f"[train_random] input shape [C,F,T']: {sample_arr.shape}")

    model = ViTCSI(
        in_chans=in_chans, num_classes=27,
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
    scaler = GradScaler(device="cuda", enabled=cfg["optim"]["amp"])
    batch_size = int(cfg["optim"]["batch_size"])
    grad_clip = float(cfg["optim"]["grad_clip_norm"])
    epochs = int(cfg["optim"]["epochs"])

    out_dir = Path(cfg["output_root"]) / cfg["experiment_name"]
    (out_dir / "ckpts").mkdir(parents=True, exist_ok=True)

    best = {"epoch": -1, "accuracy": 0.0, "macro_f1": 0.0}

    # 5) Train/eval
    for epoch in range(epochs):
        model.train()
        total_loss, seen = 0.0, 0

        for batch in batchify(paths, tr_idx, labels, subj_ids, batch_size):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            s = batch["subj"].to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=cfg["optim"]["amp"]):
                logits, adv_logits = model(x, (cfg["adv"]["lambda"] if cfg["adv"]["enable"] else None))
                from src.losses.adversarial import dann_loss
                loss, _ = dann_loss(
                    logits, y,
                    adv_logits=adv_logits, subj_labels=s,
                    lam=(cfg["adv"]["lambda"] if cfg["adv"]["enable"] else 0.0)
                )
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim); scaler.update()
            total_loss += float(loss.item()) * x.size(0); seen += x.size(0)

        # Evaluate
        model.eval()
        ys, ps = [], []
        with torch.inference_mode(), autocast(device_type="cuda",enabled=cfg["optim"]["amp"]):
            for batch in batchify(paths, te_idx, labels, subj_ids, batch_size):
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                logits, _ = model(x, None)
                pred = logits.argmax(dim=1)
                ys.append(y.cpu()); ps.append(pred.cpu())
        ys = torch.cat(ys).numpy(); ps = torch.cat(ps).numpy()
        m = compute_metrics(ys, ps, num_classes=27)

        avg_loss = total_loss / max(1, seen)
        print(f"[Random][Epoch {epoch:03d}] loss={avg_loss:.4f}  acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")

        if m["accuracy"] > best["accuracy"]:
            best.update({"epoch": epoch, "accuracy": float(m["accuracy"]), "macro_f1": float(m["macro_f1"])})
            torch.save({"model": model.state_dict(), "cfg": cfg}, out_dir / "ckpts" / "best_random.pt")

    save_json({"best_random": best}, out_dir / "results_random.json")
    print("[train_random] Done. Best:", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args())