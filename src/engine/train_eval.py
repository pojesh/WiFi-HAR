import torch, time
from pathlib import Path
from typing import Dict, Any
from src.utils.metrics import compute_metrics, make_confusion
from src.utils.viz import save_confusion
from src.utils.common import amp_autocast

def train_one_epoch(model, loader, optimizer, scaler, device, amp, adv_cfg):
    model.train(); total=0; n=0
    for batch in loader:
        x, y, subj = batch["x"].to(device), batch["y"].to(device), batch["subj"].to(device)
        grl_lam = None
        if adv_cfg["enable"]:
            # simple linear schedule over steps: lam * (i/total)
            grl_lam = adv_cfg["lambda"]
        optimizer.zero_grad(set_to_none=True)
        with amp_autocast(amp):
            logits, adv_logits = model(x, grl_lam)
            from src.losses.adversarial import dann_loss
            loss, stats = dann_loss(logits, y, adv_logits, subj, lam=(grl_lam or 0.0))
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        total += loss.item()*x.size(0); n += x.size(0)
    return total/max(n,1)

@torch.no_grad()
def evaluate(model, loader, device, amp, num_classes, save_dir=None, classes=None):
    model.eval(); ys=[]; ps=[]
    for batch in loader:
        x, y = batch["x"].to(device), batch["y"].to(device)
        with amp_autocast(amp):
            logits, _ = model(x, None)
        pred = logits.argmax(dim=1)
        ys.append(y.cpu()); ps.append(pred.cpu())
    ys = torch.cat(ys).numpy(); ps = torch.cat(ps).numpy()
    metrics = compute_metrics(ys, ps, num_classes)
    if save_dir is not None:
        cm = make_confusion(ys, ps, num_classes, normalize=True)
        if classes is None: classes = [f"A{i:02d}"]*num_classes
        from src.utils.viz import save_confusion
        save_confusion(cm, classes, Path(save_dir)/"confusion.png")
    return metrics