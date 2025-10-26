import torch
import torch.nn.functional as F

def dann_loss(main_logits, labels, adv_logits=None, subj_labels=None, lam=0.0):
    cls_loss = F.cross_entropy(main_logits, labels)
    if (adv_logits is None) or (subj_labels is None) or lam<=0:
        return cls_loss, {"cls": cls_loss.item()}
    adv_loss = F.cross_entropy(adv_logits, subj_labels)
    total = cls_loss + lam*adv_loss
    return total, {"cls": cls_loss.item(), "adv": adv_loss.item(), "lam": lam}