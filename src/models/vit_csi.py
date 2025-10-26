import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat

class PatchEmbed(nn.Module):
    def __init__(self, in_chans, patch_f, patch_t, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(patch_f, patch_t), stride=(patch_f, patch_t), bias=True)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):  # x: [B, C, F, T]
        x = self.proj(x)   # [B, E, F', T']
        x = Rearrange('b e f t -> b (f t) e')(x)  # tokens
        x = self.norm(x)
        return x

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        hidden = int(dim*mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x):
        x = self.fc2(self.drop(self.act(self.fc1(x))))
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.1, attn_drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop  = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_ratio, drop)
    def forward(self, x):
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = self.drop(x) + h
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop(x) + h
        return x

class ViTCSI(nn.Module):
    def __init__(self, in_chans, num_classes, embed_dim=192, depth=8, num_heads=3, mlp_ratio=4.0,
                 patch_f=8, patch_t=8, cls_token=True, rel_pos_2d=False, adv_subject_classes=0):
        super().__init__()
        self.patch = PatchEmbed(in_chans, patch_f, patch_t, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim)) if cls_token else None
        self.pos = None  # use learned absolute 1D pos; created at runtime after seeing token count
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        # Optional adversarial subject classifier (DANN)
        self.adv_head = nn.Linear(embed_dim, adv_subject_classes) if adv_subject_classes>0 else None
        nn.init.trunc_normal_(self.head.weight, std=0.02)

    def _init_pos(self, n_tokens, dim, device):
        if (self.pos is None) or (self.pos.shape[1] != n_tokens):
            self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim, device=device))
            nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x, grl_lambda=None):
        # x: [B, C, F, T]
        B = x.shape[0]
        x = self.patch(x)  # [B, N, E]
        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
        self._init_pos(x.shape[1], x.shape[2], x.device)
        x = x + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.cls_token is not None:
            feat = x[:,0]      # [B, E]
        else:
            feat = x.mean(dim=1)
        logits = self.head(feat)
        adv_logits = None
        if (self.adv_head is not None) and (grl_lambda is not None):
            from .grl import grad_reverse
            adv_logits = self.adv_head(grad_reverse(feat, grl_lambda))
        return logits, adv_logits