import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, channels, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(channels * patch_size, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        # x: (B, channels, time)
        B, C, T = x.shape
        num_patches = T // self.patch_size
        patches = []
        for i in range(num_patches):
            # Select patch along the time axis
            patch = x[:, :, i*self.patch_size:(i+1)*self.patch_size]  # (B, C, patch_size)
            patch = patch.reshape(B, C * self.patch_size)
            patches.append(patch)
        # Now patches is list of (B, C*patch_size), shape: num_patches x (B, patch_feat)
        patches = torch.stack(patches, dim=1)  # (B, num_patches, patch_dim)
        patches = patches.view(-1, patches.shape[-1])  # flatten so shape: [B*num_patches, patch_dim]
        patches = self.proj(patches)  # (B*num_patches, embed_dim)
        patches = patches.view(B, num_patches, -1)  # reshape back: (B, num_patches, embed_dim)
        return patches


class TemporalUncertaintyTransformer(nn.Module):
    def __init__(self, channels=90, patch_size=10, embed_dim=128, num_layers=6, num_heads=8, num_classes=7):
        super().__init__()
        self.patch_embed = PatchEmbed(channels, patch_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.cls_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, corruption_mask=None):
        patches = self.patch_embed(x)  # (B, num_patches, embed_dim)
        # Mask corrupted patches (for temporal dropout)
        if corruption_mask is not None:
            patches = torch.where(corruption_mask.unsqueeze(-1), patches, self.patch_embed.mask_token)
        feats = self.transformer(patches)
        # Use the mean feature or [CLS] equivalent
        pooled = feats.mean(dim=1)
        pred = self.cls_head(pooled)
        return pred
