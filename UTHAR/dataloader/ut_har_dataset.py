import numpy as np
import torch
from torch.utils.data import Dataset

class UTHARDataset(Dataset):
    def __init__(self, x_path, y_path, corruption_level=0.2, jitter=False, fixed_length=250):
        self.x = np.load(x_path)
        self.y = np.load(y_path)
        self.corruption_level = corruption_level
        self.jitter = jitter
        self.fixed_length = fixed_length
        self.feat_dim = self.x.shape[1]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        sample = self.x[idx].copy()  # e.g., (90*250,) -> (90, 250)
        label = self.y[idx]
        if self.feat_dim == 22500:
            sample = sample.reshape(90, 250)

        # Temporal corruption
        mask = np.ones(sample.shape[1], dtype=bool)
        if self.corruption_level > 0:
            mask = np.random.rand(sample.shape[1]) > self.corruption_level
            if mask.sum() == 0:
                mask[np.random.randint(sample.shape[1])] = True
            sample = sample[:, mask]

        # Jitter
        if self.jitter:
            perm = np.random.permutation(sample.shape[1])
            sample = sample[:, perm]
        
        # Pad or truncate to fixed_length (on time dimension)
        curr_len = sample.shape[1]
        if curr_len < self.fixed_length:
            pad_width = self.fixed_length - curr_len
            # Pad with zeros at the end
            sample = np.pad(sample, ((0, 0), (0, pad_width)), 'constant', constant_values=0)
        elif curr_len > self.fixed_length:
            sample = sample[:, :self.fixed_length]

        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)