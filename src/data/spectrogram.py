import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from pathlib import Path

def to_amp_phase(csi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if np.iscomplexobj(csi):
        amp = np.abs(csi).astype(np.float32)
        phase = np.angle(csi).astype(np.float32)
    else:
        amp = csi.astype(np.float32)
        phase = None
    return amp, phase

def stft_spectrogram(x: torch.Tensor,
                     n_fft=128,
                     win_length=96,
                     hop_length=32,
                     window='hann',
                     power=2.0,
                     center=True,
                     **_ignored):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    win = torch.hann_window(win_length, device=x.device) if window == 'hann' else torch.ones(win_length, device=x.device)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                      window=win, return_complex=True, center=bool(center))
    return spec.abs() ** power if power is not None else spec

def make_csi_spectrogram(amp: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """
    amp: [T, L, S] amplitude-only CSI.
    PCA along subcarriers per link -> top-k PCs -> batched STFT -> stack as channels.
    Returns [C, F, T'] float32.
    """
    T, L, S = amp.shape
    device = torch.device('cpu')
    k_pcs = int(min(cfg.get("k_pcs", 4), S))

    amp_t = torch.from_numpy(amp).to(device=device, dtype=torch.float32)  # [T, L, S]
    amp_t = amp_t - amp_t.mean(dim=0, keepdim=True)                       # center per (L,S)

    stft_args = {k: cfg[k] for k in ("n_fft", "win_length", "hop_length", "window", "power") if k in cfg}
    eps = float(cfg.get("log_eps", 1e-6))

    link_specs = []
    for li in range(L):
        A = amp_t[:, li, :]                     # [T, S]
        # PCA via SVD on [T,S]
        U, Sigma, Vh = torch.linalg.svd(A, full_matrices=False)  # Vh: [min(T,S), S]
        V_top = Vh[:k_pcs, :]                   # [k_pcs, S]
        Ak = A @ V_top.T                        # [T, k_pcs]
        Ak = Ak.T.contiguous()                  # [k_pcs, T] for batched STFT

        # Batched STFT: feed [B, T] by flattening leading dims
        specs = []
        for k in range(k_pcs):
            spec_k = stft_spectrogram(Ak[k], **stft_args)  # [F, T']
            specs.append(spec_k)
        link_spec = torch.stack(specs, dim=0)  # [k_pcs, F, T']
        link_specs.append(link_spec)

    spec_all = torch.cat(link_specs, dim=0)     # [L*k_pcs, F, T']
    spec_all = torch.log(spec_all + eps).float()
    return spec_all.numpy()

def norm_spec(spec: np.ndarray, mode='instance'):
    if mode == 'instance':
        m = spec.mean(axis=(1,2), keepdims=True); s = spec.std(axis=(1,2), keepdims=True) + 1e-6
        return (spec - m)/s
    return spec

class SpectrogramCache:
    """
    Minimal, fast cache: npy or npz per sample (Windows-friendly).
    """
    def __init__(self, root: str, overwrite=False, format='npy', chunks=None):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.overwrite = overwrite
        self.format = format.lower()

    def _path(self, key: str):
        ext = ".npy" if self.format == "npy" else ".npz"
        return self.root / f"{key}{ext}"

    def save(self, key: str, arr: np.ndarray):
        path = self._path(key)
        if path.exists() and not self.overwrite:
            return
        if self.format == "npy":
            np.save(path, arr.astype(np.float32))
        else:
            np.savez_compressed(path, spec=arr.astype(np.float32))

    def load(self, key: str):
        path = self._path(key)
        if self.format == "npy":
            return np.load(path, mmap_mode='r')
        else:
            with np.load(path, allow_pickle=False) as f:
                return f["spec"]