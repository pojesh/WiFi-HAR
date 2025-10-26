import json, os
from pathlib import Path
import torch

def save_json(d, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f: f.write(json.dumps(d, indent=2))

def amp_autocast(enabled=True):
    from contextlib import nullcontext
    return torch.cuda.amp.autocast() if enabled else nullcontext()