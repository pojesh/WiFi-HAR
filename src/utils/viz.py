import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def save_confusion(cm, classes, out_path):
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=True, xticklabels=classes, yticklabels=classes)
    plt.xlabel("Pred"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()