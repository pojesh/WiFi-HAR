import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

def compute_metrics(y_true, y_pred, num_classes):
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average='macro')
    return {"accuracy": acc, "macro_f1": macro}

def make_confusion(y_true, y_pred, num_classes, normalize=True):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)), normalize=('true' if normalize else None))
    return cm