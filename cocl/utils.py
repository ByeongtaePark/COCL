import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.preprocessing import label_binarize


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device):
    """Returns (labels, probabilities) for a loader of raw images."""
    model.eval()
    labels, probs = [], []
    for images, targets in loader:
        _, logits = model(images.to(device))
        probs.append(logits.softmax(dim=1).cpu())
        labels.append(targets)
    return torch.cat(labels).numpy(), torch.cat(probs).numpy()


def compute_metrics(labels, probs):
    """ACC, weighted F1, and macro PR-AUC (in %)."""
    preds = probs.argmax(axis=1)
    num_classes = probs.shape[1]
    if num_classes == 2:
        pr_auc = average_precision_score(labels, probs[:, 1])
    else:
        y = label_binarize(labels, classes=list(range(num_classes)))
        pr_auc = average_precision_score(y, probs, average='macro')
    return {
        'ACC': accuracy_score(labels, preds) * 100,
        'F1': f1_score(labels, preds, average='weighted') * 100,
        'PR-AUC': pr_auc * 100,
    }


def format_metrics(metrics):
    return ' | '.join(f'{k}: {v:.2f}%' for k, v in metrics.items())
