"""Utilities intentionally provided to students.

The project focuses on the PyTorch model/training pipeline. Dataset engineering,
metric boilerplate, and reproducibility helpers are provided so students can
concentrate on Dataset/DataLoader usage, nn.Module, forward passes, losses,
backpropagation, optimization, validation, and model selection.
"""
from __future__ import annotations

import random
import numpy as np
import torch
import torchmetrics
import yaml


def TODO(message):
    raise NotImplementedError('TODO: ' + message)

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __repr__(self):
        items = ", ".join(f"{k}={v!r}" for k, v in sorted(self.__dict__.items()))
        return f"Struct({items})"


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return Struct(**(yaml.safe_load(f) or {}))


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def infer_feature_dim(dataset) -> int:
    if len(dataset) == 0:
        raise ValueError("Cannot infer feature dimension from an empty dataset.")
    if hasattr(dataset, "feature_dim"):
        return int(dataset.feature_dim)
    return int(dataset[0]["input"].shape[-1])


def classification_metrics(probabilities: torch.Tensor,
                           labels: torch.Tensor,
                           n_class: int):
    """Return macro AUROC, accuracy (%), and macro F1.

    Metric implementation is provided because writing AUROC/F1 from scratch is
    not a learning objective of this project.
    """
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [N, C]")
    labels = labels.long()
    device = probabilities.device
    auroc_metric = torchmetrics.AUROC(
        num_classes=int(n_class), task="multiclass", average="macro"
    ).to(device)
    f1_metric = torchmetrics.F1Score(
        num_classes=int(n_class), task="multiclass", average="macro"
    ).to(device)
    auc = float(auroc_metric(probabilities, labels).item())
    f1 = float(f1_metric(probabilities, labels).item())
    acc = float((probabilities.argmax(dim=1) == labels).float().mean().item() * 100.0)
    return {"auc": auc, "acc": acc, "f1": f1}
