"""
Metricas de avaliacao para segmentacao binaria.

Para o paper reportamos: Dice, IoU (Jaccard), Sensitivity (Recall),
Specificity, Precision e HD95 (Hausdorff 95-th percentile).
"""

from typing import Dict

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def _to_numpy_binary(tensor: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """Converte logit/prob tensor para mascara binaria numpy."""
    if tensor.requires_grad:
        tensor = tensor.detach()
    arr = tensor.cpu().numpy()
    return (arr > threshold).astype(np.uint8)


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    inter = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)


def iou_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + eps) / (union + eps)


def sensitivity(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """True positive rate / recall."""
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return (tp + eps) / (tp + fn + eps)


def specificity(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """True negative rate."""
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    return (tn + eps) / (tn + fp + eps)


def precision(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return (tp + eps) / (tp + fp + eps)


def hausdorff95(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Hausdorff distance ao 95o percentil.
    Retorna np.nan se pred ou target estiver vazio.
    """
    if pred.sum() == 0 or target.sum() == 0:
        return float("nan")

    # distancia de cada pixel ate a borda mais proxima da outra classe
    dist_pred   = distance_transform_edt(1 - pred)
    dist_target = distance_transform_edt(1 - target)

    d_pt = dist_target[pred   == 1]
    d_tp = dist_pred  [target == 1]

    return float(np.percentile(np.concatenate([d_pt, d_tp]), 95))


def compute_all_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calcula todas as metricas a partir de logits e targets (batched).

    Args:
        logits:  (B, 1, H, W) tensor de saida do modelo (raw ou prob)
        targets: (B, 1, H, W) tensor de mascaras binarias
    Returns:
        dict com dice, iou, sensitivity, specificity, precision, hd95
    """
    probs  = torch.sigmoid(logits)
    pred_b = _to_numpy_binary(probs.squeeze(1), threshold)   # (B, H, W)
    tgt_b  = targets.squeeze(1).cpu().numpy().astype(np.uint8)

    metrics: Dict[str, list] = {k: [] for k in ["dice", "iou", "sensitivity", "specificity", "precision", "hd95"]}

    for p, t in zip(pred_b, tgt_b):
        metrics["dice"].append(dice_score(p, t))
        metrics["iou"].append(iou_score(p, t))
        metrics["sensitivity"].append(sensitivity(p, t))
        metrics["specificity"].append(specificity(p, t))
        metrics["precision"].append(precision(p, t))
        metrics["hd95"].append(hausdorff95(p, t))

    return {k: float(np.nanmean(v)) for k, v in metrics.items()}
