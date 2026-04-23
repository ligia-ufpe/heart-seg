"""
Metricas de avaliacao para segmentacao binaria.

Dice, IoU, Sensitivity, Specificity e Precision sao agregados
globalmente (voxel-wise) em vez de media por slice, pois a media
por slice e inflada pelas slices vazias (pred=0 e target=0 → 1.0)
e torna as metricas incomparaveis com a literatura (ver Rodrigues
et al. 2016, que reporta Dice=97.7% voxel-wise em 10-fold CV).

HD95 permanece por slice, ignorando slices vazias (nao tem distancia
definida).
"""

from typing import Dict

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def hausdorff95(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Hausdorff distance ao 95o percentil.
    Retorna np.nan se pred ou target estiver vazio.
    """
    if pred.sum() == 0 or target.sum() == 0:
        return float("nan")

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
    Calcula metricas a partir de logits e targets (batched).

    Dice/IoU/Sens/Spec/Prec: agregacao global (TP/FP/FN/TN somados
    sobre todos os pixels de todas as slices, dividido uma vez).
    HD95: media por slice, ignorando slices vazias.

    Args:
        logits:  (N, 1, H, W) tensor de saida do modelo
        targets: (N, 1, H, W) tensor de mascaras binarias
    """
    probs  = torch.sigmoid(logits)
    pred_b = (probs > threshold).cpu().numpy().astype(np.uint8)   # (N, 1, H, W)
    tgt_b  = targets.cpu().numpy().astype(np.uint8)               # (N, 1, H, W)

    pf = pred_b.reshape(-1)
    tf = tgt_b.reshape(-1)

    tp = int(np.sum((pf == 1) & (tf == 1)))
    fp = int(np.sum((pf == 1) & (tf == 0)))
    fn = int(np.sum((pf == 0) & (tf == 1)))
    tn = int(np.sum((pf == 0) & (tf == 0)))

    eps = 1e-6
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou  = (tp + eps) / (tp + fp + fn + eps)
    sens = (tp + eps) / (tp + fn + eps)
    spec = (tn + eps) / (tn + fp + eps)
    prec = (tp + eps) / (tp + fp + eps)

    hd95_vals = []
    for p, t in zip(pred_b[:, 0], tgt_b[:, 0]):
        if p.sum() > 0 and t.sum() > 0:
            hd95_vals.append(hausdorff95(p, t))
    hd95 = float(np.mean(hd95_vals)) if hd95_vals else float("nan")

    return {
        "dice":        float(dice),
        "iou":         float(iou),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision":   float(prec),
        "hd95":        hd95,
    }
