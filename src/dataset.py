"""
Dataset para segmentacao de gordura epicardica.

Entrada:  Fat Images (BMP 8-bit, janelamento [-200, -30] HU)
Mascara:  Ground Truth - Fat Range (RGB) canal R = epicardial
Abordagem 2.5D: stack de slices adjacentes como canais extras.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ─── Splits ────────────────────────────────────────────────────────────────

def get_patient_splits(
    root_dir: str,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Divide pacientes em train / val / test com seed fixo."""
    fat_dir = Path(root_dir) / "data" / "fat_images"
    patients = sorted([d for d in os.listdir(fat_dir) if (fat_dir / d).is_dir()])

    rng = random.Random(seed)
    shuffled = patients.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = shuffled[:n_train]
    val   = shuffled[n_train : n_train + n_val]
    test  = shuffled[n_train + n_val :]

    return train, val, test


def get_patient_folds(
    root_dir: str,
    k: int = 10,
    seed: int = 42,
) -> List[Tuple[List[str], List[str], List[str]]]:
    """K-fold ao nivel de paciente.

    Cada fold retorna (train, val, test). O grupo i e o test da fold i,
    o grupo (i+1) mod k e a val, e os demais k-2 grupos compoem o train.
    Isso equivale ao protocolo 10-fold CV do paper Rodrigues et al. 2016.

    Args:
        root_dir: raiz do projeto (contem data/fat_images/).
        k:        numero de folds.
        seed:     semente para shuffle.
    """
    fat_dir = Path(root_dir) / "data" / "fat_images"
    patients = sorted([d for d in os.listdir(fat_dir) if (fat_dir / d).is_dir()])

    rng = random.Random(seed)
    shuffled = patients.copy()
    rng.shuffle(shuffled)

    if k < 2 or k > len(shuffled):
        raise ValueError(f"k={k} invalido para {len(shuffled)} pacientes")

    # np.array_split — divide em k grupos aproximadamente iguais
    groups = [list(g) for g in np.array_split(shuffled, k)]

    folds = []
    for i in range(k):
        test = groups[i]
        val  = groups[(i + 1) % k]
        train = [p for j, g in enumerate(groups) if j != i and j != (i + 1) % k for p in g]
        folds.append((train, val, test))
    return folds


# ─── Dataset ───────────────────────────────────────────────────────────────

class EpicardialDataset(Dataset):
    """
    Dataset 2.5D para segmentacao de gordura epicardica.

    Cada amostra e um stack de (2*context_slices + 1) imagens de gordura
    centrado no slice alvo, e a mascara binaria do slice central.

    Criterio de mascara (canal R dominante, per README):
        R > 10  AND  R > G  AND  R > B
    """

    EPICARDIAL_THRESHOLD = 10   # limiar absoluto no canal R
    EPICARDIAL_MARGIN    = 0    # margem sobre G e B (0 = sem margem)

    def __init__(
        self,
        root_dir: str,
        patient_ids: List[str],
        augment: bool = False,
        context_slices: int = 1,
    ):
        self.root_dir      = Path(root_dir)
        self.fat_dir       = self.root_dir / "data" / "fat_images"
        self.mask_dir      = self.root_dir / "data" / "ground_truth"
        self.augment       = augment
        self.context_slices = context_slices

        self.samples: List[dict] = []
        skipped = 0

        for pid in patient_ids:
            fat_path  = self.fat_dir  / pid
            mask_path = self.mask_dir / pid

            if not fat_path.exists() or not mask_path.exists():
                skipped += 1
                continue

            fat_files  = sorted(f for f in os.listdir(fat_path)  if f.upper().endswith(".BMP"))
            mask_files = sorted(f for f in os.listdir(mask_path) if f.lower().endswith(".bmp"))

            n_pairs = min(len(fat_files), len(mask_files))
            if n_pairs == 0:
                skipped += 1
                continue

            for idx in range(n_pairs):
                self.samples.append({
                    "patient_id": pid,
                    "fat_file":   fat_files[idx],
                    "mask_file":  mask_files[idx],
                    "slice_idx":  idx,
                    "fat_files":  fat_files,
                    "num_slices": len(fat_files),
                })

        if skipped:
            print(f"  [aviso] {skipped} pacientes ignorados (pasta ausente).")
        print(f"  Dataset: {len(self.samples)} slices de {len(patient_ids) - skipped} pacientes")

    # ── helpers ──────────────────────────────────────────────────────────

    def _load_fat(self, pid: str, fname: str) -> np.ndarray:
        img = Image.open(self.fat_dir / pid / fname).convert("L")
        return np.array(img, dtype=np.float32) / 255.0   # [0, 1]

    def _load_mask(self, pid: str, fname: str) -> np.ndarray:
        """Mascara binaria de gordura epicardica (R dominante)."""
        rgb = np.array(Image.open(self.mask_dir / pid / fname))
        R = rgb[:, :, 0].astype(np.float32)
        G = rgb[:, :, 1].astype(np.float32)
        B = rgb[:, :, 2].astype(np.float32)
        m = self.EPICARDIAL_MARGIN
        epi = (R > self.EPICARDIAL_THRESHOLD) & (R > G + m) & (R > B + m)
        return epi.astype(np.float32)

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        """Augmentacao geometrica simples (flip + rotate90)."""
        if random.random() < 0.5:
            image = np.flip(image, axis=-1).copy()
            mask  = np.flip(mask,  axis=-1).copy()
        if random.random() < 0.5:
            image = np.flip(image, axis=-2).copy()
            mask  = np.flip(mask,  axis=-2).copy()
        k = random.randint(0, 3)
        if k:
            image = np.rot90(image, k, axes=(-2, -1)).copy()
            mask  = np.rot90(mask,  k, axes=(-2, -1)).copy()
        return image, mask

    # ── interface ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s   = self.samples[idx]
        pid = s["patient_id"]
        si  = s["slice_idx"]
        ffs = s["fat_files"]
        n   = s["num_slices"]
        c   = self.context_slices

        # stack de slices adjacentes  (C, H, W)
        slices = []
        for offset in range(-c, c + 1):
            clamped = max(0, min(n - 1, si + offset))
            slices.append(self._load_fat(pid, ffs[clamped]))
        image = np.stack(slices, axis=0)          # (C, H, W)

        mask  = self._load_mask(pid, s["mask_file"])[np.newaxis]  # (1, H, W)

        if self.augment:
            image, mask = self._augment(image, mask)

        return torch.from_numpy(image), torch.from_numpy(mask)
