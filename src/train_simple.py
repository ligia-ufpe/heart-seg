"""
Treinamento simplificado para segmentacao de gordura epicardica.

Usa:
- Fat Images (BMP, ja com janelamento de gordura) como entrada
- Ground Truth - Fat Range (RGB) para mascaras
- SegResNet (MONAI) - modelo mais leve e eficiente
- Tversky Loss - melhor para desbalanceamento de classes
"""

import argparse
import os
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# MONAI
from monai.networks.nets import SegResNet, UNet
from monai.losses import DiceLoss, TverskyLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, RandFlipd, RandRotate90d, 
    RandGaussianNoised, RandAdjustContrastd
)

import warnings
warnings.filterwarnings('ignore')


def set_seed(seed: int = 42):
    """Configura seeds para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class FatImageDataset(Dataset):
    
    def __init__(
        self,
        root_dir: str,
        patient_ids: List[str],
        augment: bool = False,
        context_slices: int = 1
    ):
        self.root_dir = Path(root_dir)
        self.fat_dir = self.root_dir / 'Fat Images'
        self.mask_dir = self.root_dir / 'Ground Truth - Fat Range'
        self.augment = augment
        self.context_slices = context_slices
        
        # Criar lista de samples
        self.samples = []
        
        for patient_id in patient_ids:
            fat_path = self.fat_dir / patient_id
            mask_path = self.mask_dir / patient_id
            
            if not fat_path.exists() or not mask_path.exists():
                continue
            
            # Listar arquivos ordenados por indice
            fat_files = sorted([f for f in os.listdir(fat_path) if f.upper().endswith('.BMP')])
            mask_files = sorted([f for f in os.listdir(mask_path) if f.lower().endswith('.bmp')])
            
            # Usar indice sequencial em vez de numero do arquivo
            # Fat files e mask files devem corresponder 1:1 pela ordem
            num_pairs = min(len(fat_files), len(mask_files))
            
            if num_pairs == 0:
                continue
            
            for idx in range(num_pairs):
                self.samples.append({
                    'patient_id': patient_id,
                    'fat_file': fat_files[idx],
                    'mask_file': mask_files[idx],
                    'slice_idx': idx,
                    'fat_files': fat_files,
                    'num_slices': len(fat_files)
                })
        
        print(f"  Dataset: {len(self.samples)} amostras de {len(patient_ids)} pacientes")
        
        # Transforms para augmentation
        if self.augment:
            self.transforms = Compose([
                RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0),
                RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1),
                RandRotate90d(keys=['image', 'mask'], prob=0.5, max_k=3),
            ])
        else:
            self.transforms = None
    
    def _load_fat_image(self, patient_id: str, filename: str) -> np.ndarray:
        """Carrega Fat Image e normaliza para [0, 1]."""
        path = self.fat_dir / patient_id / filename
        img = np.array(Image.open(path), dtype=np.float32) / 255.0
        return img
    
    def _load_mask(self, patient_id: str, filename: str) -> np.ndarray:
        """Carrega mascara e extrai gordura epicardica (R dominante)."""
        path = self.mask_dir / patient_id / filename
        mask_rgb = np.array(Image.open(path))
        
        R = mask_rgb[:, :, 0].astype(np.float32)
        G = mask_rgb[:, :, 1].astype(np.float32)
        B = mask_rgb[:, :, 2].astype(np.float32)
        
        # R dominante com margem - criterio mais relaxado
        # Aceita pixels onde R eh significativamente maior que G e B
        epicardial = ((R > 100) & (R > G + 20) & (R > B + 20)).astype(np.float32)
        
        return epicardial
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        patient_id = sample['patient_id']
        slice_idx = sample['slice_idx']
        fat_files = sample['fat_files']
        num_slices = sample['num_slices']
        
        # Carregar slice central
        center_img = self._load_fat_image(patient_id, sample['fat_file'])
        
        # Criar stack 2.5D com slices adjacentes
        slices = [center_img]
        
        for offset in range(1, self.context_slices + 1):
            # Slice anterior
            prev_idx = max(0, slice_idx - offset)
            prev_img = self._load_fat_image(patient_id, fat_files[prev_idx])
            slices.insert(0, prev_img)
            
            # Slice posterior
            next_idx = min(num_slices - 1, slice_idx + offset)
            next_img = self._load_fat_image(patient_id, fat_files[next_idx])
            slices.append(next_img)
        
        # Stack: (C, H, W) onde C = 2*context_slices + 1
        image = np.stack(slices, axis=0)
        
        # Carregar mascara
        mask = self._load_mask(patient_id, sample['mask_file'])
        mask = mask[np.newaxis, ...]  # (1, H, W)
        
        # Converter para tensores
        data = {
            'image': torch.from_numpy(image),
            'mask': torch.from_numpy(mask)
        }
        
        # Aplicar augmentation
        if self.transforms:
            data = self.transforms(data)
        
        return data['image'], data['mask']


class TverskyFocalLoss(nn.Module):
    """
    Combina Tversky Loss com Focal Loss para classes desbalanceadas.
    
    Tversky: Permite ajustar peso entre FP e FN via alpha/beta
    Focal: Foca em pixels dificeis
    """
    
    def __init__(self, alpha=0.3, beta=0.7, gamma=2.0):
        super().__init__()
        # alpha=0.3, beta=0.7 -> penaliza mais FN (falsos negativos)
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, sigmoid=True)
        self.gamma = gamma
    
    def forward(self, pred, target):
        # Tversky loss
        tversky = self.tversky(pred, target)
        
        # Focal component
        pred_sig = torch.sigmoid(pred)
        pt = target * pred_sig + (1 - target) * (1 - pred_sig)
        focal_weight = (1 - pt) ** self.gamma
        bce = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
        focal = (focal_weight * bce).mean()
        
        # Combinar
        return 0.7 * tversky + 0.3 * focal


def get_patient_splits(root_dir: str, seed: int = 42):
    """Divide pacientes em train/val/test."""
    fat_dir = Path(root_dir) / 'Fat Images'
    patients = sorted([d for d in os.listdir(fat_dir) if os.path.isdir(fat_dir / d)])
    
    random.seed(seed)
    random.shuffle(patients)
    
    n = len(patients)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    train = patients[:n_train]
    val = patients[n_train:n_train + n_val]
    test = patients[n_train + n_val:]
    
    return train, val, test


def train_epoch(model, loader, optimizer, criterion, scaler, device):
    """Treina uma epoca."""
    model.train()
    total_loss = 0
    
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 20 == 0:
            print(f"    Batch {batch_idx + 1}/{len(loader)} - Loss: {loss.item():.4f}")
    
    return total_loss / len(loader)


def validate(model, loader, criterion, metric, device):
    """Valida o modelo."""
    model.eval()
    total_loss = 0
    metric.reset()
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            # Calcular Dice
            preds = (torch.sigmoid(outputs) > 0.5).float()
            metric(preds, masks)
    
    dice = metric.aggregate().item()
    metric.reset()
    
    return total_loss / len(loader), dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='ct_data')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--context_slices', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None, help='Caminho para checkpoint para continuar treinamento')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Paths
    data_dir = Path(args.data_dir).resolve()
    
    # Se resume, usar mesmo diretório do checkpoint
    if args.resume:
        save_dir = Path(args.resume).parent
        print(f"Continuando treinamento de: {args.resume}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(f"checkpoints/epicardial_simple_{timestamp}")
        save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("TREINAMENTO SIMPLIFICADO - GORDURA EPICARDICA")
    print("=" * 70)
    print(f"  Data dir: {data_dir}")
    print(f"  Save dir: {save_dir}")
    print(f"  Context slices: {args.context_slices} (input channels: {2*args.context_slices + 1})")
    
    # Splits
    train_ids, val_ids, test_ids = get_patient_splits(data_dir, args.seed)
    print(f"\n  Pacientes: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    
    # Datasets
    in_channels = 2 * args.context_slices + 1
    train_ds = FatImageDataset(data_dir, train_ids, augment=True, context_slices=args.context_slices)
    val_ds = FatImageDataset(data_dir, val_ids, augment=False, context_slices=args.context_slices)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)
    
    # Modelo - SegResNet (mais leve e eficiente)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SegResNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=1,
        init_filters=32,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        dropout_prob=0.2
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Modelo: SegResNet")
    print(f"  Parametros: {num_params:,}")
    
    # Loss e otimizador
    criterion = TverskyFocalLoss(alpha=0.3, beta=0.7, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.amp.GradScaler('cuda')
    metric = DiceMetric(include_background=True, reduction='mean')
    
    # Carregar checkpoint se resume
    start_epoch = 0
    best_dice = 0
    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'lr': []}
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_dice = checkpoint.get('val_dice', 0)
        print(f"\n  Carregado checkpoint da época {start_epoch - 1}")
        print(f"  Melhor Dice anterior: {best_dice:.4f}")
        
        # Carregar histórico anterior se existir
        history_path = save_dir / 'training_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
    
    print(f"  Loss: TverskyFocalLoss (alpha=0.3, beta=0.7)")
    print(f"  Device: {device}")
    print("=" * 70)
    
    # Loop de treinamento
    patience_counter = 0
    
    for epoch in range(start_epoch, args.max_epochs):
        print(f"\nEpoch {epoch + 1}/{args.max_epochs}")
        print("-" * 40)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        
        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, metric, device)
        
        # Atualiza scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Dice: {val_dice:.4f}")
        print(f"  LR: {current_lr:.2e}")
        
        # Salva historico
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['lr'].append(current_lr)
        
        # Salva o melhor modelo
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_loss': val_loss
            }, save_dir / 'best_model.pth')
            print(f"  >> Novo melhor modelo! Dice: {val_dice:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping apos {args.patience} epocas sem melhora.")
                break
        
        # Salva checkpoint a cada 10 epocas
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_dice': val_dice
            }, save_dir / f'checkpoint_epoch_{epoch + 1}.pth')
    
    # Salva histórico
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TREINAMENTO CONCLUIDO")
    print(f"  Melhor Dice: {best_dice:.4f}")
    print(f"  Checkpoints salvos em: {save_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
