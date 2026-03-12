"""
Script para avaliar o modelo treinado de segmentação de gordura epicárdica.
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# MONAI
from monai.networks.nets import SegResNet
from monai.metrics import DiceMetric, HausdorffDistanceMetric

# Importar do train_simple
sys.path.append(str(Path(__file__).parent))
from train_simple import FatImageDataset, get_patient_splits


def load_model(checkpoint_path: str, in_channels: int = 3, device: str = 'cuda'):
    """Carrega o modelo a partir do checkpoint."""
    model = SegResNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=1,
        init_filters=32,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        dropout_prob=0.2
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def evaluate_model(model, dataloader, device):
    """Avalia o modelo no conjunto de dados."""
    dice_metric = DiceMetric(include_background=True, reduction='mean_batch')
    
    all_dices = []
    all_predictions = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            # Calcular Dice para cada amostra
            dice_metric(preds, masks)
            
            # Salvar para análise
            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(masks.cpu().numpy())
    
    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    
    return mean_dice, all_predictions, all_targets


def calculate_metrics(predictions, targets):
    """Calcula métricas detalhadas."""
    metrics = {
        'dice': [],
        'precision': [],
        'recall': [],
        'iou': []
    }
    
    for pred, target in zip(predictions, targets):
        pred = pred.squeeze()
        target = target.squeeze()
        
        # Métricas pixel-wise
        tp = np.sum((pred == 1) & (target == 1))
        fp = np.sum((pred == 1) & (target == 0))
        fn = np.sum((pred == 0) & (target == 1))
        tn = np.sum((pred == 0) & (target == 0))
        
        # Dice
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        metrics['dice'].append(dice)
        
        # Precision
        precision = tp / (tp + fp + 1e-8)
        metrics['precision'].append(precision)
        
        # Recall (Sensitivity)
        recall = tp / (tp + fn + 1e-8)
        metrics['recall'].append(recall)
        
        # IoU (Jaccard)
        iou = tp / (tp + fp + fn + 1e-8)
        metrics['iou'].append(iou)
    
    return metrics


def visualize_results(model, dataset, device, num_samples=6, save_path=None):
    """Visualiza algumas predições do modelo."""
    model.eval()
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]
            image_input = image.unsqueeze(0).to(device)
            
            output = model(image_input)
            pred = (torch.sigmoid(output) > 0.5).float()
            
            # Pegar slice central da imagem (para visualização)
            center_slice = image.shape[0] // 2
            img_display = image[center_slice].cpu().numpy()
            mask_display = mask.squeeze().cpu().numpy()
            pred_display = pred.squeeze().cpu().numpy()
            
            # Plot
            axes[i, 0].imshow(img_display, cmap='gray')
            axes[i, 0].set_title('Imagem de Entrada')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask_display, cmap='Reds', vmin=0, vmax=1)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_display, cmap='Reds', vmin=0, vmax=1)
            axes[i, 2].set_title('Predição')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Visualização salva em: {save_path}")
    
    plt.show()


def main():
    # Encontrar checkpoint mais recente
    checkpoints_dir = Path(__file__).parent.parent / 'checkpoints'
    
    checkpoint_dirs = sorted([
        d for d in checkpoints_dir.iterdir() 
        if d.is_dir() and d.name.startswith('epicardial_simple')
    ])
    
    if not checkpoint_dirs:
        print("Nenhum checkpoint encontrado!")
        return
    
    latest_dir = checkpoint_dirs[-1]
    checkpoint_path = latest_dir / 'best_model.pth'
    
    if not checkpoint_path.exists():
        print(f"Arquivo best_model.pth não encontrado em {latest_dir}")
        return
    
    print("=" * 70)
    print("AVALIAÇÃO DO MODELO - SEGMENTAÇÃO DE GORDURA EPICÁRDICA")
    print("=" * 70)
    print(f"  Checkpoint: {checkpoint_path}")
    
    # Configuração
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = Path(__file__).parent.parent / 'ct_data'
    context_slices = 1
    in_channels = 2 * context_slices + 1
    
    print(f"  Device: {device}")
    print(f"  Data dir: {data_dir}")
    
    # Carregar modelo
    print("\nCarregando modelo...")
    model, checkpoint_info = load_model(str(checkpoint_path), in_channels, device)
    
    print(f"  Época do checkpoint: {checkpoint_info.get('epoch', 'N/A')}")
    print(f"  Val Dice no checkpoint: {checkpoint_info.get('val_dice', 'N/A'):.4f}")
    print(f"  Val Loss no checkpoint: {checkpoint_info.get('val_loss', 'N/A'):.4f}")
    
    # Carregar training history se existir
    history_path = latest_dir / 'training_history.json'
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        print(f"\n  Histórico de treinamento:")
        print(f"    Épocas treinadas: {len(history['train_loss'])}")
        print(f"    Melhor Val Dice: {max(history['val_dice']):.4f}")
        print(f"    Loss final (train): {history['train_loss'][-1]:.4f}")
        print(f"    Loss final (val): {history['val_loss'][-1]:.4f}")
    
    # Criar datasets
    print("\nCarregando dados...")
    train_ids, val_ids, test_ids = get_patient_splits(str(data_dir), seed=42)
    
    print(f"  Pacientes - Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    test_ds = FatImageDataset(str(data_dir), test_ids, augment=False, context_slices=context_slices)
    val_ds = FatImageDataset(str(data_dir), val_ids, augment=False, context_slices=context_slices)
    
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    
    # Avaliar no conjunto de validação
    print("\n" + "-" * 50)
    print("RESULTADOS NO CONJUNTO DE VALIDAÇÃO")
    print("-" * 50)
    
    val_dice, val_preds, val_targets = evaluate_model(model, val_loader, device)
    val_metrics = calculate_metrics(val_preds, val_targets)
    
    print(f"  Dice Score:  {np.mean(val_metrics['dice']):.4f} ± {np.std(val_metrics['dice']):.4f}")
    print(f"  Precision:   {np.mean(val_metrics['precision']):.4f} ± {np.std(val_metrics['precision']):.4f}")
    print(f"  Recall:      {np.mean(val_metrics['recall']):.4f} ± {np.std(val_metrics['recall']):.4f}")
    print(f"  IoU:         {np.mean(val_metrics['iou']):.4f} ± {np.std(val_metrics['iou']):.4f}")
    
    # Avaliar no conjunto de teste
    if len(test_ds) > 0:
        print("\n" + "-" * 50)
        print("RESULTADOS NO CONJUNTO DE TESTE")
        print("-" * 50)
        
        test_dice, test_preds, test_targets = evaluate_model(model, test_loader, device)
        test_metrics = calculate_metrics(test_preds, test_targets)
        
        print(f"  Dice Score:  {np.mean(test_metrics['dice']):.4f} ± {np.std(test_metrics['dice']):.4f}")
        print(f"  Precision:   {np.mean(test_metrics['precision']):.4f} ± {np.std(test_metrics['precision']):.4f}")
        print(f"  Recall:      {np.mean(test_metrics['recall']):.4f} ± {np.std(test_metrics['recall']):.4f}")
        print(f"  IoU:         {np.mean(test_metrics['iou']):.4f} ± {np.std(test_metrics['iou']):.4f}")
    
    print("\n" + "=" * 70)
    print("AVALIAÇÃO CONCLUÍDA")
    print("=" * 70)
    
    # Visualizar algumas predições
    print("\nGerando visualizações...")
    vis_path = latest_dir / 'predictions_visualization.png'
    visualize_results(model, test_ds if len(test_ds) > 0 else val_ds, device, 
                     num_samples=4, save_path=str(vis_path))


if __name__ == '__main__':
    main()
