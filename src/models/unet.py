"""
unet.py

Script para treinar uma U-Net para segmentação cardíaca.
O modelo treinado será salvo em src/models/trained/

Uso:
    python src/models/unet.py --train
    python src/models/unet.py --predict --input path/to/image.png
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

# Adiciona o diretório src ao path
src_dir = Path(__file__).resolve().parents[1]
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from monai.networks.nets import UNet
    from monai.losses import DiceLoss
    from monai.metrics import DiceMetric
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, 
        ToTensord, RandFlipd, RandRotate90d, Resized
    )
    from monai.data import Dataset, DataLoader
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("MONAI não encontrado. Usando implementação básica de U-Net.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurações padrão
DEFAULT_MODEL_PATH = Path(__file__).parent / "trained" / "unet_heart.pt"
DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[2] / "data" / "test_dataset_rgb"


class BasicUNet(nn.Module):
    """U-Net básica caso MONAI não esteja disponível."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        
        # Encoder
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._block(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._block(128, 64)
        
        self.out = nn.Conv2d(64, out_channels, 1)
        
    def _block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return torch.sigmoid(self.out(d1))


def create_unet_model(in_channels: int = 3, out_channels: int = 1) -> nn.Module:
    """Cria o modelo U-Net (MONAI se disponível, senão implementação básica)."""
    if MONAI_AVAILABLE:
        model = UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    else:
        model = BasicUNet(in_channels, out_channels)
    
    return model.to(device)


def load_dataset(dataset_dir: Path, img_size: int = 256) -> Tuple[DataLoader, DataLoader]:
    """Carrega o dataset de imagens e labels."""
    
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    
    all_images = sorted(glob.glob(str(images_dir / "*.png")))
    all_labels = sorted(glob.glob(str(labels_dir / "*.png")))
    
    if len(all_images) == 0:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {images_dir}")
    
    print(f"Encontradas {len(all_images)} imagens e {len(all_labels)} labels")
    
    # Split train/val (80/20)
    split_idx = int(len(all_images) * 0.8)
    
    train_data = [{"image": img, "label": lbl} for img, lbl in zip(all_images[:split_idx], all_labels[:split_idx])]
    val_data = [{"image": img, "label": lbl} for img, lbl in zip(all_images[split_idx:], all_labels[split_idx:])]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Dataset simples sem MONAI transforms complexos
    train_dataset = SimpleDataset(train_data, img_size=img_size, augment=True)
    val_dataset = SimpleDataset(val_data, img_size=img_size, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    return train_loader, val_loader


class SimpleDataset(torch.utils.data.Dataset):
    """Dataset simples para imagens e labels."""
    
    def __init__(self, data_list: list, img_size: int = 256, augment: bool = False):
        self.data_list = data_list
        self.img_size = img_size
        self.augment = augment
        
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data_list[idx]
        
        # Carregar imagem e label
        image = cv2.imread(item["image"], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(item["label"], cv2.IMREAD_GRAYSCALE)
        
        # Redimensionar
        image = cv2.resize(image, (self.img_size, self.img_size))
        label = cv2.resize(label, (self.img_size, self.img_size))
        
        # Normalizar
        image = image.astype(np.float32) / 255.0
        label = (label > 0).astype(np.float32)
        
        # Augmentation simples
        if self.augment and np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()
        
        # Converter para tensor (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = torch.from_numpy(label).unsqueeze(0)
        
        return {"image": image, "label": label}


def train_unet(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    model_save_path: Path = DEFAULT_MODEL_PATH,
    epochs: int = 50,
    lr: float = 1e-4,
    img_size: int = 256
) -> nn.Module:
    """Treina o modelo U-Net."""
    
    print(f"Iniciando treinamento U-Net...")
    print(f"Dataset: {dataset_dir}")
    print(f"Modelo será salvo em: {model_save_path}")
    print(f"Device: {device}")
    
    # Criar diretório de saída
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Carregar dataset
    train_loader, val_loader = load_dataset(dataset_dir, img_size)
    
    # Criar modelo
    model = create_unet_model(in_channels=3, out_channels=1)
    
    # Loss e optimizer
    criterion = nn.BCELoss()
    if MONAI_AVAILABLE:
        dice_loss = DiceLoss(sigmoid=False)
        criterion = lambda pred, target: nn.BCELoss()(pred, target) + dice_loss(pred, target)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calcular Dice
                pred = (outputs > 0.5).float()
                intersection = (pred * labels).sum()
                dice = (2 * intersection) / (pred.sum() + labels.sum() + 1e-8)
                val_dice += dice.item()
        
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Dice: {val_dice:.4f}")
        
        # Salvar melhor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_dice': val_dice,
            }, model_save_path)
            print(f"  -> Modelo salvo! (Val Loss: {val_loss:.4f})")
    
    print(f"\nTreinamento concluído! Melhor modelo salvo em: {model_save_path}")
    return model


def load_unet_model(model_path: Path = DEFAULT_MODEL_PATH) -> nn.Module:
    """Carrega um modelo U-Net treinado."""
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    model = create_unet_model(in_channels=3, out_channels=1)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Modelo carregado de: {model_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Dice: {checkpoint.get('val_dice', 'N/A'):.4f}")
    
    return model


def predict_unet(
    model: nn.Module,
    image: np.ndarray,
    img_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Faz predição com U-Net e retorna a máscara e o centro de massa.
    
    Args:
        model: Modelo U-Net carregado
        image: Imagem RGB (H, W, 3)
        img_size: Tamanho para redimensionar
        
    Returns:
        mask: Máscara binária (H, W)
        centroid: Coordenadas do centro de massa (x, y)
    """
    original_size = image.shape[:2]
    
    # Preprocessar
    img_resized = cv2.resize(image, (img_size, img_size))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Predição
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
    
    # Pós-processamento
    mask = output.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)
    
    # Redimensionar de volta ao tamanho original
    mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    
    # Calcular centro de massa
    if mask.sum() > 0:
        y_coords, x_coords = np.where(mask > 0)
        centroid = (int(x_coords.mean()), int(y_coords.mean()))
    else:
        centroid = (original_size[1] // 2, original_size[0] // 2)
    
    return mask, centroid


def get_heart_centroid(image: np.ndarray, model_path: Path = DEFAULT_MODEL_PATH) -> Tuple[int, int]:
    """
    Função de conveniência para obter o centro do coração usando U-Net.
    
    Args:
        image: Imagem RGB (H, W, 3)
        model_path: Caminho para o modelo treinado
        
    Returns:
        Coordenadas (x, y) do centro do coração
    """
    model = load_unet_model(model_path)
    _, centroid = predict_unet(model, image)
    return centroid


def parse_args() -> argparse.Namespace:
    """Parse argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description="Treinar/usar U-Net para segmentação cardíaca")
    
    parser.add_argument('--train', action='store_true', help='Treinar o modelo')
    parser.add_argument('--predict', action='store_true', help='Fazer predição')
    parser.add_argument('--input', '-i', type=str, default=None, help='Imagem de entrada para predição')
    parser.add_argument('--output', '-o', type=str, default=None, help='Pasta de saída para predições')
    parser.add_argument('--model', '-m', type=str, default=str(DEFAULT_MODEL_PATH), help='Caminho do modelo')
    parser.add_argument('--dataset', '-d', type=str, default=str(DEFAULT_DATASET_DIR), help='Diretório do dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Número de epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.train:
        train_unet(
            dataset_dir=Path(args.dataset),
            model_save_path=Path(args.model),
            epochs=args.epochs,
            lr=args.lr
        )
    
    elif args.predict:
        if args.input is None:
            print("Erro: --input é necessário para predição")
            return
        
        model = load_unet_model(Path(args.model))
        image = cv2.imread(args.input, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask, centroid = predict_unet(model, image)
        
        print(f"Centro do coração detectado: {centroid}")
        
        # Salvar resultado
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            input_name = Path(args.input).stem
            mask_path = output_dir / f"{input_name}_mask.png"
            cv2.imwrite(str(mask_path), mask * 255)
            print(f"Máscara salva em: {mask_path}")
    
    else:
        print("Uso:")
        print("  Treinar: python src/models/unet.py --train")
        print("  Predizer: python src/models/unet.py --predict --input imagem.png")
        print("\nOpções:")
        print("  --dataset: Diretório do dataset para treino")
        print("  --model: Caminho para salvar/carregar modelo")
        print("  --epochs: Número de epochs de treino")
        print("  --output: Pasta de saída para predições")


if __name__ == "__main__":
    main()
