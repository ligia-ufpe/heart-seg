"""
Functions for processing heart images for segmentation tasks.
"""
import glob
import os
import sys
import pathlib
import torch
import numpy as np
from PIL import Image
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord, DivisiblePadd
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet
from utils import visualize_sample, save_pred_label
import cv2
import matplotlib.pyplot as plt

def main():   
    # Diretórios e padrões
    IMAGE_DIR = 'data/processed'
    LABEL_DIR = 'data/segmented'
    OUTPUT_DIR = 'data/test_dataset_rgb'
    IMAGE_PATTERN = 'CINE_EC_*.nii.gz'
    LABEL_SUFFIX = '_SEG.nii'

    # Coletar imagens e rótulos
    images = sorted(glob.glob(os.path.join(IMAGE_DIR, IMAGE_PATTERN)))
    data_dicts = []
    for img in images:
        base = os.path.basename(img).replace('.nii.gz', '')
        label = os.path.join(LABEL_DIR, f'{base}{LABEL_SUFFIX}')
        if os.path.exists(label):
            data_dicts.append({'image': img, 'label': label})
        else:
            print(f'Label não encontrada para: {img}')


    # Transforms com padding divisível por 16
    transforms = Compose([
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        ScaleIntensityd(keys=['image']),
        ToTensord(keys=['image', 'label']),
        DivisiblePadd(keys=["image", "label"], k=16)
    ])

    # Dataset e DataLoader
    train_ds = Dataset(data=data_dicts, transform=transforms)
    train_loader = DataLoader(train_ds, batch_size=1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels'), exist_ok=True)

    for i, batch in enumerate(train_loader):
        # Skip the first sample since its segmentation mask only segments the EAT and not the entire heart
        if i == 0:
            continue

        images = batch['image']
        labels = batch['label']

        for j in range(images.shape[4]):            
            image_slice = images[0, 0, :, :, j].cpu().numpy()
            # Skip the slice if its only one unique value (empty slice)
            if len(np.unique(image_slice)) == 1:
                continue
            
            image_rgb = cv2.cvtColor((image_slice * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            plt.imsave(os.path.join(OUTPUT_DIR, 'images', f'image_{i}_{j}.png'), image_rgb)

            label_slice = labels[0, 0, :, :, j].cpu().numpy()
            label_slice[label_slice == 2] = 1
            plt.imsave(os.path.join(OUTPUT_DIR, 'labels', f'label_{i}_{j}.png'), label_slice, cmap='gray')

if __name__ == "__main__":
    main()