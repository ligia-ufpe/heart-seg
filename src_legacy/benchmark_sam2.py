import glob
import os
import sys
import pathlib
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Ensure we can import the local sibling `sam2` repository when running this script
# Project layout (example):
#  ../heart-seg/src/test_sam2.py  (this file)
#  ../sam2/sam2/                  (the sam2 package directory)
repo_root = pathlib.Path(__file__).resolve().parents[1]
workspace_root = repo_root.parent
sam2_path = workspace_root / 'sam2'
if str(sam2_path) not in sys.path:
    sys.path.insert(0, str(sam2_path))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord, DivisiblePadd
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet
from utils import visualize_sample, save_pred_label, show_masks, show_points, show_box
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any, Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
BASE_DATASET_DIR = 'data/test_dataset_rgb/'

def predict_with_sam2(predictor: SAM2ImagePredictor,
                      image: np.ndarray, 
                      input_point: np.ndarray, 
                      input_label: np.ndarray,
                      debug: bool = False) -> tuple:
    
    predictor.set_image(image)

    if debug:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_points(input_point, input_label, plt.gca())
        plt.axis('on')
        plt.show()

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    if debug:
        show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)


    return masks, scores, logits

def benchmark_sam2(point_coords: np.ndarray, 
                   point_labels: np.ndarray, 
                   sam2_ckpt: str, 
                   model_cfg: str) -> pd.DataFrame:
    
    print("Loading SAM2 model...")
    sam2_model = build_sam2(model_cfg, sam2_ckpt, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    all_images = sorted(glob.glob(os.path.join(BASE_DATASET_DIR, 'images', '*.png')))
    all_labels = sorted(glob.glob(os.path.join(BASE_DATASET_DIR, 'labels', '*.png')))

    evaluation_table = {"image": [], 
                        "dice": [], 
                        "iou": [], 
                        "accuracy": []}

    print("Starting inference and evaluation...")
    for img_path, label_path in tqdm(zip(all_images[33:34], all_labels[33:34])):
        image = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Imagem de testeo não encontrada em: {img_path}")
            print(f"Pulando")
            continue

        if label is None:
            print(f"Rótulo não encontrada em: {label_path}")
            print(f"Pulando")
            continue

        masks, scores, logits = predict_with_sam2(predictor, image, point_coords, point_labels, debug=True)
        exit()

        # Evaluate prediction
        pred_mask = masks[0].squeeze()
        gt_mask = (label > 0).squeeze()

        accuracy = np.mean(pred_mask == gt_mask)

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = intersection / union if union != 0 else 1.0


        dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) != 0 else 1.0
        
        evaluation_table['image'].append(os.path.basename(img_path))
        evaluation_table['dice'].append(dice)
        evaluation_table['iou'].append(iou)
        evaluation_table['accuracy'].append(accuracy)

    # Convert to pandas DataFrame and save
    df = pd.DataFrame(evaluation_table)
    
    try:
        out_csv = os.path.join(BASE_DATASET_DIR, f'evaluation_results_{os.path.basename(sam2_ckpt)}.csv')
        df.to_csv(out_csv, index=False)
        print(f"Saved evaluation results to: {out_csv}")
    except Exception as e:
        print(f"Warning: could not save evaluation CSV: {e}")
    
    return df

def benchmark_loop_sam2() -> pd.DataFrame:
    # Build a list of per-model summary dicts, then convert to a DataFrame.
    results = []

    point_coords = np.array([[120, 75]])
    point_labels = np.array([1])

    ckpt_list = [
        "../sam2/checkpoints/sam2.1_hiera_tiny.pt",
        "../sam2/checkpoints/sam2.1_hiera_small.pt",
        "../sam2/checkpoints/sam2.1_hiera_base_plus.pt",
        "../sam2/checkpoints/sam2.1_hiera_large.pt",
    ]

    cfg_list = [
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "configs/sam2.1/sam2.1_hiera_l.yaml",
    ]

    for ckpt, cfg in zip(ckpt_list, cfg_list):
        print(f"Benchmarking {ckpt} with config {cfg}...")
        evaluation = benchmark_sam2(point_coords, point_labels, sam2_ckpt=ckpt, model_cfg=cfg)

        # allow mixed value types (str for model, floats for metrics)
        row: Dict[str, Any] = {"model": os.path.basename(ckpt)}

        # evaluation is a DataFrame with columns like 'image', 'dice', 'iou', 'accuracy'
        for col in evaluation.columns:
            if col == 'image':
                continue
            # compute mean and std per metric and store with clear column names
            row[f"{col}_mean"] = float(evaluation[col].mean())
            row[f"{col}_std"] = float(evaluation[col].std())

        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv("full_heart_benchmark_results.csv", index=False)
    return df

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark SAM2 on a small dataset")
    parser.add_argument('--ckpt', 
                        type=str, 
                        default="../sam2/checkpoints/sam2.1_hiera_tiny.pt", 
                        help='Path to SAM2 checkpoint')
    parser.add_argument('--cfg', 
                        type=str, 
                        default="configs/sam2.1/sam2.1_hiera_t.yaml", 
                        help='Path to SAM2 model config')
    parser.add_argument('--x', 
                        type=int, 
                        default=120, 
                        help='x coordinate of a prompt point')
    parser.add_argument('--y', 
                        type=int, 
                        default=75, 
                        help='y coordinate of a prompt point')
    
    args = parser.parse_args()

    point_coords = np.array([[args.x, args.y]])
    point_labels = np.array([1])
    evaluation_table = benchmark_sam2(point_coords, point_labels, sam2_ckpt=args.ckpt, model_cfg=args.cfg)

    print("================ Evaluation Results ================")
    for metric, values in evaluation_table.items():
        if metric == 'image':
            continue
        print(f"  {metric}-Mean: {np.mean(values):.4f}")
        print(f"  {metric}-Std: {np.std(values):.4f}")

if __name__ == "__main__":
    main()
    # df = benchmark_loop_sam2()