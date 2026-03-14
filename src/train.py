"""
Treinamento do SegResNet (MONAI) para segmentacao de gordura epicardica.

Uso:
    uv run python src/train.py
    uv run python src/train.py --epochs 100 --batch_size 8 --lr 1e-3
    uv run python src/train.py --resume checkpoints/<run>/best_model.pth
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# -- adiciona raiz do projeto ao path para imports de src.*
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import EpicardialDataset, get_patient_splits
from src.losses  import TverskyFocalLoss
from src.metrics import compute_all_metrics

from monai.networks.nets import SegResNet
from monai.metrics      import DiceMetric


# ─── Reproducibilidade ─────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# ─── Epoch de treino ───────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler, device, use_amp: bool):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="  train", leave=False, unit="batch")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss    = criterion(outputs, masks)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


# ─── Validacao ─────────────────────────────────────────────────────────────

def validate(model, loader, criterion, device, use_amp: bool):
    model.eval()
    total_loss = 0.0
    all_logits, all_masks = [], []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="  val  ", leave=False, unit="batch"):
            images, masks = images.to(device), masks.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss    = criterion(outputs, masks)

            total_loss += loss.item()
            all_logits.append(outputs.cpu())
            all_masks.append(masks.cpu())

    logits_cat = torch.cat(all_logits)
    masks_cat  = torch.cat(all_masks)
    metrics    = compute_all_metrics(logits_cat, masks_cat)

    return total_loss / len(loader), metrics


# ─── Main ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Treino SegResNet - gordura epicardica")
    p.add_argument("--data_dir",      default=str(ROOT), help="Raiz do dataset")
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--patience",      type=int,   default=20)
    p.add_argument("--context_slices",type=int,   default=1,   help="Slices adjacentes (2.5D)")
    p.add_argument("--workers",       type=int,   default=4)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--resume",        default=None, help="Checkpoint para continuar")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # ── Diretorio de saida ────────────────────────────────────────────────
    if args.resume:
        save_dir = Path(args.resume).parent
        print(f"Retomando treino de: {args.resume}")
    else:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = ROOT / "checkpoints" / f"segresnet_{ts}"
        save_dir.mkdir(parents=True, exist_ok=True)

    # ── Splits ────────────────────────────────────────────────────────────
    train_ids, val_ids, test_ids = get_patient_splits(args.data_dir, seed=args.seed)

    # Salvar splits para reproducibilidade
    splits_path = save_dir / "splits.json"
    if not splits_path.exists():
        with open(splits_path, "w") as f:
            json.dump({"train": train_ids, "val": val_ids, "test": test_ids}, f, indent=2)

    in_ch = 2 * args.context_slices + 1

    # ── Datasets & loaders ───────────────────────────────────────────────
    train_ds = EpicardialDataset(args.data_dir, train_ids, augment=True,  context_slices=args.context_slices)
    val_ds   = EpicardialDataset(args.data_dir, val_ids,   augment=False, context_slices=args.context_slices)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=pin)

    # ── Modelo ───────────────────────────────────────────────────────────
    model = SegResNet(
        spatial_dims=2,
        in_channels=in_ch,
        out_channels=1,
        init_filters=32,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        dropout_prob=0.2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    # ── Otimizador, scheduler, scaler ────────────────────────────────────
    criterion = TverskyFocalLoss(alpha=0.3, beta=0.7, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler    = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Checkpoint / resume ───────────────────────────────────────────────
    start_epoch = 0
    best_dice   = 0.0
    history     = {k: [] for k in ["train_loss", "val_loss", "val_dice", "val_iou", "lr"]}

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dice   = ckpt.get("val_dice", 0.0)
        hist_path   = save_dir / "history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                history = json.load(f)
        print(f"  Retomando da epoca {start_epoch} | best Dice: {best_dice:.4f}")

    # ── Header ────────────────────────────────────────────────────────────
    print("=" * 65)
    print("  SEGRESNET — Gordura Epicardica (MONAI Baseline)")
    print("=" * 65)
    print(f"  Device      : {device} ({torch.cuda.get_device_name(0) if use_amp else 'CPU'})")
    print(f"  Parametros  : {n_params:,}")
    print(f"  In channels : {in_ch}  (context={args.context_slices})")
    print(f"  Train/Val   : {len(train_ds)}/{len(val_ds)} slices")
    print(f"  Pacientes   : train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
    print(f"  Save dir    : {save_dir}")
    print("=" * 65)

    # ── Loop principal ────────────────────────────────────────────────────
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp)
        val_loss, val_metrics = validate(model, val_loader, criterion, device, use_amp)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        dice = val_metrics["dice"]
        iou  = val_metrics["iou"]
        sens = val_metrics["sensitivity"]
        spec = val_metrics["specificity"]

        print(f"  loss  train={train_loss:.4f}  val={val_loss:.4f}")
        print(f"  Dice={dice:.4f}  IoU={iou:.4f}  Sens={sens:.4f}  Spec={spec:.4f}  LR={lr:.2e}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(dice)
        history["val_iou"].append(iou)
        history["lr"].append(lr)

        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Melhor modelo
        if dice > best_dice:
            best_dice        = dice
            patience_counter = 0
            torch.save({
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "val_dice":            dice,
                "val_metrics":         val_metrics,
                "args":                vars(args),
            }, save_dir / "best_model.pth")
            print(f"  >> Novo melhor modelo salvo!  Dice={dice:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping apos {args.patience} epocas sem melhora.")
                break

        # Checkpoint periodico
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_dice":         dice,
            }, save_dir / f"epoch_{epoch + 1:04d}.pth")

    print("\n" + "=" * 65)
    print(f"  TREINO CONCLUIDO  |  Melhor Dice: {best_dice:.4f}")
    print(f"  Checkpoints em: {save_dir}")
    print("=" * 65)

    # ── Avaliacao no test set ─────────────────────────────────────────────
    print("\nCarregando melhor modelo para avaliacao no test set...")
    best_ckpt = torch.load(save_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_ds     = EpicardialDataset(args.data_dir, test_ids, augment=False, context_slices=args.context_slices)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=pin)

    _, test_metrics = validate(model, test_loader, criterion, device, use_amp)

    print("\n  ── Test Set Metrics ──────────────────────────")
    for k, v in test_metrics.items():
        print(f"  {k:<14}: {v:.4f}")

    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\n  Metricas salvas em: {save_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
