"""
Treinamento para segmentacao de gordura epicardica.

Uso — treino simples:
    uv run python src/train.py
    uv run python src/train.py --model unet --epochs 100 --lr 1e-3
    uv run python src/train.py --resume checkpoints/<run>/best_model.pth

Uso — busca de hiperparametros com Optuna:
    uv run python src/train.py --optuna --n_trials 30
    uv run python src/train.py --optuna --n_trials 50 --model unet --epochs 60
"""

import argparse
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import EpicardialDataset, get_patient_splits
from src.losses  import TverskyFocalLoss
from src.metrics import compute_all_metrics
import src.models as models


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


# ─── Loop principal de treino ───────────────────────────────────────────────

def train_single_run(
    data_dir:       str,
    save_dir:       Path,
    model_name:     str,
    model_config:   dict,
    epochs:         int,
    batch_size:     int,
    lr:             float,
    patience:       int,
    context_slices: int,
    tversky_alpha:  float,
    workers:        int,
    seed:           int,
    resume:         str | None = None,
    trial=None,
) -> float:
    """
    Executa um treino completo e retorna o melhor val Dice.

    Args:
        trial: optuna.Trial opcional — quando fornecido, ativa reporting e pruning.
    """
    set_seed(seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    pin     = device.type == "cuda"

    # ── Splits ────────────────────────────────────────────────────────────
    train_ids, val_ids, test_ids = get_patient_splits(data_dir, seed=seed)

    splits_path = save_dir / "splits.json"
    if not splits_path.exists():
        with open(splits_path, "w") as f:
            json.dump({"train": train_ids, "val": val_ids, "test": test_ids}, f, indent=2)

    in_ch = 2 * context_slices + 1

    # ── Datasets & loaders ───────────────────────────────────────────────
    train_ds = EpicardialDataset(data_dir, train_ids, augment=True,  context_slices=context_slices)
    val_ds   = EpicardialDataset(data_dir, val_ids,   augment=False, context_slices=context_slices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=pin)

    # ── Modelo ───────────────────────────────────────────────────────────
    model_module = models.get(model_name)
    model        = model_module.build(in_ch, model_config, device)
    n_params     = sum(p.numel() for p in model.parameters())

    # ── Otimizador, scheduler, scaler ────────────────────────────────────
    criterion = TverskyFocalLoss(alpha=tversky_alpha, beta=1.0 - tversky_alpha, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler    = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Checkpoint / resume ───────────────────────────────────────────────
    start_epoch      = 0
    best_dice        = 0.0
    patience_counter = 0
    history          = {k: [] for k in ["train_loss", "val_loss", "val_dice", "val_iou", "lr"]}

    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device, weights_only=False)
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
    if trial is None:
        print("=" * 65)
        print(f"  {model_name.upper()} — Gordura Epicardica")
        print("=" * 65)
        print(f"  Device      : {device} ({torch.cuda.get_device_name(0) if use_amp else 'CPU'})")
        print(f"  Parametros  : {n_params:,}")
        print(f"  In channels : {in_ch}  (context={context_slices})")
        print(f"  Train/Val   : {len(train_ds)}/{len(val_ds)} slices")
        print(f"  Pacientes   : train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
        print(f"  LR          : {lr}  |  Batch: {batch_size}  |  Patience: {patience}")
        print(f"  Save dir    : {save_dir}")
        print("=" * 65)

    # ── Loop principal ────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        if trial is None:
            print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp)
        val_loss, val_metrics = validate(model, val_loader, criterion, device, use_amp)

        scheduler.step()
        lr_current = optimizer.param_groups[0]["lr"]
        dice = val_metrics["dice"]

        if trial is None:
            iou  = val_metrics["iou"]
            sens = val_metrics["sensitivity"]
            spec = val_metrics["specificity"]
            print(f"  loss  train={train_loss:.4f}  val={val_loss:.4f}")
            print(f"  Dice={dice:.4f}  IoU={iou:.4f}  Sens={sens:.4f}  Spec={spec:.4f}  LR={lr_current:.2e}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(dice)
        history["val_iou"].append(val_metrics["iou"])
        history["lr"].append(lr_current)

        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # ── Early stopping ────────────────────────────────────────────────
        if dice > best_dice:
            best_dice        = dice
            patience_counter = 0
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice":             dice,
                "val_metrics":          val_metrics,
                "args": {
                    "model":           model_name,
                    "context_slices":  context_slices,
                    "batch_size":      batch_size,
                    "lr":              lr,
                    "tversky_alpha":   tversky_alpha,
                    "epochs":          epochs,
                    "patience":        patience,
                    "seed":            seed,
                    "data_dir":        str(data_dir),
                },
                "model_config": {"arch": model_name, **model_config},
            }, save_dir / "best_model.pth")
            if trial is None:
                print(f"  >> Novo melhor modelo salvo!  Dice={dice:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if trial is None:
                    print(f"\n  Early stopping apos {patience} epocas sem melhora.")
                break

        # Checkpoint periodico (apenas em runs normais)
        if trial is None and (epoch + 1) % 10 == 0:
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_dice":         dice,
                "model_config":     {"arch": model_name, **model_config},
            }, save_dir / f"epoch_{epoch + 1:04d}.pth")

        # ── Optuna: reporting e pruning ───────────────────────────────────
        if trial is not None:
            import optuna
            trial.report(dice, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    if trial is None:
        print("\n" + "=" * 65)
        print(f"  TREINO CONCLUIDO  |  Melhor Dice: {best_dice:.4f}")
        print(f"  Checkpoints em: {save_dir}")
        print("=" * 65)

        # ── Avaliacao no test set ─────────────────────────────────────────
        print("\nCarregando melhor modelo para avaliacao no test set...")
        best_ckpt = torch.load(save_dir / "best_model.pth", map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])

        test_ds     = EpicardialDataset(data_dir, test_ids, augment=False, context_slices=context_slices)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=workers, pin_memory=pin)

        _, test_metrics = validate(model, test_loader, criterion, device, use_amp)

        print("\n  ── Test Set Metrics ──────────────────────────")
        for k, v in test_metrics.items():
            print(f"  {k:<14}: {v:.4f}")

        with open(save_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"\n  Metricas salvas em: {save_dir / 'test_metrics.json'}")

    return best_dice


# ─── Optuna ────────────────────────────────────────────────────────────────

def run_optuna(args):
    import optuna

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = ROOT / "checkpoints" / f"{args.model}_{ts}" / "optuna"
    study_dir.mkdir(parents=True, exist_ok=True)

    model_module = models.get(args.model)

    def objective(trial: "optuna.Trial") -> float:
        lr             = trial.suggest_float("lr",            1e-4, 1e-2, log=True)
        batch_size     = trial.suggest_categorical("batch_size",     [4, 8, 16])
        context_slices = trial.suggest_categorical("context_slices", [1, 2, 3])
        tversky_alpha  = trial.suggest_float("tversky_alpha",  0.2, 0.5)
        model_config   = model_module.optuna_space(trial)

        trial_dir = study_dir / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        return train_single_run(
            data_dir       = args.data_dir,
            save_dir       = trial_dir,
            model_name     = args.model,
            model_config   = model_config,
            epochs         = args.epochs,
            batch_size     = batch_size,
            lr             = lr,
            patience       = args.patience,
            context_slices = context_slices,
            tversky_alpha  = tversky_alpha,
            workers        = args.workers,
            seed           = args.seed,
            trial          = trial,
        )

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study  = optuna.create_study(
        direction  = "maximize",
        pruner     = pruner,
        study_name = args.study_name,
    )
    study.optimize(objective, n_trials=args.n_trials)

    # ── Resultados ────────────────────────────────────────────────────────
    results = {
        "best_trial":  study.best_trial.number,
        "best_dice":   study.best_value,
        "best_params": study.best_params,
        "model":       args.model,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
            for t in study.trials
        ],
    }
    with open(study_dir / "study_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Copia melhor trial ────────────────────────────────────────────────
    best_src = study_dir / f"trial_{study.best_trial.number:03d}"
    best_dst = study_dir / "best_trial"
    shutil.copytree(best_src, best_dst)

    print(f"\n{'=' * 65}")
    print(f"  OPTUNA CONCLUIDO  |  {args.n_trials} trials  |  Modelo: {args.model}")
    print(f"  Melhor trial : {study.best_trial.number}")
    print(f"  Melhor Dice  : {study.best_value:.4f}")
    print(f"  Melhores params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print(f"\n  Resultados  : {study_dir / 'study_results.json'}")
    print(f"  Melhor ckpt : {best_dst / 'best_model.pth'}")
    print("=" * 65)


# ─── Args ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Treino - segmentacao de gordura epicardica")
    p.add_argument("--model",        default="segresnet", choices=models.AVAILABLE,
                   help=f"Arquitetura do modelo. Opcoes: {models.AVAILABLE}")
    p.add_argument("--data_dir",     default=str(ROOT))
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--patience",     type=int,   default=20,
                   help="Epocas sem melhora para early stopping")
    p.add_argument("--context_slices", type=int, default=1)
    p.add_argument("--tversky_alpha",  type=float, default=0.3)
    p.add_argument("--workers",      type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--resume",       default=None, help="Checkpoint para continuar")
    # Optuna
    p.add_argument("--optuna",       action="store_true",
                   help="Ativar busca de hiperparametros com Optuna")
    p.add_argument("--n_trials",     type=int,   default=30,
                   help="Numero de trials Optuna (so com --optuna)")
    p.add_argument("--study_name",   default="epicardial_seg",
                   help="Nome do estudo Optuna")
    return p.parse_args()


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.optuna:
        run_optuna(args)
        return

    # Treino simples
    set_seed(args.seed)

    if args.resume:
        save_dir = Path(args.resume).parent
    else:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = ROOT / "checkpoints" / f"{args.model}_{ts}"
        save_dir.mkdir(parents=True, exist_ok=True)

    model_module = models.get(args.model)

    train_single_run(
        data_dir       = args.data_dir,
        save_dir       = save_dir,
        model_name     = args.model,
        model_config   = model_module.DEFAULT_CONFIG,
        epochs         = args.epochs,
        batch_size     = args.batch_size,
        lr             = args.lr,
        patience       = args.patience,
        context_slices = args.context_slices,
        tversky_alpha  = args.tversky_alpha,
        workers        = args.workers,
        seed           = args.seed,
        resume         = args.resume,
    )


if __name__ == "__main__":
    main()
