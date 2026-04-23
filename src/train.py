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

from src.dataset import EpicardialDataset, get_patient_splits, get_patient_folds
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


# ─── Excecao para loss nao-finito ──────────────────────────────────────────

class NanLossError(RuntimeError):
    """Sinaliza que uma epoca inteira produziu loss NaN/Inf."""


# ─── Epoch de treino ───────────────────────────────────────────────────────

def train_epoch(
    model, loader, optimizer, criterion, scaler, device,
    use_amp: bool, max_grad_norm: float = 1.0,
):
    """
    Executa uma epoca. Aplica unscale + clip_grad_norm para evitar
    divergencia em modelos tipo transformer (UNETR/SwinUNETR) e pula
    batches com loss nao-finito (protege o running average).

    Levanta NanLossError se todos os batches da epoca forem invalidos.
    """
    model.train()
    total_loss = 0.0
    n_valid    = 0
    n_skipped  = 0

    pbar = tqdm(loader, desc="  train", leave=False, unit="batch")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss    = criterion(outputs, masks)

        if not torch.isfinite(loss):
            n_skipped += 1
            pbar.set_postfix(loss="nan", skipped=n_skipped)
            continue

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += loss.item()
        n_valid   += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    if n_valid == 0:
        raise NanLossError(f"Todos os {n_skipped} batches da epoca produziram loss NaN/Inf")

    return total_loss / n_valid


# ─── Validacao ─────────────────────────────────────────────────────────────

def validate(model, loader, criterion, device, use_amp: bool):
    model.eval()
    total_loss = 0.0
    n_valid    = 0
    all_logits, all_masks = [], []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="  val  ", leave=False, unit="batch"):
            images, masks = images.to(device), masks.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss    = criterion(outputs, masks)

            if torch.isfinite(loss):
                total_loss += loss.item()
                n_valid   += 1
            all_logits.append(outputs.float().cpu())
            all_masks.append(masks.cpu())

    logits_cat = torch.cat(all_logits)
    masks_cat  = torch.cat(all_masks)
    metrics    = compute_all_metrics(logits_cat, masks_cat)

    mean_loss = total_loss / n_valid if n_valid > 0 else float("nan")
    return mean_loss, metrics, logits_cat, masks_cat


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
    splits:         tuple[list, list, list] | None = None,
    save_test_preds: bool = False,
) -> float:
    """
    Executa um treino completo e retorna o melhor val Dice.

    Args:
        trial:            optuna.Trial opcional — ativa reporting e pruning.
        splits:           (train_ids, val_ids, test_ids) explicitos. Se None,
                          usa get_patient_splits(seed).
        save_test_preds:  se True, salva logits+masks do test set para pooling
                          em execucoes K-fold.
    """
    set_seed(seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    pin     = device.type == "cuda"

    # ── Splits ────────────────────────────────────────────────────────────
    if splits is None:
        train_ids, val_ids, test_ids = get_patient_splits(data_dir, seed=seed)
    else:
        train_ids, val_ids, test_ids = splits

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
        val_loss, val_metrics, _, _ = validate(model, val_loader, criterion, device, use_amp)

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

        _, test_metrics, test_logits, test_masks = validate(
            model, test_loader, criterion, device, use_amp
        )

        print("\n  ── Test Set Metrics ──────────────────────────")
        for k, v in test_metrics.items():
            print(f"  {k:<14}: {v:.4f}")

        with open(save_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"\n  Metricas salvas em: {save_dir / 'test_metrics.json'}")

        if save_test_preds:
            torch.save(
                {"logits": test_logits, "masks": test_masks},
                save_dir / "test_predictions.pt",
            )

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

        try:
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
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise optuna.TrialPruned("CUDA OOM — combinacao de hiperparametros muito pesada")
        except NanLossError as e:
            torch.cuda.empty_cache()
            raise optuna.TrialPruned(f"NaN loss — {e}")

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


# ─── K-fold CV ─────────────────────────────────────────────────────────────

def run_kfold(args):
    """K-fold CV ao nivel de paciente (protocolo do paper Rodrigues et al. 2016).

    Cada fold: treina no train-split, valida para early stopping, avalia no
    test-split. Ao final:
      - per_fold_summary: media ± std por metrica entre folds.
      - pooled: Dice/IoU/etc global sobre as predicoes de todos os folds
        concatenadas (equivalente ao protocolo do paper).
    """
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "checkpoints" / f"{args.model}_kfold{args.kfold}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    folds = get_patient_folds(args.data_dir, k=args.kfold, seed=args.seed)
    model_module = models.get(args.model)

    per_fold_metrics = []
    pooled_logits, pooled_masks = [], []

    for fold_idx, (train_ids, val_ids, test_ids) in enumerate(folds):
        print(f"\n{'#' * 65}")
        print(f"#  FOLD {fold_idx + 1}/{args.kfold}")
        print(f"#  train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")
        print(f"#  test_patients={test_ids}")
        print("#" * 65)

        fold_dir = run_dir / f"fold_{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        try:
            train_single_run(
                data_dir        = args.data_dir,
                save_dir        = fold_dir,
                model_name      = args.model,
                model_config    = model_module.DEFAULT_CONFIG,
                epochs          = args.epochs,
                batch_size      = args.batch_size,
                lr              = args.lr,
                patience        = args.patience,
                context_slices  = args.context_slices,
                tversky_alpha   = args.tversky_alpha,
                workers         = args.workers,
                seed            = args.seed,
                splits          = (train_ids, val_ids, test_ids),
                save_test_preds = True,
            )
        except NanLossError as e:
            print(f"  [fold {fold_idx}] ABORTADO: {e}")
            continue
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  [fold {fold_idx}] ABORTADO: CUDA OOM")
            continue

        tm_path = fold_dir / "test_metrics.json"
        if tm_path.exists():
            with open(tm_path) as f:
                per_fold_metrics.append(json.load(f))

        pp_path = fold_dir / "test_predictions.pt"
        if pp_path.exists():
            preds = torch.load(pp_path, weights_only=False)
            pooled_logits.append(preds["logits"])
            pooled_masks.append(preds["masks"])

    # ── Agregacao ─────────────────────────────────────────────────────────
    summary = {}
    if per_fold_metrics:
        keys = list(per_fold_metrics[0].keys())
        summary = {
            k: {
                "mean": float(np.nanmean([m[k] for m in per_fold_metrics])),
                "std":  float(np.nanstd ([m[k] for m in per_fold_metrics])),
            }
            for k in keys
        }

    pooled = {}
    if pooled_logits:
        pooled = compute_all_metrics(torch.cat(pooled_logits), torch.cat(pooled_masks))

    with open(run_dir / "kfold_results.json", "w") as f:
        json.dump({
            "model":            args.model,
            "k":                args.kfold,
            "hyperparameters": {
                "lr":             args.lr,
                "batch_size":     args.batch_size,
                "context_slices": args.context_slices,
                "tversky_alpha":  args.tversky_alpha,
                "epochs":         args.epochs,
                "patience":       args.patience,
                "seed":           args.seed,
            },
            "per_fold":         per_fold_metrics,
            "per_fold_summary": summary,
            "pooled":           pooled,
        }, f, indent=2)

    print(f"\n{'=' * 65}")
    print(f"  K-FOLD CV CONCLUIDO  |  K={args.kfold}  |  Modelo: {args.model}")
    print(f"  Folds completos: {len(per_fold_metrics)}/{args.kfold}")
    if summary:
        print(f"\n  ── Media ± Std entre folds ────────────────────")
        for k, v in summary.items():
            print(f"  {k:<14}: {v['mean']:.4f} ± {v['std']:.4f}")
    if pooled:
        print(f"\n  ── Pooled (paper protocol) ────────────────────")
        for k, v in pooled.items():
            print(f"  {k:<14}: {v:.4f}")
    print(f"\n  Resultados : {run_dir / 'kfold_results.json'}")
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
    # K-fold
    p.add_argument("--kfold",        type=int,   default=10,
                   help="Rodar K-fold CV ao nivel de paciente (ex: --kfold 10 p/ matchear o paper)")
    return p.parse_args()


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.optuna:
        run_optuna(args)
        return

    if args.kfold and args.kfold > 0:
        run_kfold(args)
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
