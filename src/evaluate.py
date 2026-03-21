"""
Inferencia e avaliacao de um ou mais modelos para segmentacao epicardica.

Modos de uso:

  # Avaliar um modelo no test set
  uv run python src/evaluate.py --checkpoints checkpoints/segresnet_.../best_model.pth

  # Avaliar multiplos modelos no mesmo conjunto de pacientes
  uv run python src/evaluate.py --checkpoints checkpoints/segresnet_.../best_model.pth checkpoints/unet_.../best_model.pth

  # Pacientes especificos (obrigatorio quando usando multiplos checkpoints com splits diferentes)
  uv run python src/evaluate.py --checkpoints ckpt1.pth ckpt2.pth --patients DSan ACel

  # Salvar mascaras PNG alem das probabilidades
  uv run python src/evaluate.py --checkpoints ckpt1.pth --save_masks

  # Threshold customizado (ex: apos calibracao conformal)
  uv run python src/evaluate.py --checkpoints ckpt1.pth --threshold 0.35

  # Desativar salvamento de prob.npy + gt.npy
  uv run python src/evaluate.py --checkpoints ckpt1.pth --no_save_logits
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import EpicardialDataset
from src.metrics import compute_all_metrics
import src.models as models


# ─── Carregamento do modelo ────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device, model_override: str | None = None):
    ckpt         = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args         = ckpt.get("args", {})
    model_config = ckpt.get("model_config", {})

    context_slices = args.get("context_slices", 1)
    in_ch          = 2 * context_slices + 1

    model_name   = model_override or model_config.get("arch") or args.get("model", "segresnet")
    model_module = models.get(model_name)
    model        = model_module.build(in_ch, model_config, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, {
        "context_slices": context_slices,
        "model_name":     model_name,
        "epoch":          ckpt.get("epoch", "?"),
        "val_dice":       ckpt.get("val_dice"),
    }


# ─── Inferencia ────────────────────────────────────────────────────────────

def run_inference(
    model,
    loader:          DataLoader,
    device:          torch.device,
    threshold:       float,
    save_masks_dir:  Path | None,
    save_logits_dir: Path | None,
    dataset:         EpicardialDataset,
) -> dict:
    all_logits, all_masks = [], []
    sample_idx = 0
    use_amp    = device.type == "cuda"

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="    inferencia", unit="batch", leave=False):
            images = images.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)

            all_logits.append(logits.cpu())
            all_masks.append(masks.cpu())

            if save_masks_dir is not None or save_logits_dir is not None:
                probs_np = torch.sigmoid(logits).cpu().squeeze(1).numpy()  # (B, H, W)
                gt_np    = masks.squeeze(1).numpy()                        # (B, H, W)

                for i in range(probs_np.shape[0]):
                    s    = dataset.samples[sample_idx]
                    stem = Path(s["fat_file"]).stem

                    if save_masks_dir is not None:
                        out_dir = save_masks_dir / s["patient_id"]
                        out_dir.mkdir(parents=True, exist_ok=True)
                        pred_bin = (probs_np[i] >= threshold).astype(np.uint8)
                        Image.fromarray(pred_bin * 255).save(out_dir / f"{stem}_mask.png")

                    if save_logits_dir is not None:
                        out_dir = save_logits_dir / s["patient_id"]
                        out_dir.mkdir(parents=True, exist_ok=True)
                        np.save(out_dir / f"{stem}_prob.npy", probs_np[i].astype(np.float32))
                        np.save(out_dir / f"{stem}_gt.npy",   gt_np[i].astype(np.float32))

                    sample_idx += 1

    return compute_all_metrics(torch.cat(all_logits), torch.cat(all_masks), threshold=threshold)


# ─── Args ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Inferencia - segmentacao de gordura epicardica")
    p.add_argument("--checkpoints", nargs="+", required=True, metavar="CHECKPOINT",
                   help="Um ou mais caminhos para arquivos .pth")
    p.add_argument("--model",       default=None, choices=models.AVAILABLE + [None],
                   help="Sobrescreve a arquitetura para todos os checkpoints. "
                        "Normalmente desnecessario — o modelo e lido do checkpoint.")
    p.add_argument("--data_dir",    default=str(ROOT))
    p.add_argument("--patients",    nargs="+", default=None,
                   help="IDs dos pacientes. Padrao: test set do splits.json do primeiro checkpoint.")
    p.add_argument("--threshold",   type=float, default=0.5)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--workers",     type=int,   default=4)
    p.add_argument("--save_masks",  action="store_true",
                   help="Salva mascaras binarizadas em results/masks/<ckpt>/")
    p.add_argument("--no_save_logits", action="store_true",
                   help="Desativa salvamento de prob.npy + gt.npy em results/logits/<ckpt>/")
    return p.parse_args()


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("  Inferencia / Avaliacao — Segmentacao Epicardica")
    print("=" * 65)
    print(f"  Device     : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Modelos    : {len(args.checkpoints)}")
    print(f"  Threshold  : {args.threshold}")

    # ── Determinar pacientes (a partir do primeiro checkpoint) ────────────
    if args.patients:
        patient_ids = args.patients
        print(f"  Pacientes  : {patient_ids}")
    else:
        first_ckpt  = Path(args.checkpoints[0])
        splits_path = first_ckpt.parent / "splits.json"
        if not splits_path.exists():
            print(f"\nERRO: splits.json nao encontrado em {first_ckpt.parent}")
            print("Use --patients para especificar os pacientes manualmente.")
            sys.exit(1)
        with open(splits_path) as f:
            patient_ids = json.load(f)["test"]
        print(f"  Pacientes  : {patient_ids}  (test set de {splits_path})")

    # ── Diretorios base ───────────────────────────────────────────────────
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # ── Loop por checkpoint ───────────────────────────────────────────────
    all_metrics: dict[str, dict] = {}

    for ckpt_path in args.checkpoints:
        # ex: "segresnet_20260313" ou "unet_20260321/optuna/best_trial"
        try:
            ckpt_id = str(Path(ckpt_path).parent.relative_to(ROOT / "checkpoints")).replace("/", "_")
        except ValueError:
            ckpt_id = Path(ckpt_path).parent.name

        print(f"\n{'─' * 65}")
        print(f"  [{ckpt_id}]")

        model, meta = load_model(ckpt_path, device, model_override=args.model)
        context_slices = meta["context_slices"]

        if isinstance(meta["val_dice"], float):
            print(f"  Epoch      : {meta['epoch']}  |  Val Dice: {meta['val_dice']:.4f}")
        print(f"  Arquitetura: {meta['model_name']}  |  context={context_slices}")

        dataset = EpicardialDataset(
            args.data_dir, patient_ids, augment=False, context_slices=context_slices
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )

        # Diretorios de saida por modelo
        save_masks_dir = None
        if args.save_masks:
            save_masks_dir = results_dir / "masks" / ckpt_id
            save_masks_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Mascaras   : {save_masks_dir}")

        save_logits_dir = None
        if not args.no_save_logits:
            save_logits_dir = results_dir / "logits" / ckpt_id
            save_logits_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Logits     : {save_logits_dir}  (prob.npy + gt.npy por slice)")

        metrics = run_inference(
            model, loader, device, args.threshold,
            save_masks_dir, save_logits_dir, dataset,
        )

        all_metrics[ckpt_id] = metrics

        print(f"  Metricas:")
        for k, v in metrics.items():
            print(f"    {k:<14}: {v:.4f}")

        # Libera memoria do modelo antes do proximo
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Resumo comparativo ────────────────────────────────────────────────
    if len(all_metrics) > 1:
        print(f"\n{'=' * 65}")
        print("  RESUMO COMPARATIVO")
        print(f"{'=' * 65}")
        metric_names = list(next(iter(all_metrics.values())).keys())
        col_w = max(len(k) for k in all_metrics) + 2
        header = f"  {'Modelo':<{col_w}}" + "".join(f"  {m:<10}" for m in metric_names)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for ckpt_id, m in all_metrics.items():
            row = f"  {ckpt_id:<{col_w}}" + "".join(f"  {m[k]:<10.4f}" for k in metric_names)
            print(row)

    # ── Salvar metricas ───────────────────────────────────────────────────
    out_file = results_dir / "eval_metrics.json"
    with open(out_file, "w") as f:
        json.dump({
            "patients":   patient_ids,
            "threshold":  args.threshold,
            "models":     all_metrics,
        }, f, indent=2)
    print(f"\n  Metricas salvas em: {out_file}")


if __name__ == "__main__":
    main()
