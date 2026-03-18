"""
Inferencia e avaliacao do SegResNet treinado.

Modos de uso:

  # Avaliar no test set (usa splits.json do checkpoint)
  uv run python src/evaluate.py --checkpoint checkpoints/segresnet_20260313_131412/best_model.pth

  # Avaliar em pacientes especificos
  uv run python src/evaluate.py --checkpoint checkpoints/segresnet_20260313_131412/best_model.pth --patients DSan ACel

  # Salvar mascaras preditas como PNG
  uv run python src/evaluate.py --checkpoint checkpoints/segresnet_20260313_131412/best_model.pth --save_masks

  # Usar threshold customizado (ex: conformal prediction)
  uv run python src/evaluate.py --checkpoint checkpoints/segresnet_20260313_131412/best_model.pth --threshold 0.35
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
from src.metrics import compute_all_metrics, _to_numpy_binary

from monai.networks.nets import SegResNet


def load_model(checkpoint_path: str, device: torch.device) -> tuple[SegResNet, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    context_slices = args.get("context_slices", 1)
    in_ch = 2 * context_slices + 1

    model = SegResNet(
        spatial_dims=2,
        in_channels=in_ch,
        out_channels=1,
        init_filters=32,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        dropout_prob=0.2,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Epoch      : {ckpt.get('epoch', '?')}")
    print(f"  Val Dice   : {ckpt.get('val_dice', '?'):.4f}" if isinstance(ckpt.get('val_dice'), float) else "")
    print(f"  In channels: {in_ch}  (context={context_slices})")

    return model, {"context_slices": context_slices, "args": args}


def run_inference(
    model: SegResNet,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    save_masks_dir: Path | None,
    patient_ids: list[str],
    dataset: EpicardialDataset,
) -> dict:
    all_logits, all_masks = [], []
    sample_idx = 0

    use_amp = device.type == "cuda"

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="  inferencia", unit="batch"):
            images = images.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)

            all_logits.append(logits.cpu())
            all_masks.append(masks.cpu())

            if save_masks_dir is not None:
                probs = torch.sigmoid(logits).cpu()
                preds = _to_numpy_binary(probs.squeeze(1), threshold)  # (B, H, W)
                for pred in preds:
                    s = dataset.samples[sample_idx]
                    out_dir = save_masks_dir / s["patient_id"]
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fname = Path(s["fat_file"]).stem + "_mask.png"
                    Image.fromarray((pred * 255).astype(np.uint8)).save(out_dir / fname)
                    sample_idx += 1

    logits_cat = torch.cat(all_logits)
    masks_cat  = torch.cat(all_masks)
    metrics    = compute_all_metrics(logits_cat, masks_cat, threshold=threshold)
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Inferencia SegResNet - gordura epicardica")
    p.add_argument("--checkpoint", required=True, help="Caminho para o .pth")
    p.add_argument("--data_dir",   default=str(ROOT), help="Raiz do dataset")
    p.add_argument("--patients",   nargs="+", default=None,
                   help="IDs dos pacientes a avaliar (padrao: test set do splits.json)")
    p.add_argument("--threshold",  type=float, default=0.5, help="Threshold de binarizacao")
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--workers",    type=int,   default=4)
    p.add_argument("--save_masks", action="store_true",
                   help="Salva mascaras preditas em results/masks/")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  SEGRESNET — Inferencia / Avaliacao")
    print("=" * 60)
    print(f"  Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    model, meta = load_model(args.checkpoint, device)
    context_slices = meta["context_slices"]

    # Determinar pacientes
    if args.patients:
        patient_ids = args.patients
        print(f"  Pacientes : {patient_ids}")
    else:
        splits_path = Path(args.checkpoint).parent / "splits.json"
        if not splits_path.exists():
            print(f"ERRO: splits.json nao encontrado em {splits_path.parent}")
            print("Use --patients para especificar os pacientes manualmente.")
            sys.exit(1)
        with open(splits_path) as f:
            splits = json.load(f)
        patient_ids = splits["test"]
        print(f"  Pacientes : {patient_ids}  (test set de splits.json)")

    print(f"  Threshold : {args.threshold}")

    # Dataset
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

    # Diretorio de mascaras
    save_masks_dir = None
    if args.save_masks:
        save_masks_dir = ROOT / "results" / "masks"
        save_masks_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Mascaras  : {save_masks_dir}")

    print("=" * 60)

    metrics = run_inference(model, loader, device, args.threshold, save_masks_dir, patient_ids, dataset)

    print("\n  ── Metricas ─────────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<14}: {v:.4f}")

    # Salvar metricas
    out_dir  = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "eval_metrics.json"
    with open(out_file, "w") as f:
        json.dump({"patients": patient_ids, "threshold": args.threshold, "metrics": metrics}, f, indent=2)
    print(f"\n  Metricas salvas em: {out_file}")


if __name__ == "__main__":
    main()
