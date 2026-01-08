"""
run_model.py

Script principal para executar modelos de segmentacao cardiaca.
Modelos disponiveis:
  - sam2: Segment Anything Model 2 (metodos: coord, auto, unet)
  - unet: U-Net para segmentacao

Uso:
    python src/run_model.py --model sam2 --method auto
    python src/run_model.py --model sam2 --method coord --x 120 --y 75
    python src/run_model.py --model sam2 --method unet
    python src/run_model.py --model unet --train
    python src/run_model.py --model unet --predict --input imagem.png
"""

import argparse
import sys
from pathlib import Path

# Adiciona o diretorio src ao path para imports relativos
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    """Faz o parsing dos argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description='Segmentacao Cardiaca com Modelos Pre-treinados')
    
    # Argumentos gerais
    parser.add_argument('--model', '-m', default=None, 
                        choices=['sam2', 'unet'],
                        help='Modelo a ser utilizado (sam2, unet)')
    parser.add_argument('--input', '-i', default="src/input", 
                        help='Pasta/arquivo de entrada (padrao: src/input)')
    parser.add_argument('--output', '-o', default="src/output", 
                        help='Pasta de saida (padrao: src/output)')
    parser.add_argument('--debug', action='store_true', 
                        help='Mostrar visualizacoes')
    
    # Argumentos SAM2
    parser.add_argument('--method', type=str, default="auto",
                        choices=["coord", "auto", "unet"],
                        help='Metodo SAM2: coord, auto ou unet')
    parser.add_argument('--x', type=int, default=None, 
                        help='Coordenada x (para method=coord)')
    parser.add_argument('--y', type=int, default=None, 
                        help='Coordenada y (para method=coord)')
    parser.add_argument('--size', type=str, default="tiny",
                        choices=["tiny", "small", "base", "large"],
                        help='Tamanho do modelo SAM2')
    parser.add_argument('--ckpt', type=str, default=None, 
                        help='Checkpoint SAM2 customizado')
    parser.add_argument('--cfg', type=str, default=None, 
                        help='Config SAM2 customizado')
    
    # Argumentos U-Net
    parser.add_argument('--train', action='store_true', 
                        help='Treinar modelo U-Net')
    parser.add_argument('--predict', action='store_true', 
                        help='Fazer predicao com U-Net')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Numero de epochs para treino')
    parser.add_argument('--dataset', '-d', type=str, default=None, 
                        help='Diretorio do dataset para treino U-Net')

    return parser.parse_args()


def run_sam2(args):
    """Executa o modelo SAM2."""
    from models.sam2 import (
        run_coord_method, 
        run_auto_mask_method, 
        run_unet_guided_method,
        process_batch,
        CHECKPOINTS
    )
    
    # Configurar checkpoint
    if args.ckpt and args.cfg:
        ckpt, cfg = args.ckpt, args.cfg
    else:
        ckpt, cfg = CHECKPOINTS[args.size]
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    print(f"=== SAM2 Segmentacao ===")
    print(f"Metodo: {args.method}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Modelo: {args.size}")
    print()
    
    # Processar
    if input_path.is_file():
        if args.method == "coord":
            if args.x is None or args.y is None:
                print("Erro: --x e --y sao necessarios para method=coord")
                return
            run_coord_method(input_path, output_path, args.x, args.y, ckpt, cfg, args.debug)
        elif args.method == "auto":
            run_auto_mask_method(input_path, output_path, ckpt, cfg, debug=args.debug)
        elif args.method == "unet":
            run_unet_guided_method(input_path, output_path, ckpt=ckpt, cfg=cfg, debug=args.debug)
    elif input_path.is_dir():
        process_batch(input_path, output_path, args.method, ckpt, cfg, args.x, args.y, args.debug)
    else:
        print(f"Erro: {input_path} nao existe")


def run_unet(args):
    """Executa operacoes com U-Net."""
    from models.unet import train_unet, load_unet_model, predict_unet, DEFAULT_MODEL_PATH, DEFAULT_DATASET_DIR
    import cv2
    
    if args.train:
        print("=== Treinando U-Net ===")
        dataset_dir = Path(args.dataset) if args.dataset else DEFAULT_DATASET_DIR
        train_unet(
            dataset_dir=dataset_dir,
            model_save_path=DEFAULT_MODEL_PATH,
            epochs=args.epochs
        )
    
    elif args.predict:
        if args.input == "src/input":
            print("Erro: --input eh necessario para predicao")
            return
        
        print("=== Predicao U-Net ===")
        input_path = Path(args.input)
        output_path = Path(args.output)
        
        model = load_unet_model(DEFAULT_MODEL_PATH)
        
        if input_path.is_file():
            image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask, centroid = predict_unet(model, image)
            print(f"Centro do coracao: {centroid}")
            
            output_path.mkdir(parents=True, exist_ok=True)
            mask_file = output_path / f"{input_path.stem}_unet_mask.png"
            cv2.imwrite(str(mask_file), mask * 255)
            print(f"Mascara salva em: {mask_file}")
        
        elif input_path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
            for img_file in input_path.glob("*.png"):
                image = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                mask, centroid = predict_unet(model, image)
                mask_file = output_path / f"{img_file.stem}_unet_mask.png"
                cv2.imwrite(str(mask_file), mask * 255)
            print(f"Mascaras salvas em: {output_path}")
    
    else:
        print("Uso U-Net:")
        print("  Treinar:  python src/run_model.py --model unet --train")
        print("  Predizer: python src/run_model.py --model unet --predict --input imagem.png")


def show_help():
    """Mostra ajuda detalhada."""
    print("""
=== Heart Segmentation - Modelos Disponiveis ===

1. SAM2 (Segment Anything Model 2)
   Metodos:
   - coord: Segmenta a partir de coordenadas (x, y)
   - auto:  Gera todas as mascaras automaticamente
   - unet:  Usa U-Net para encontrar o centro, depois SAM2 refina

   Exemplos:
     python src/run_model.py --model sam2 --method auto --input src/input
     python src/run_model.py --model sam2 --method coord --x 120 --y 75 --input imagem.png
     python src/run_model.py --model sam2 --method unet --input src/input

2. U-Net
   - Treinar modelo para segmentacao cardiaca
   - Fazer predicao em novas imagens

   Exemplos:
     python src/run_model.py --model unet --train --epochs 50
     python src/run_model.py --model unet --predict --input imagem.png

=== Opcoes Gerais ===
  --input, -i   Pasta ou arquivo de entrada
  --output, -o  Pasta de saida
  --debug       Mostrar visualizacoes

=== Opcoes SAM2 ===
  --method      coord, auto ou unet
  --size        tiny, small, base ou large
  --x, --y      Coordenadas para method=coord

=== Opcoes U-Net ===
  --train       Treinar o modelo
  --predict     Fazer predicao
  --epochs      Numero de epochs
  --dataset     Diretorio do dataset
""")


def main():
    args = parse_args()
    
    if args.model == "sam2":
        run_sam2(args)
    elif args.model == "unet":
        run_unet(args)
    else:
        show_help()


if __name__ == "__main__":
    main()