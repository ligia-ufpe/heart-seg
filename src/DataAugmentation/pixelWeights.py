"""
pixelWeights.py
---------------
Gera weight maps por pixel para máscaras de segmentação 2D,
seguindo a abordagem do artigo original da U-Net (Ronneberger et al., 2015).

Fórmula do artigo:
    w(x) = w_c(x) + w_0 * exp( -(d1(x) + d2(x))^2 / (2 * sigma^2) )

onde:
    w_c(x)  = peso de balanceamento de classe (inversamente proporcional à
               frequência de cada classe no slice)
    d1(x)   = distância do pixel x ao contorno da célula/objeto mais próximo
    d2(x)   = distância do pixel x ao contorno do segundo objeto mais próximo
    w_0     = peso máximo aplicado nas fronteiras (default: 10)
    sigma   = largura da "gaussiana" de fronteira em pixels (default: 5)

Para datasets com uma única classe de interesse (gordura cardíaca) e fundo,
d1 é a distância ao contorno da máscara mais próxima e d2 é a distância ao
segundo contorno mais próximo (quando há múltiplos componentes conexos).

Uso:
    # Processar todos os pacientes e salvar weight maps como .npy
    python pixelWeights.py

    # Ou importar as funções individualmente:
    from pixelWeights import compute_weight_map, process_patient
"""

import os
import glob
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import (
    distance_transform_edt,
    label as connected_label,
    binary_erosion,
)
from typing import Optional

# ---------------------------------------------------------------------------
# Hiperparâmetros (mesmos valores do paper original da U-Net)
# ---------------------------------------------------------------------------
W0: float = 10.0    # amplitude do peso de fronteira
SIGMA: float = 5.0  # desvio padrão da gaussiana de fronteira (em pixels)


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def load_mask(path: str | Path) -> np.ndarray:
    """Carrega um .bmp como máscara binária uint8 (0 ou 1)."""
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    # Binariza: qualquer valor > 0 vira 1
    return (arr > 0).astype(np.uint8)


def class_balance_weights(mask: np.ndarray) -> np.ndarray:
    """
    Calcula w_c(x): peso de balanceamento de classe.

    O peso de cada pixel é inversamente proporcional à frequência da sua
    classe no slice (normalizado para que a média seja 1.0).

    mask: array 2D binário (0 = fundo, 1 = objeto)
    """
    h, w = mask.shape
    n_pixels = h * w

    freq_fg = mask.sum()
    freq_bg = n_pixels - freq_fg

    # Evita divisão por zero
    freq_fg = max(freq_fg, 1)
    freq_bg = max(freq_bg, 1)

    w_c = np.where(mask == 1, n_pixels / (2.0 * freq_fg),
                              n_pixels / (2.0 * freq_bg))
    return w_c.astype(np.float32)


def extract_contour(binary_mask: np.ndarray) -> np.ndarray:
    """
    Extrai a borda de uma máscara binária usando erosão morfológica.
    Retorna máscara booleana onde True = pixel de fronteira.
    """
    eroded = binary_erosion(binary_mask, iterations=1)
    return binary_mask.astype(bool) & ~eroded


def border_weight_map(
    mask: np.ndarray,
    w0: float = W0,
    sigma: float = SIGMA,
) -> np.ndarray:
    """
    Calcula o termo de fronteira do weight map da U-Net.

    Separa os componentes conexos da máscara, calcula a distância de cada
    pixel ao contorno de cada componente e usa as duas menores distâncias
    (d1, d2) para montar o mapa gaussiano de fronteira.

    Parâmetros
    ----------
    mask  : array 2D binário (0 fundo, 1 objeto)
    w0    : amplitude máxima do peso de fronteira
    sigma : desvio padrão da gaussiana

    Retorna
    -------
    border_w : array 2D float32 com o peso de fronteira por pixel
    """
    labeled, n_components = connected_label(mask)

    if n_components == 0:
        # Slice sem nenhum objeto — peso de fronteira zerado
        return np.zeros(mask.shape, dtype=np.float32)

    if n_components == 1:
        # Apenas um componente: d2 = infinito → termo gaussiano → 0
        # Mas ainda calculamos d1 para o caso de objetos finos
        contour = extract_contour(mask.astype(bool))
        if contour.sum() == 0:
            return np.zeros(mask.shape, dtype=np.float32)
        # distance_transform_edt mede distância ao True mais próximo
        # Queremos distância ao contorno (True), então invertemos:
        d1 = distance_transform_edt(~contour).astype(np.float32)
        # d2 = infinito → contribuição nula na gaussiana quando somada com d1
        # Para manter consistência com o paper, usamos apenas d1² / (2σ²)
        # quando só há um componente.
        border_w = w0 * np.exp(-(d1 ** 2) / (2.0 * sigma ** 2))
        # Zera dentro dos objetos (os contornos já têm d=0, peso=w0)
        return border_w.astype(np.float32)

    # -----------------------------------------------------------------------
    # Caso geral: múltiplos componentes
    # -----------------------------------------------------------------------
    # Para cada componente, calcula a distância de cada pixel ao seu contorno
    dist_maps = []
    for comp_id in range(1, n_components + 1):
        comp_mask = (labeled == comp_id)
        contour = extract_contour(comp_mask)
        if contour.sum() == 0:
            # Componente sem bordas detectáveis (ex.: 1 pixel isolado)
            dist = distance_transform_edt(~comp_mask).astype(np.float32)
        else:
            dist = distance_transform_edt(~contour).astype(np.float32)
        dist_maps.append(dist)

    # Empilha e pega as 2 menores distâncias por pixel
    dist_stack = np.stack(dist_maps, axis=0)  # (n_comp, H, W)
    dist_stack.sort(axis=0)
    d1 = dist_stack[0]  # menor distância
    d2 = dist_stack[1]  # segunda menor distância

    border_w = w0 * np.exp(-((d1 + d2) ** 2) / (2.0 * sigma ** 2))

    # Zera pixels que estão dentro de qualquer componente
    # (o paper aplica o peso de fronteira apenas no fundo entre objetos)
    inside_any = mask.astype(bool)
    border_w[inside_any] = 0.0

    return border_w.astype(np.float32)


def compute_weight_map(
    mask: np.ndarray,
    w0: float = W0,
    sigma: float = SIGMA,
) -> np.ndarray:
    """
    Calcula o weight map completo para uma máscara 2D:

        w(x) = w_c(x) + border_weight(x)

    Parâmetros
    ----------
    mask  : array 2D binário (0 fundo, 1 objeto)
    w0    : amplitude do peso de fronteira
    sigma : desvio padrão da gaussiana de fronteira

    Retorna
    -------
    weight_map : array 2D float32
    """
    w_c = class_balance_weights(mask)
    b_w = border_weight_map(mask, w0=w0, sigma=sigma)
    return (w_c + b_w).astype(np.float32)


# ---------------------------------------------------------------------------
# Processamento em batch
# ---------------------------------------------------------------------------

def process_patient(
    patient_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    w0: float = W0,
    sigma: float = SIGMA,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """
    Processa todos os slices (.bmp) de um paciente e gera os weight maps.

    Parâmetros
    ----------
    patient_dir : pasta com os arquivos 001.bmp, 002.bmp, ...
    output_dir  : se fornecido, salva os weight maps como .npy nessa pasta
    w0, sigma   : hiperparâmetros do weight map
    verbose     : exibe progresso

    Retorna
    -------
    dict mapeando nome do arquivo -> weight_map (array float32 H×W)
    """
    patient_dir = Path(patient_dir)
    bmp_files = sorted(patient_dir.glob("*.bmp"))

    if not bmp_files:
        print(f"[AVISO] Nenhum .bmp encontrado em {patient_dir}")
        return {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for bmp_path in bmp_files:
        mask = load_mask(bmp_path)
        wmap = compute_weight_map(mask, w0=w0, sigma=sigma)
        results[bmp_path.name] = wmap

        if output_dir is not None:
            out_path = output_dir / (bmp_path.stem + "_weights.npy")
            np.save(out_path, wmap)

        if verbose:
            print(f"  {bmp_path.name} -> min={wmap.min():.3f} "
                  f"max={wmap.max():.3f} mean={wmap.mean():.3f}")

    return results


def process_all_patients(
    groundtruth_root: str | Path,
    output_root: Optional[str | Path] = None,
    w0: float = W0,
    sigma: float = SIGMA,
    verbose: bool = True,
) -> None:
    """
    Itera sobre todos os subfolders de pacientes em groundtruth_root e
    gera os weight maps para cada slice.

    Parâmetros
    ----------
    groundtruth_root : pasta raiz com os 20 subfolders de pacientes
    output_root      : pasta raiz de saída (espelha a estrutura de entrada)
                       Se None, os weight maps não são salvos em disco.
    w0, sigma        : hiperparâmetros
    verbose          : exibe progresso por paciente/slice
    """
    groundtruth_root = Path(groundtruth_root)
    patient_dirs = sorted([d for d in groundtruth_root.iterdir() if d.is_dir()])

    if not patient_dirs:
        print(f"[ERRO] Nenhum subdiretório encontrado em {groundtruth_root}")
        return

    print(f"Encontrados {len(patient_dirs)} pacientes em {groundtruth_root}\n")

    for patient_dir in patient_dirs:
        if verbose:
            print(f"Paciente: {patient_dir.name}")

        out_dir = None
        if output_root is not None:
            out_dir = Path(output_root) / patient_dir.name

        process_patient(
            patient_dir,
            output_dir=out_dir,
            w0=w0,
            sigma=sigma,
            verbose=verbose,
        )

    print("\nProcessamento concluído.")


# ---------------------------------------------------------------------------
# Entrada principal
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Gera weight maps estilo U-Net para máscaras de segmentação 2D."
    )
    parser.add_argument(
        "--gt-root",
        default=os.path.expanduser(
            "~/Projetos/Data/VisualLabHeartFatCT/GroundTruth"
        ),
        help="Pasta raiz com os subdiretórios de pacientes (padrão: %(default)s)",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.expanduser(
            "~/Projetos/Data/VisualLabHeartFatCT/WeightMaps"
        ),
        help="Pasta raiz de saída para os .npy de weight maps (padrão: %(default)s)",
    )
    parser.add_argument(
        "--w0",
        type=float,
        default=W0,
        help=f"Amplitude do peso de fronteira (padrão: {W0})",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=SIGMA,
        help=f"Desvio padrão da gaussiana de fronteira em pixels (padrão: {SIGMA})",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Não salva os weight maps em disco (apenas imprime estatísticas)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suprime a saída por slice",
    )

    args = parser.parse_args()

    process_all_patients(
        groundtruth_root=args.gt_root,
        output_root=None if args.no_save else args.output_root,
        w0=args.w0,
        sigma=args.sigma,
        verbose=not args.quiet,
    )