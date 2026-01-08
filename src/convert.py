"""
convert.py

Script para:
 - Ler uma série DICOM (pasta com DICOMs
 - Converter a série para NIfTI (.nii.gz)

Uso exemplo:
    python src/dicon_ingest.py --input /path/to/dicom_folder --output /workspace/data/converted 
"""

import argparse
from pathlib import Path
import logging
from typing import Tuple, List, Optional
import SimpleITK as sitk
import pydicom

logger = logging.getLogger('dicom_ingest')

def setup_logging(level: int = logging.INFO) -> None:
    """Configura o logging do script."""
    logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')


def read_dicom_series(folder_path: Path) -> Tuple[sitk.Image, List[str]]:
    """
    Lê uma série DICOM de um diretório.
    Args:
        folder_path (Path): Caminho para a pasta com arquivos DICOM.
    Returns:
        Tuple[SimpleITK.Image, List[str]]: Imagem e lista de arquivos da série.
    Raises:
        RuntimeError: Se nenhuma série DICOM for encontrada.
    """
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(str(folder_path))
    if not series_IDs:
        raise RuntimeError(f"Nenhuma série DICOM encontrada em: {folder_path}")
    # pega a primeira série por padrão
    series_files = reader.GetGDCMSeriesFileNames(str(folder_path), series_IDs[0])
    reader.SetFileNames(series_files)
    image = reader.Execute()
    return image, list(series_files)



def write_nifti_from_series(image: sitk.Image, out_path: Path) -> None:
    """
    Salva uma imagem SimpleITK como NIfTI.
    Args:
        image (sitk.Image): Imagem a ser salva.
        out_path (Path): Caminho do arquivo de saída.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(out_path))
    logger.info(f"Escrito NIfTI: {out_path}")



def parse_args() -> argparse.Namespace:
    """Faz o parsing dos argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description='Ingestão DICOM -> NIfTI')
    parser.add_argument('--input', '-i', default=None, help='Pasta contendo arquivos DICOM (padrão: data/raw)')
    parser.add_argument('--output', '-o', default=None, help='Pasta de saída para NIfTI (padrão: data/processed)')
    parser.add_argument('--overwrite', action='store_true', help='Sobrescrever arquivos existentes')
    parser.add_argument('--loglevel', default='INFO', help='Nível de logging (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--output-name', default=None, help='Nome do arquivo de saída (opcional, padrão: nome da pasta de entrada)')
    return parser.parse_args()

def process_dicom_to_nifti(
    input_path: Path,
    output_path: Path,
    overwrite: bool = False,
    output_name: Optional[str] = None
) -> Path:
    """
    Processa uma série DICOM e salva como NIfTI.
    Args:
        input_path (Path): Pasta de entrada com DICOMs.
        output_path (Path): Pasta ou arquivo de saída
        overwrite (bool): Se True, sobrescreve arquivos existentes.
        output_name (str, opcional): Nome do arquivo de saída.
    Returns:
        Path: Caminho do arquivo NIfTI gerado.
    """
    if not input_path.exists():
        logger.error(f"Pasta de entrada não existe: {input_path}")
        raise FileNotFoundError(f"Pasta de entrada não existe: {input_path}")

    work_input = input_path

    image, _ = read_dicom_series(work_input)

    # determina saída .nii.gz
    if output_path.is_dir() or str(output_path).endswith(('/', '\\')):
        out_name = output_name or f"{input_path.name}.nii.gz"
        out_file = output_path / out_name
    else:
        out_file = output_path
        if not str(out_file).endswith('.nii') and not str(out_file).endswith('.nii.gz'):
            out_file = out_file.with_suffix('.nii.gz')

    write_nifti_from_series(image, out_file)
    logger.info(f'Concluído: {out_file}')
    return out_file

def process_all_dicom_folders(
    raw_dir: Path,
    processed_dir: Path,
    overwrite: bool = False
) -> List[Path]:
    """
    Processa todas as pastas DICOM em um diretório e salva como NIfTI.
    Args:
        raw_dir (Path): Pasta contendo subpastas com séries DICOM.
        processed_dir (Path): Pasta de saída para arquivos NIfTI.
        overwrite (bool): Se True, sobrescreve arquivos existentes.
    Returns:
        List[Path]: Lista de caminhos dos arquivos NIfTI gerados.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_files = []
    
    # Itera sobre todas as subpastas em raw_dir
    for folder in sorted(raw_dir.iterdir()):
        if folder.is_dir():
            try:
                out_file = process_dicom_to_nifti(
                    input_path=folder,
                    output_path=processed_dir,
                    overwrite=overwrite,
                    output_name=f"{folder.name}.nii.gz"
                )
                output_files.append(out_file)
            except Exception as e:
                logger.error(f"Erro ao processar {folder.name}: {e}")
    
    logger.info(f"Processados {len(output_files)} arquivos DICOM.")
    return output_files


def main():
    args = parse_args()
    setup_logging(getattr(logging, args.loglevel.upper(), logging.INFO))
    
    # Define caminhos padrão baseado na estrutura do projeto
    script_dir = Path(__file__).parent.parent  # Volta para raiz do projeto
    default_raw = script_dir / "data" / "raw"
    default_processed = script_dir / "data" / "processed"
    
    # Usa caminhos padrão se não forem especificados
    input_path = Path(args.input) if args.input else default_raw
    output_path = Path(args.output) if args.output else default_processed
    
    # Se input for um diretório com subpastas DICOM, processa todas
    if input_path.is_dir():
        # Verifica se há subpastas (modo batch)
        subdirs = [d for d in input_path.iterdir() if d.is_dir()]
        if subdirs:
            # Modo batch: processa todas as subpastas
            logger.info(f"Modo batch: processando {len(subdirs)} pastas DICOM")
            process_all_dicom_folders(
                raw_dir=input_path,
                processed_dir=output_path,
                overwrite=args.overwrite
            )
            return
    
    # Modo individual: processa uma única série
    try:
        process_dicom_to_nifti(
            input_path=input_path,
            output_path=output_path,
            overwrite=args.overwrite,
            output_name=args.output_name
        )
    except Exception as e:
        logger.error(f"Erro no processamento: {e}")


if __name__ == '__main__':
    main()
