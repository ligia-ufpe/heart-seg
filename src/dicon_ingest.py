"""
dicom_ingest.py

Script para:
 - Ler uma série DICOM (pasta com DICOMs)
 - (Opcional) anonimizar os arquivos DICOM em saída
 - Converter a série para NIfTI (.nii.gz)

Uso exemplo:
    python src/dicon_ingest.py --input /path/to/dicom_folder --output /workspace/data/converted --anonymize
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


def anonymize_and_copy(src_folder: Path, dst_folder: Path, overwrite: bool = False) -> None:
    """
    Copia e anonimiza arquivos DICOM de uma pasta para outra.
    Args:
        src_folder (Path): Pasta de origem.
        dst_folder (Path): Pasta de destino.
        overwrite (bool): Sobrescrever arquivos existentes.
    """
    dst_folder.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_folder.iterdir()):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f))
            # remove tags identificáveis simples
            for tag in ["PatientName", "PatientID", "PatientBirthDate", "PatientAddress", "InstitutionName"]:
                if tag in ds:
                    ds.data_element(tag).value = "ANON"
            ds.remove_private_tags()
            out_path = dst_folder / f.name
            if out_path.exists() and not overwrite:
                logger.info(f"Pulando (já existe): {out_path}")
                continue
            ds.save_as(str(out_path))
        except Exception as e:
            logger.warning(f"Falha ao anonimizar/copiar {f}: {e}")


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
    parser.add_argument('--input', '-i', required=True, help='Pasta contendo arquivos DICOM (uma série)')
    parser.add_argument('--output', '-o', required=True, help='Pasta de saída para NIfTI (arquivo .nii.gz será criado ou pasta de saída)')
    parser.add_argument('--anonymize', action='store_true', help='Se setado, copia e anonimiza os DICOMs para uma pasta _anon antes de converter')
    parser.add_argument('--overwrite', action='store_true', help='Sobrescrever arquivos existentes')
    parser.add_argument('--loglevel', default='INFO', help='Nível de logging (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--output-name', default=None, help='Nome do arquivo de saída (opcional, padrão: nome da pasta de entrada)')
    return parser.parse_args()

def process_dicom_to_nifti(
    input_path: Path,
    output_path: Path,
    anonymize: bool = False,
    overwrite: bool = False,
    output_name: Optional[str] = None
) -> Path:
    """
    Processa uma série DICOM e salva como NIfTI.
    Args:
        input_path (Path): Pasta de entrada com DICOMs.
        output_path (Path): Pasta ou arquivo de saída.
        anonymize (bool): Se True, anonimiza antes de converter.
        overwrite (bool): Se True, sobrescreve arquivos existentes.
        output_name (str, opcional): Nome do arquivo de saída.
    Returns:
        Path: Caminho do arquivo NIfTI gerado.
    """
    if not input_path.exists():
        logger.error(f"Pasta de entrada não existe: {input_path}")
        raise FileNotFoundError(f"Pasta de entrada não existe: {input_path}")

    work_input = input_path
    if anonymize:
        anon_folder = input_path.parent / (input_path.name + "_anon")
        logger.info(f"Anonymizing DICOMs -> {anon_folder}")
        anonymize_and_copy(input_path, anon_folder, overwrite=overwrite)
        work_input = anon_folder

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

def main():
    args = parse_args()
    setup_logging(getattr(logging, args.loglevel.upper(), logging.INFO))
    try:
        process_dicom_to_nifti(
            input_path=Path(args.input),
            output_path=Path(args.output),
            anonymize=args.anonymize,
            overwrite=args.overwrite,
            output_name=args.output_name
        )
    except Exception as e:
        logger.error(f"Erro no processamento: {e}")


if __name__ == '__main__':
    main()