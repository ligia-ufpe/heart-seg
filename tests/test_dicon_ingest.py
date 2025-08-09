
import SimpleITK as sitk
import numpy as np
from pathlib import Path

def test_nifti_can_be_loaded_and_is_3d():
    """
    Testa se o arquivo NIfTI gerado pode ser lido e convertido para numpy array,
    garantindo compatibilidade com MONAI.
    """
    nifti_path = Path('data/processed/CINE_EC_12.nii.gz')
    assert nifti_path.exists(), f"Arquivo não encontrado: {nifti_path}"
    image = sitk.ReadImage(str(nifti_path))
    arr = sitk.GetArrayFromImage(image)
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 3, f"Esperado 3D, mas tem {arr.ndim} dimensões"
    assert arr.shape[0] > 0 and arr.shape[1] > 0 and arr.shape[2] > 0, f"Shape inválido: {arr.shape}"
