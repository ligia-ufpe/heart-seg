import glob
import os

images = sorted(glob.glob('data/processed/CINE_EC_*.nii.gz'))
labels = sorted(glob.glob('data/segmented/CINE_EC_*_SEG.nii'))

print(images)

data_dicts = []
for img in images:
    base = os.path.basename(img).replace('.nii.gz', '')
    label = f'data/segmented/{base}_SEG.nii'
    print(label)
    if os.path.exists(label):
        data_dicts.append({'image': img, 'label': label})

print(data_dicts)