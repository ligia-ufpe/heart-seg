import os
import nibabel as nib
import numpy as np
import torch
from train_crop_model import BoundingBoxModel

def load_model(model_path):
    model = BoundingBoxModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def crop_image(image_path, model):
    # Load the image
    image = nib.load(image_path)
    image_data = image.get_fdata()
    image_data = np.expand_dims(image_data, axis=0)  # Add channel dimension
    image_tensor = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Predict bounding box
    with torch.no_grad():
        bbox = model(image_tensor).squeeze(0).numpy()

    # Extract bounding box coordinates
    x_min, y_min, z_min, x_max, y_max, z_max = map(int, bbox)

    # Crop the image
    cropped_data = image_data[0, x_min:x_max, y_min:y_max, z_min:z_max]

    # Create a new NIfTI image
    cropped_image = nib.Nifti1Image(cropped_data, affine=image.affine)
    return cropped_image

def process_images(input_dir, output_dir, model):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.nii'):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            cropped_image = crop_image(input_path, model)
            nib.save(cropped_image, output_path)
            print(f"Cropped image saved to {output_path}")

if __name__ == "__main__":
    model_path = "models/crop_model.pth"  # Path to the trained model
    input_directory = "data/raw"  # Directory containing the input images
    output_directory = "data/processed"  # Directory to save the cropped images

    model = load_model(model_path)
    process_images(input_directory, output_directory, model)