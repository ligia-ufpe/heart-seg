import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Define the dataset class
class HeartDataset(Dataset):
    def __init__(self, image_paths, bbox_labels):
        self.image_paths = image_paths
        self.bbox_labels = bbox_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = nib.load(self.image_paths[idx]).get_fdata()
        image = np.expand_dims(image, axis=0)  # Add channel dimension
        bbox = self.bbox_labels[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(bbox, dtype=torch.float32)

# Define a simple model for bounding box prediction
class BoundingBoxModel(nn.Module):
    def __init__(self):
        super(BoundingBoxModel, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(2, 2)
        self.fc1 = nn.Linear(16 * 32 * 32 * 32, 128)  # Adjust dimensions as needed
        self.fc2 = nn.Linear(128, 6)  # Output: x_min, y_min, z_min, x_max, y_max, z_max

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 32 * 32 * 32)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train_model(image_dir, bbox_labels, model_save_path, epochs=10, batch_size=4, learning_rate=0.001):
    # Load data
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.nii')]
    train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, bbox_labels, test_size=0.2, random_state=42)

    train_dataset = HeartDataset(train_paths, train_labels)
    val_dataset = HeartDataset(val_paths, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    model = BoundingBoxModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, bboxes in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, bboxes)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, bboxes in val_loader:
                outputs = model(images)
                loss = criterion(outputs, bboxes)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    # Example usage
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    image_directory = os.path.join(script_dir, "../../data/processed")  # Ensure absolute path to data/processed
    bounding_box_labels = os.path.join(script_dir, "../../data/bbox")  # Ensure absolute path to bbox_labels
    model_output_path = os.path.join(script_dir, "../../models/crop_model.pth")  # Ensure absolute path to models

    train_model(image_directory, bounding_box_labels, model_output_path)