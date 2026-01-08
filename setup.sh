#!/bin/bash

# Setup script for Heart Segmentation project
# This script installs all necessary dependencies

echo "Setting up Heart Segmentation project..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8 or higher first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed. Please install pip first."
    exit 1
fi

# Install dependencies from requirements file
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Optional: Install pytest for testing
echo "Installing pytest for testing..."
pip3 install pytest

echo "Setup complete! You can now run the project scripts."
echo "Example: python src/convert.py --input data/raw/CINE_EC_12 --output data/processed"