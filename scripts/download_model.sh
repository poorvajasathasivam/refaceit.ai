#!/bin/bash
set -e

# Create directory for model weights if it doesn't exist
mkdir -p model_weights

echo "Downloading GFPGAN model weights..."

# Download GFPGAN model
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -O model_weights/GFPGANv1.3.pth

echo "GFPGAN model weights downloaded to model_weights/GFPGANv1.3.pth"
echo "Model setup complete!"
