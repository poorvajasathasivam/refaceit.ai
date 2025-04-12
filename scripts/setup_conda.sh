#!/bin/bash
set -e

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Miniconda or Anaconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create main environment for backend
echo "Creating main Conda environment for backend..."
conda env create -f backend/environment.yml

# Create frontend environment
echo "Creating Conda environment for frontend..."
conda env create -f frontend/environment.yml

# Create development environment
echo "Creating Conda environment for development tools..."
conda env create -f environment-dev.yml

# Setup pre-commit hooks
echo "Setting up pre-commit hooks..."
conda activate facial-enhancement-dev
pre-commit install

echo ""
echo "===== Setup Complete ====="
echo ""
echo "To activate the backend environment, run:"
echo "conda activate facial-enhancement"
echo ""
echo "To activate the frontend environment, run:"
echo "conda activate facial-enhancement-frontend"
echo ""
echo "To activate the development environment, run:"
echo "conda activate facial-enhancement-dev"
