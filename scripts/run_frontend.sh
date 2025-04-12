#!/bin/bash
set -e

# Activate the Conda environment
eval "$(conda shell.bash hook)"
conda activate facial-enhancement-frontend

# Change to frontend directory
cd frontend

# Start the Streamlit app
streamlit run streamlit_app.py
