set -e
eval "$(conda shell.bash hook)"
conda activate facial-enhancement
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
