# AI Facial Enhancement System

A complete MLOps project demonstrating the deployment of a facial enhancement AI system using GAN/Diffusion models.

## Setup Instructions

### Prerequisites

- Git
- Conda (Miniconda or Anaconda)
- Docker and Docker Compose (for deployment)

### Development Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/poorvajasathasivam/refaceit.ai.git
   cd facial-enhancement
   ```

2. Set up Conda environments:
   ```bash
   ./scripts/setup_conda.sh
   ```

3. Download model weights:
   ```bash
   ./scripts/download_model.sh
   ```

4. Activate the backend environment:
   ```bash
   conda activate facial-enhancement
   ```

### Running the Application Locally

1. Start the backend (in the facial-enhancement conda environment):
   ```bash
   cd backend
   uvicorn app:app --reload
   ```

2. In a separate terminal, activate frontend environment and start Streamlit:
   ```bash
   conda activate facial-enhancement-frontend
   cd frontend
   streamlit run streamlit_app.py
   ```

3. Access the application:
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000

### Docker Deployment

```bash
docker-compose up --build
```

