import io
import os
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from PIL import Image
import base64
from prometheus_fastapi_instrumentator import Instrumentator
import logging

from model import FacialEnhancementModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Facial Enhancement API", version="1.0.0")

# Set up Prometheus metrics 
# This must be done before the application starts
instrumentator = Instrumentator().instrument(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
MODEL_PATH = os.getenv("MODEL_PATH", "../model_weights/GFPGANv1.3.pth")
model = None

@app.on_event("startup")
async def startup_event():
    global model
    # Expose metrics after app startup
    instrumentator.expose(app)
    
    # Load model
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        model = FacialEnhancementModel(MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/enhance/")
async def enhance_image(file: UploadFile = File(...)):
    """
    Enhance facial features in the uploaded image
    
    Args:
        file: Image file to enhance
    
    Returns:
        JSON with base64 encoded enhanced image
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Process the image
        logger.info(f"Processing image: {file.filename}")
        enhanced_image = model.enhance(image)
        
        # Convert enhanced image to base64
        buffered = io.BytesIO()
        enhanced_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Log processing time
        processing_time = time.time() - start_time
        logger.info(f"Image processed in {processing_time:.2f} seconds")
        
        return JSONResponse(content={
            "enhanced_image": img_str,
            "processing_time": processing_time
        })
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing image: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)