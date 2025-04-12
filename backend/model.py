import os
import sys
import numpy as np
from PIL import Image
import torch
import logging
from typing import Union
import cv2

# Import GFPGAN
try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGAN_AVAILABLE = False
    print("GFPGAN not available. Falling back to basic enhancement.")

logger = logging.getLogger(__name__)

class FacialEnhancementModel:
    """
    Wrapper for GFPGAN facial enhancement model
    """
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the facial enhancement model
        
        Args:
            model_path: Path to the pretrained model weights
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize GFPGAN
        self.model = None
        if GFPGAN_AVAILABLE:
            try:
                self._init_gfpgan()
                logger.info("GFPGAN model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load GFPGAN: {e}")
                logger.warning("Falling back to basic enhancement")
        else:
            logger.warning("GFPGAN not available. Using basic enhancement.")
    
    def _init_gfpgan(self):
        """Initialize GFPGAN model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Initialize model
        self.model = GFPGANer(
            model_path=self.model_path,
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
            device=self.device
        )
        
    def _basic_enhance(self, img_array):
        """Basic enhancement fallback if GFPGAN fails"""
        # Convert RGB to BGR for OpenCV if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array.copy()
        
        # Apply basic image enhancement
        # Denoise
        img_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, 5, 5, 7, 21)
        
        # Enhance details (sharpening)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img_bgr = cv2.filter2D(img_bgr, -1, kernel)
        
        # Enhance colors
        img_bgr = cv2.convertScaleAbs(img_bgr, alpha=1.1, beta=10)
        
        # Convert back to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return img_rgb
        return img_bgr
        
    def enhance(self, image: Union[Image.Image, np.ndarray, bytes]) -> Image.Image:
        """
        Enhance a facial image
        
        Args:
            image: Input image (PIL Image, numpy array, or bytes)
            
        Returns:
            Enhanced image as PIL Image
        """
        # Handle different input types
        if isinstance(image, bytes):
            import io
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Use GFPGAN if available
        if self.model is not None:
            try:
                # GFPGAN expects BGR input
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img_array
                
                # Process with GFPGAN
                _, _, output = self.model.enhance(
                    img_bgr, 
                    has_aligned=False, 
                    only_center_face=False, 
                    paste_back=True
                )
                
                # Convert back to RGB
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                
                logger.info("Image enhanced with GFPGAN")
                return Image.fromarray(output)
                
            except Exception as e:
                logger.error(f"GFPGAN enhancement failed: {e}")
                logger.warning("Falling back to basic enhancement")
        
        # Fallback to basic enhancement
        logger.info("Using basic enhancement")
        enhanced = self._basic_enhance(img_array)
        return Image.fromarray(enhanced)
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Call the enhance method directly when model is called
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        return self.enhance(image)