import os
import sys
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../backend'))

from model import FacialEnhancementModel

class TestFacialEnhancementModel(unittest.TestCase):
    
    def setUp(self):
        # Create a mock model file
        os.makedirs('test_models', exist_ok=True)
        open('test_models/test_model.pth', 'a').close()
        
        # Create instance of the model
        self.model = FacialEnhancementModel('test_models/test_model.pth')
    
    def tearDown(self):
        # Clean up
        import shutil
        if os.path.exists('test_models'):
            shutil.rmtree('test_models')
    
    def test_initialization(self):
        """Test that model initializes correctly"""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.model.model)
    
    def test_preprocess(self):
        """Test preprocessing function"""
        # Create a test image
        img = Image.new('RGB', (100, 100), color='red')
        
        # Preprocess
        tensor = self.model.preprocess(img)
        
        # Check output shape and type
        self.assertEqual(tensor.shape[0], 1)  # Batch dimension
        self.assertEqual(tensor.shape[1], 3)  # RGB channels
    
    def test_postprocess(self):
        """Test postprocessing function"""
        # Create a dummy tensor
        tensor = np.random.rand(3, 512, 512).astype(np.float32)
        tensor = torch.from_numpy(tensor)
        
        # Postprocess
        img = self.model.postprocess(tensor)
        
        # Check output type and dimensions
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (512, 512))
    
    def test_enhance(self):
        """Test end-to-end enhancement"""
        # Create a test image
        img = Image.new('RGB', (100, 100), color='red')
        
        # Enhance
        enhanced = self.model.enhance(img)
        
        # Check output
        self.assertIsInstance(enhanced, Image.Image)
        self.assertEqual(enhanced.size, (100, 100))

if __name__ == '__main__':
    import torch
    unittest.main()
