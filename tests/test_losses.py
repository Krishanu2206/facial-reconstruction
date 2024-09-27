## tests the generator and discriminator losses
import sys
import os
import torch
import unittest
import logging

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.constructor.model_builder import Generator, Discriminator

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
    )

class TestLosses(unittest.TestCase):
    def test_generator_loss(self):
        ############
        b_size = 2
        ############
        g_model = Generator(c_dim=7)
        d_model = Discriminator(c_dim=3)
        
        real_labels = torch.ones((b_size, 1, 30, 30), dtype=torch.float32)
        fake_labels = torch.zeros((b_size, 1, 30, 30), dtype=torch.float32)
        low_res = torch.randn(b_size, 3, 30, 30)
        
        d_loss = self._discriminator_loss(low_res=low_res, real_images=real_images, 
                                    real_labels=real_labels, fake_labels=fake_labels)
            
        

    def test_discriminator_loss(self):
        pass
    