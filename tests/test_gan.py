import sys
import os
import torch
import unittest
import logging

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_builder import Generator, Discriminator

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
    )

class TestLoader(unittest.TestCase):
    def test_generator_forward_pass(self):
        """
        Test that the generator model runs a forward pass successfully.
        
        Specifically, this test creates a random input tensor of shape (32, 3, 256, 256) and passes it through the
        generator model. The output should be a tensor of the same shape as the input.
        """
        
        low_res_image = torch.randn((32, 3, 256, 256))  # Use torch.randn to create a random tensor with shape

        g_model = Generator()
        
        logging.info("Running generator forward pass")
        output = g_model(low_res_image)

        self.assertTrue(output.shape == (32, 3, 256, 256))

    def test_discriminator_pass(self):
        """
        Test that the discriminator model runs a forward pass successfully.
        
        Specifically, this test creates two random input tensors of shape (32, 3, 256, 256) and passes them through the
        discriminator model. The output should be a tensor of shape (32, 1, 30, 30).
        """
        low_res_image = torch.randn((32, 3, 256, 256))
        high_res_image = torch.randn((32, 3, 256, 256))

        d_model = Discriminator()
        
        logging.info("Running dsicriminator forward pass")
        output = d_model(low_res_image, high_res_image)

        self.assertTrue(output.shape == (32, 1, 30, 30))
        

if __name__ == "__main__":
    unittest.main()
