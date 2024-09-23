import sys
import os
import torch
import unittest
import logging
from pathlib import Path

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
    )


class TestDir(unittest.TestCase):
    def test_dir(self):
        train_dir = Path("data/processed/train")
        test_dir = Path("data/processed/test")

        self.assertTrue(train_dir.exists(), msg="Train directory does not exist")
        self.assertTrue(test_dir.exists(), msg="Test directory does not exist")

        self.assertTrue((train_dir / "high_res").exists(), msg="Train high-res directory does not exist")
        self.assertTrue((train_dir / "low_res").exists(), msg="Train low-res directory does not exist")
        self.assertTrue((test_dir / "high_res").exists(), msg="Test high-res directory does not exist")
        self.assertTrue((test_dir / "low_res").exists(), msg="Test low-res directory does not exist")
        
        
if __name__ == "__main__":
    unittest.main()
