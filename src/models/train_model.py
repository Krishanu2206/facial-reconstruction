import torch
from torch import nn
from torch.utils.data import DataLoader
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.make_dataset import CreateDataset
from models.model_builder import Generator, Discriminator
from data.preprocess import ProcessFeatures, ProcessTarget
from models.engine import Engine


# add logging basic config
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using Device: {device}")

BATCH_SIZE = 32
EPOCHS = 100


def init_model():
    """
    Initialize the Engine object with the appropriate models, loss functions, optimizers, 
    data loaders, device, and epochs.

    Returns:
        Engine: The Engine object with all the parameters set.
    """
    g_model = Generator(c_dim=7).to(device)
    d_model = Discriminator().to(device)

    d_loss = nn.BCELoss()
    g_loss = nn.L1Loss()

    g_optimizer = torch.optim.Adam(g_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(d_model.parameters(), lr=0.0002, betas=(0.5, 0.999))



    train_dataset = CreateDataset(lowResImagesPath="data/processed/train/low_res",
                                highResImagesPath="data/processed/train/high_res",
                                feature_transform=ProcessFeatures,
                                target_transform=ProcessTarget)

    test_dataset = CreateDataset(lowResImagesPath="data/processed/test/low_res",
                                highResImagesPath="data/processed/test/high_res",
                                feature_transform=ProcessFeatures,
                                target_transform=ProcessTarget)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    engine = Engine(g_model=g_model, d_model=d_model,
                g_loss=g_loss, d_loss=d_loss,
                g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                train_dataloader=train_loader, test_dataloader=test_loader,
                device=device, epochs=EPOCHS)

    return engine

if __name__ == "__main__":
    engine = init_model()
    engine.train()