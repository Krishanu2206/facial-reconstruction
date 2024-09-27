import torch
from torch import nn
from torch.utils.data import DataLoader
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.make_dataset import CreateDataset
from models.constructor.model_builder import UnetGenerator, Discriminator
from src.data.preprocess import ProcessFeatures, ProcessTarget
from models.constructor.engine import Engine


# add logging basic config
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using Device: {device}")

BATCH_SIZE = 2
EPOCHS = 10


def init_model():
    g_model = generator = UnetGenerator(input_nc=3, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False)
    d_model = Discriminator().to(device)

    d_loss = nn.BCELoss()
    g_loss = nn.L1Loss()

    g_optimizer = torch.optim.Adam(g_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(d_model.parameters(), lr=0.00001, betas=(0.5, 0.999))

    try:
        train_dataset = CreateDataset(lowResImagesPath="data/processed/train/low_res",
                                    highResImagesPath="data/processed/train/high_res",
                                    feature_transform=ProcessFeatures,
                                    target_transform=ProcessTarget)

        test_dataset = CreateDataset(lowResImagesPath="data/processed/test/low_res",
                                    highResImagesPath="data/processed/test/high_res",
                                    feature_transform=ProcessFeatures,
                                    target_transform=ProcessTarget)

        if len(train_dataset) == 0 or len(test_dataset) == 0:
            raise ValueError("One or both datasets are empty")

        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        

    except Exception as e:
        logging.error(f"Error creating datasets or dataloaders: {str(e)}")
        raise

    engine = Engine(g_model=g_model, d_model=d_model,
                g_loss=g_loss, d_loss=d_loss,
                g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                train_dataloader=train_loader, test_dataloader=test_loader,
                content_weight=0.001,
                device=device, epochs=EPOCHS)

    return engine

if __name__ == "__main__":
    engine = init_model()
    engine.train()