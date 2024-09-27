from torch import nn
import torch
import torch.utils
from tqdm.auto import tqdm
import logging
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from abc import abstractmethod
from pathlib import Path
from torchvision.utils import save_image
# add basic logging config
logging.basicConfig(level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
    )


class Engine:
    def __init__(self, g_model: nn.Module,
                 d_model: nn.Module,
                 g_loss: nn.Module,
                 d_loss: nn.Module,
                 g_optimizer: torch.optim,
                 d_optimizer: torch.optim,
                 train_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 content_weight: float,
                 device: torch.device,
                 epochs: int) -> None:
        
        """
        Initialize the Engine object

        Args:
            g_model (nn.Module): Generator model
            d_model (nn.Module): Discriminator model
            g_loss (nn.Module): Generator loss function
            d_loss (nn.Module): Discriminator loss function
            g_optimizer (torch.optim): Generator optimizer
            d_optimizer (torch.optim): Discriminator optimizer
            train_dataloader (DataLoader): Training data loader
            test_dataloader (DataLoader): Testing data loader
            device (torch.device): Device to use for training and inference
            epochs (int): Number of epochs to train for
        """
        self.g_model = g_model
        self.d_model = d_model
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.content_weight = content_weight
        self.device = device
        self.epochs = epochs
        
    def train(self):
        """
        Train the generator and discriminator for the specified number of epochs.

        Args:

        Returns:
        """
       
        best = float('inf')
        for epoch in range(self.epochs):
            g_loss, d_loss = self._train_one_epoch()
            # print(f"Generator Loss: {g_loss} | Discriminator Loss {d_loss}")
            logging.info(f"{epoch+1}/{self.epochs} | Generator Train Loss: {g_loss} | Discriminator Train Loss {d_loss}")
            
            if g_loss < best:
                self.save_model(model=self.g_model)
                best = g_loss
            
            if epoch % 20 == 0:
                g_loss, d_loss = self._test_model()
                logging.info(f"At epoch {epoch+1}: Generator Test Loss {g_loss} | Discriminator Test Loss {d_loss}")
        
        self.save_model(model=self.g_model, model_name="last.pt")
    
    def save_model(self, model: nn.Module, save_dir: str = "models/saved_models", model_name="best.pt"):
        """
        Saves the model state dictionary to the specified save directory.

        Args:
            model (nn.Module): Model to save
            save_dir (str, optional): Directory to save the model. Defaults to "models/saved_models".
            model_name (str, optional): Name of the saved model. Defaults to "best.pt".
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        torch.save(model.state_dict(), save_path / model_name)
    
    
    def train_epoch(self, epoch, n_epochs, dataloader, generator, optimizer_G, scheduler):
        total_loss = 0
        for i, (low_res, high_res) in enumerate(dataloader):
            low_res = low_res.to(self.device)
            high_res = high_res.to(self.device)

            # Generate high-res image
            gen_high_res = generator(low_res)

            # Losses
            loss_pixel = criterion_pixelwise(gen_high_res, high_res)
            loss_perceptual = criterion_perceptual(gen_high_res, high_res)

            # Total generator loss
            loss_G = lambda_pixel * loss_pixel + lambda_perceptual * loss_perceptual

            # Backward pass and optimize
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            total_loss += loss_G.item()

            # Print log
            if i % 10 == 0:
                print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] [G loss: {loss_G.item()}]")

            # Save sample images
            if i % 10 == 0:
                img_sample = torch.cat((low_res.data, gen_high_res.data, high_res.data), -2)
                save_image(img_sample, f"images/b_{epoch}_{i}.png", nrow=3, normalize=True)

        # Step the scheduler
        scheduler.step()

        return total_loss / len(dataloader)