from torch import nn
import torch
import torch.utils
from tqdm.auto import tqdm
import logging
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from abc import abstractmethod
from pathlib import Path

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
    
    def _test_model(self):
        """
        Evaluate the generator and discriminator on the test set.

        Returns:
            g_loss: The generator loss on the test set.
            d_loss: The discriminator loss on the test set.
        """
        self.g_model.eval()
        self.d_model.eval()
        running_gloss, running_dloss = 0, 0
        with torch.inference_mode():
            for low_res, high_res in tqdm(self.test_dataloader):
                low_res = low_res.to(self.device)
                real_images = high_res[:, :3, :, :].to(self.device)
                b_size = low_res.shape[0]    
                
                real_labels = torch.ones((b_size, 1, 30, 30))
                fake_labels = torch.zeros((b_size, 1, 30, 30))
                
                d_loss = self._discriminator_loss(low_res=low_res, real_images=real_images, 
                                              real_labels=real_labels, fake_labels=fake_labels) 
                
                g_loss, fake_images = self._generator_loss(low_res=low_res,
                                          real_images=real_images, real_labels=real_labels)
                
                running_dloss += d_loss
                running_gloss += g_loss
                
            running_dloss /= len(self.test_dataloader)
            running_gloss /= len(self.test_dataloader)
            return running_gloss, running_dloss
                
    def _discriminator_loss(self, low_res, real_images, real_labels, fake_labels):
        """
        Calculate the discriminator loss. This function takes in the low-resolution images, real
        high-resolution images, real labels, and fake labels. It calculates the discriminator loss
        by passing the real and fake images through the discriminator and calculating the binary
        cross-entropy loss.

        Args:
            low_res (torch.Tensor): Low-resolution images
            real_images (torch.Tensor): Real high-resolution images
            real_labels (torch.Tensor): Real labels
            fake_labels (torch.Tensor): Fake labels

        Returns:
            d_loss (torch.Tensor): The discriminator loss
        """
        real_patch = self.d_model(low_res[:, :3, :, :], real_images)
        
        with torch.no_grad():
            fake_images = self.g_model(low_res.to(self.device))
        fake_patch = self.d_model(low_res[:, :3, :, :], fake_images)
        
        d_loss_real = self.d_loss(real_patch, real_labels)
        d_loss_fake = self.d_loss(fake_patch, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        return d_loss

    def _generator_loss(self, low_res, real_images, real_labels):
        """
        Calculate the generator loss. This function takes in the low-resolution images, real
        high-resolution images, and real labels. It calculates the generator loss by passing the
        fake images through the discriminator and calculating the binary cross-entropy loss.

        Args:
            low_res (torch.Tensor): Low-resolution images
            real_images (torch.Tensor): Real high-resolution images
            real_labels (torch.Tensor): Real labels

        Returns:
            tuple: (generator loss, fake images)
        """
        fake_images = self.g_model(low_res)
        fake_patch = self.d_model(low_res[:, :3, :, :], fake_images)
        
        ## to be explored further
        adversarial_loss = self.d_loss(fake_patch, real_labels)
        content_loss = self.g_loss(fake_images, real_images)
        g_loss = adversarial_loss + self.content_weight * content_loss
        return g_loss, fake_images
        
    def _train_one_epoch(self):
        """
        Train the generator and discriminator for one epoch.

        Args:
            None

        Returns:
            tuple: (generator loss, discriminator loss)
        """
        
        per_epoch_loss_g = 0.0
        per_epoch_loss_d = 0.0
        
        for low_res, high_res in tqdm(self.train_dataloader):
            low_res = low_res.to(self.device)
            real_images = high_res[:, :3, :, :].to(self.device)
            b_size = low_res.shape[0]
            
            real_labels = torch.ones((b_size, 1, 30, 30), dtype=torch.float32) * 0.9
            fake_labels = torch.zeros((b_size, 1, 30, 30), dtype=torch.float32) * 0.1
            
            
            # train the discriminator
            self.d_optimizer.zero_grad()
            d_loss = self._discriminator_loss(low_res=low_res, real_images=real_images, 
                                              real_labels=real_labels, fake_labels=fake_labels)
            d_loss.backward()
            self.d_optimizer.step()
            
            # train the generator
            self.g_optimizer.zero_grad()
            g_loss, fake_images = self._generator_loss(low_res=low_res,
                                          real_images=real_images, real_labels=real_labels)
            g_loss.backward()
            self.g_optimizer.step()
            
            per_epoch_loss_g += g_loss.item()
            per_epoch_loss_d += d_loss.item()
            
        return per_epoch_loss_g / len(self.train_dataloader), per_epoch_loss_d / len(self.train_dataloader)
            