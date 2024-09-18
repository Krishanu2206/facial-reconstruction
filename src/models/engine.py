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
                 device: torch.device,
                 epochs: int) -> None:
        
        self.g_model = g_model
        self.d_model = d_model
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.epochs = epochs
        
    def train(self):
        best = float('inf')
        for epoch in range(self.epochs):
            g_loss, d_loss = self._train_one_epoch()
            # print(f"Generator Loss: {g_loss} | Discriminator Loss {d_loss}")
            logging.info(f"{epoch+1}/{self.epochs} | Generator Train Loss: {g_loss} | Discriminator Train Loss {d_loss}")
            
            if g_loss < best:
                self.save_model(self.g_model)
                best = g_loss
            
            if epoch % 20 == 0:
                g_loss, d_loss = self._test_model()
                logging.info(f"At epoch {epoch+1}: Generator Test Loss {g_loss} | Discriminator Test Loss {d_loss}")
        
        self.save_model(self.g_model, model_name="last.pt")
    
    @abstractmethod
    def save_model(model: nn.Module, save_dir: str = "models/saved_models", model_name="best.pt"):
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        torch.save(model.state_dict(), save_path / model_name)
    
    def _test_model(self):
        self.g_model.eval()
        self.d_model.eval()
        with torch.inference_mode():
            for low_res, high_res in tqdm(self.test_dataloader):
                low_res = low_res.to(self.device)
                real_images = high_res.to(self.device)
                b_size = low_res.shape[0]    
                
                real_labels = torch.ones((b_size, 1, 30, 30))
                fake_labels = torch.zeros((b_size, 1, 30, 30))
                
                d_loss = self._discriminator_loss(low_res=low_res, real_images=real_images, 
                                              real_labels=real_labels, fake_labels=fake_labels) 
                
                g_loss = self._generator_loss(low_res=low_res,
                                          real_images=real_images, real_labels=real_labels)
                
                return g_loss, d_loss
                
    def _discriminator_loss(self, low_res, real_images, real_labels, fake_labels):
        real_patch = self.d_model(low_res, real_images)
        
        fake_images = self.g_model(low_res.to(self.device))
        fake_patch = self.d_model(low_res, fake_images)
        
        d_loss_real = self.d_loss(real_patch, real_labels)
        d_loss_fake = self.d_loss(fake_patch, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        
        return d_loss

    def _generator_loss(self, low_res, real_images, real_labels):
        fake_images = self.g_model(low_res)
        fake_patch = self.d_model(low_res, fake_images)
        
        ## to be explored further
        fake_gan_loss = self.d_loss(fake_patch, real_labels)
        real_gan_loss = self.g_loss(fake_images, real_images)
        g_loss = fake_gan_loss + real_gan_loss
        return g_loss
        
    def _train_one_epoch(self):
        per_epoch_loss_g = 0.0
        per_epoch_loss_d = 0.0
        
        for low_res, high_res in tqdm(self.train_dataloader):
            low_res = low_res.to(self.device)
            real_images = high_res.to(self.device)
            b_size = low_res.shape[0]
            
            real_labels = torch.ones((b_size, 1, 30, 30))
            fake_labels = torch.zeros((b_size, 1, 30, 30))
            
            # train the discriminator
            d_loss = self._discriminator_loss(low_res=low_res, real_images=real_images, 
                                              real_labels=real_labels, fake_labels=fake_labels)
            per_epoch_loss_d += d_loss
            
            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()
            
            # train the generator
            g_loss = self._generator_loss(low_res=low_res,
                                          real_images=real_images, real_labels=real_labels)
            per_epoch_loss_g += g_loss
            
            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()
            
        return per_epoch_loss_g / len(self.train_dataloader), per_epoch_loss_d / len(self.train_dataloader)