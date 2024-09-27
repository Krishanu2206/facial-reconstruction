import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# engine.py

import torch
import os
from tqdm.auto import tqdm
from torchvision.utils import save_image

class Engine:
    def __init__(self, base_model, refinement_model, optimizer_base, optimizer_refine, scheduler_base, scheduler_refine, criterion, device='cuda'):
        self.base_model = base_model
        self.refinement_model = refinement_model
        self.optimizer_base = optimizer_base
        self.optimizer_refine = optimizer_refine
        self.scheduler_base = scheduler_base
        self.scheduler_refine = scheduler_refine
        self.criterion = criterion
        self.device = device
        self.best_loss = float('inf')
        self.best_epoch = 0

    def train(self, n_epochs, train_loader, save_dir, train_base=True, train_refine=True):
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(n_epochs):
            self.base_model.train()
            self.refinement_model.train()
            total_loss = 0
            
            for i, (low_res, high_res) in enumerate(train_loader):
                low_res = low_res.to(self.device)
                high_res = high_res.to(self.device)
                
                # Forward pass through both models
                base_output = self.base_model(low_res)
                refined_output = self.refinement_model(base_output)
                
                # Loss calculation for both models
                loss_base = self.criterion(base_output, high_res)
                loss_refined = self.criterion(refined_output, high_res)

                # Backward pass and optimization with conditionals
                if train_base:
                    self.optimizer_base.zero_grad()
                    loss_base.backward(retain_graph=True)
                    self.optimizer_base.step()

                if train_refine:
                    self.optimizer_refine.zero_grad()
                    loss_refined.backward()
                    self.optimizer_refine.step()
                
                total_loss += loss_refined.item()

                # Save sample images periodically
                if i % 10 == 0:
                    img_sample = torch.cat((low_res.data, base_output.data, refined_output.data, high_res.data), -2)
                    save_image(img_sample, os.path.join(save_dir, f"epoch_{epoch}_batch_{i}.png"), nrow=4, normalize=True)
            
            # Scheduler step for both models
            if train_base:
                self.scheduler_base.step()
            if train_refine:
                self.scheduler_refine.step()

            avg_loss = total_loss / len(train_loader)
            
            # Save best model
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_epoch = epoch
                torch.save({
                    'base_model_state_dict': self.base_model.state_dict(),
                    'refinement_model_state_dict': self.refinement_model.state_dict()
                }, os.path.join(save_dir, "best_stacked_model.pth"))
            
            tqdm.write(f"Epoch {epoch+1}/{n_epochs}, Avg Loss: {avg_loss:.4f}")
        
        tqdm.write(f"Training completed. Best model saved at epoch {self.best_epoch} with loss {self.best_loss:.4f}")
