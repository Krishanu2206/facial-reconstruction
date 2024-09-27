import torch
from torch import nn
from torch.nn import functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import sys
import os
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import StepLR
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.models.constructor import UnetGenerator
from src.models.utils import load_model, save_model
from src.data import CreateDataset, ProcessFeatures, ProcessTarget
from src.models.stacked_refined.model_builder import StackedRefinementSR, RefinementNetwork
from src.models.stacked_refined.loss import PSNRSSIMLoss
from src.models.stacked_refined.engine import Engine

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_dataloader():
    train_dataset = CreateDataset(lowResImagesPath="data/processed/train/low_res",
                                  highResImagesPath="data/processed/train/high_res",
                                  feature_transform=ProcessFeatures,
                                  target_transform=ProcessTarget)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    return train_loader



device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Models
model_path = "models/saved_models/generator_best_30.pth"
generator = UnetGenerator(input_nc=3, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False)
state_dict = torch.load(model_path, map_location=device)
generator.load_state_dict(state_dict)
generator.to(device)
refinement_net = RefinementNetwork().to(device)
stacked_model = StackedRefinementSR(generator, refinement_net).to(device)

# Optimizers
optimizer_base = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=0.000005, betas=(0.5, 0.999))
optimizer_refine = optim.Adam(refinement_net.parameters(), lr=0.0001, betas=(0.5, 0.999))

# Schedulers
scheduler_base = StepLR(optimizer_base, step_size=5, gamma=0.1)
scheduler_refine = StepLR(optimizer_refine, step_size=5, gamma=0.1)

# Loss function
criterion = PSNRSSIMLoss(alpha=0.5).to(device)

train_loader = get_dataloader()

# Instantiate Engine with two models, optimizers, and schedulers
if __name__ == "__main__":
    save_dir = "./training_results"
    
    engine = Engine(
        base_model=generator,
        refinement_model=refinement_net,
        optimizer_base=optimizer_base,
        optimizer_refine=optimizer_refine,
        scheduler_base=scheduler_base,
        scheduler_refine=scheduler_refine,
        criterion=criterion,
        device=device
    )

    # Train base model first, then refinement model
    print("Training base model...")
    engine.train(n_epochs=5, train_loader=train_loader, save_dir=save_dir, train_base=True, train_refine=False)
    
    print("Training refinement model and fine-tuning base model...")
    engine.train(n_epochs=5, train_loader=train_loader, save_dir=save_dir, train_base=True, train_refine=True)