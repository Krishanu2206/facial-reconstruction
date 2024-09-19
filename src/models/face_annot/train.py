import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
import logging
from pathlib import Path


from databuilder import create_dataset
from model_builder import LandMarkModel


class init_engine:
    def __init__(self) -> None:
        self.model = LandMarkModel
        self.train_set, test_set = ...