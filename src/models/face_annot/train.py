import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm.auto import tqdm
import logging
from pathlib import Path


from databuilder import create_dataset
from model_builder import LandMarkModel


BATCH_SIZE = 32
EPOCHS = 3


class init_engine:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = create_dataset()
        train_set, test_set = random_split(dataset=dataset,  lengths=[int(0.8*len(dataset)), int(0.2*len(dataset))])
        
        self.train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
        
        self.epochs = EPOCHS
        self.model = LandMarkModel().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.001)
        
    def train(self):
        for epoch in range(self.epochs):
            train_loss = self._train_one_epoch()
            test_loss = self._test()
            
            logging.info(f"Epoch: {epoch} | Train Loss: {train_loss} | Test Loss: {test_loss}")
        
    def save_model(self):
        torch.save(self.model.state_dict(), "model.pt")    
    
    def _train_one_epoch(self):
        per_epoch_loss = 0.0
        for img, annot in tqdm(self.train_loader):
            img = img.to(self.device)
            annot = annot.to(self.device)
            
            pred = self.model(img)
            loss = self.criterion(pred, annot)
            per_epoch_loss += loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return per_epoch_loss / len(self.train_loader)
    
    def _test(self):
        test_loss = 0.0
        
        self.model.eval()
        with torch.inference_mode():
            for img, annot in tqdm(self.test_loader):
                img = img.to(self.device)
                annot = annot.to(self.device)
                
                pred = self.model(img)
                loss = self.criterion(pred, annot)
                test_loss += loss
                
        return test_loss / len(self.test_loader)
    
    
if __name__ == "__main__":
    engine = init_engine()
    engine.train()
    engine.save_model()