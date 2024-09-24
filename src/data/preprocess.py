import PIL.Image
import torch
from torchvision import transforms
from abc import ABC, abstractmethod

class PreProcessTemplate(ABC):

    @abstractmethod
    def process(img):
        pass
    
    
class NumpyToTensor(object):
    def __call__(self, numpy_array):
        # Convert numpy array to PyTorch tensor
        tensor = torch.from_numpy(numpy_array)
        # Ensure the tensor is in the format (C, H, W)
        tensor /= 255
        
        return tensor
    
class ProcessFeatures(PreProcessTemplate):

    def process(img):
        """
        Process an image for training.
        
        Args:
            img: A PIL image.
            
        Returns:
            A tensor of size (3, 256, 256) that is ready for training.
        """
        transformation = transforms.Compose([
            NumpyToTensor(),
            transforms.Resize((286, 286)),
            transforms.RandomRotation((-15, 15)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomPerspective(0.5),
            transforms.RandomCrop((256, 256)),
        ])
        return transformation(img)
    

class ProcessTarget(PreProcessTemplate):

    def process(img):
        """
        Process an image for target.
        
        Args:
            img: A PIL image.
            
        Returns:
            A tensor of size (3, 256, 256) that is ready for training.
        """
        target_transform = transforms.Compose([
            NumpyToTensor(),
            transforms.Resize((256, 256)),
        ])
        
        return target_transform(img)
    
    
if __name__ == "__main__":
    import numpy as np

    target_transform = transforms.Compose([
            NumpyToTensor(),
            transforms.Resize((256, 256)),
        ])
    
    numpy_array = np.random.rand(7, 512, 512)  # Example 7-channel image
    result = target_transform(numpy_array)

    print("Input shape:", numpy_array.shape)
    print("Output shape:", result.shape)