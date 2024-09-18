import PIL.Image
from torchvision import transforms
from abc import ABC, abstractmethod

class PreProcessTemplate(ABC):

    @abstractmethod
    def process(img):
        pass
    
    
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
            transforms.Resize((286, 286)),
            transforms.RandomRotation((-15, 15)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomPerspective(0.5),
            transforms.RandomCrop((256, 256)),
            transforms.ToTensor(),
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
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
        ])
        
        return target_transform(img)
    