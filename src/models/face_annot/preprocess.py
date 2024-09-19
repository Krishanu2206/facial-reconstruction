import torch
from torchvision import transforms
from abc import ABC, abstractmethod

class PreProcessTemplate(ABC):

    @abstractmethod
    def process(self, img):
        pass


class ProcessFeatures(PreProcessTemplate):
    def __init__(self):
        pass

    def process(img):
        """
        Process an image for training.

        Args:
            img: A PIL image.

        Returns:
            A tensor of size (3, 256, 256) that is ready for training.
        """
        transform = transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
            # transforms.RandomHorizontalFlip(),  
            transforms.ToTensor(),  
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(img)
    

class ProcessTarget(PreProcessTemplate):
    def __init__(self):
        pass

    def process(landmarks, img_size, new_size):
        """
        Transforms the landmark coordinates based on the transformations applied to the image.
        
        Args:
            landmarks (torch.Tensor): Tensor of shape (8,) containing the (x, y) landmark coordinates.
            img_size (tuple): Original image size (height, width).
            new_size (tuple): New image size (height, width).
        
        Returns:
            torch.Tensor: Transformed landmark coordinates.
        """
        original_height, original_width, _ = img_size
        new_height, new_width, _ = new_size
        
        scale_x = new_width / original_width
        scale_y = new_height / original_height
        transformed_landmarks = []
    
        for i in range(0, len(landmarks), 2):
            x = landmarks[i] * scale_x 
            y = landmarks[i+1] * scale_y 
           
            transformed_landmarks.extend([x, y])
        
        return torch.tensor(transformed_landmarks)  