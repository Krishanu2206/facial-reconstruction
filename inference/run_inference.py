import torch
from torch import nn
from torchvision import transforms
from glob import glob
import sys
import os
import random
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg (non-GUI)
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.constructor.model_builder import UnetGenerator

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_of_images = 1
image_path = Path("data/raw/zip/SCface_database/surveillance_cameras_all")
gt_path = Path('data/raw/zip/SCface_database/mugshot_frontal_original_all')


def load_images_and_gt(images_path, gt_path):
    images = glob(pathname='*.jpg', root_dir=images_path)
    images = random.sample(images, k=num_of_images) 
    gts = [img.split('_')[0] + '_frontal.jpg' for img in images]
    
    images = [Image.open(image_path / img) for img in images]
    gts = [Image.open(gt_path / img) for img in gts]
    
    return images, gts


def load_model():
    model_path = 'models/saved_models/generator_best_30.pth'
    model = UnetGenerator(input_nc=3, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def inference(model, images, gts):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    preds = []
    model.eval()
    model.to(device)
    for i in range(num_of_images):
        img = images[i]
        gt = gts[i]

        img = transform(img)
        img = img.unsqueeze(0)
        img = img.to(device)

        pred = model(img)
        pred = pred.squeeze(0)
        pred = pred.permute(1, 2, 0)
        pred = pred.cpu().detach().numpy()
        pred = (pred * 255).astype('uint8')

        gt = transforms.ToTensor()(gt)
        gt = transforms.Resize((256, 256))(gt)
        gt = gt.permute(1, 2, 0)
        gt = gt.cpu().detach().numpy()
        gt = (gt * 255).astype('uint8')

        preds.append(pred)
    return preds
    

def plot_images(images, preds, gts):
    fig, axs = plt.subplots(num_of_images, 3, figsize=(15, 5*num_of_images))
    
    if num_of_images == 1:
        axs = [axs]  # Make axs iterable when there's only one row
    
    for i in range(num_of_images):
        axs[i][0].imshow(images[i])
        axs[i][0].set_title('Input Image')
        axs[i][0].axis('off')
        
        axs[i][1].imshow(preds[i])
        axs[i][1].set_title('Predicted Image')
        axs[i][1].axis('off')
        
        axs[i][2].imshow(gts[i])
        axs[i][2].set_title('Ground Truth')
        axs[i][2].axis('off')

    plt.tight_layout()
    plt.savefig('inference/inference.png')
    plt.close()  # Close the figure to free up memory

if __name__ == '__main__':
    images, gts = load_images_and_gt(image_path, gt_path)
    model = load_model()
    preds = inference(model, images, gts)
    plot_images(images, preds, gts)
    print("Inference complete. Image saved as 'inference/inference.png'")