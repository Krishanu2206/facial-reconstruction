import torch
import sys
import os
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_builder import Generator
from data.preprocess import ProcessFeatures
from face_annot.annot_module import process_single_image

def load_model():
    model = Generator()
    model.load_state_dict(torch.load("models/saved_models/best.pt"))
    return model


img_path = "data/raw/cctv_footage/surveillance_cameras_all/001_cam1_2.jpg"
img_processed = process_single_image(img_path)
img_processed = ProcessFeatures.process(img_processed)


reconstructor = load_model()
reconstructor.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
reconstructor.to(device)

img_processed = img_processed.to(device)

with torch.inference_mode():
    reconstructed_img = reconstructor(img_processed.unsqueeze(0))

reconstructed_img = (reconstructed_img[0].permute(1, 2, 0).cpu().numpy() + 1) / 2
print(reconstructed_img.shape)

plt.imshow(reconstructed_img)
plt.axis(False)
plt.savefig("reports/figures/reconstructed_image.png")
