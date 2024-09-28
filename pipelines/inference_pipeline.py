import cv2
import torch
import numpy as np
from torch import nn
from pathlib import Path
from ultralytics import YOLO
import torch.nn.functional as F
from torchvision import transforms
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import StackedRefinementSR, UnetGenerator, RefinementNetwork

def load_stacked_model(model_path):
    generator = UnetGenerator(input_nc=3, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False)
    refinement_net = RefinementNetwork()
    stacked_model = StackedRefinementSR(generator, refinement_net)
    stacked_model.load_state_dict(torch.load(model_path))
    stacked_model.eval()
    return stacked_model

def load_yolo_model(model_path):
    return YOLO(model_path)

def preprocess_image(image, device):
    # Define the transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations
    image = transform(image).unsqueeze(0).to(device)
    return image

def postprocess_image(image):
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image.device)
    image = image * std + mean
    
    # Convert to numpy and adjust range
    image = image.squeeze().permute(1, 2, 0).cpu().numpy()
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def detect_faces(image, yolo_model):
    results = yolo_model(image, classes=[0])  # 0 is typically the class for persons
    faces = []
    for det in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, det[:4])
        face = image[y1:y2, x1:x2]
        faces.append((face, (x1, y1, x2, y2)))
    return faces

def enhance_face(face, stacked_model, device):
    face = preprocess_image(face, device)
    with torch.no_grad():
        enhanced_face = stacked_model(face)
    return postprocess_image(enhanced_face)

def process_video(video_path, output_dir, stacked_model, yolo_model, device):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame, yolo_model)

        for idx, (face, (x1, y1, x2, y2)) in enumerate(faces):
            enhanced_face = enhance_face(face, stacked_model, device)
            
            # Resize enhanced face to original size
            enhanced_face = cv2.resize(enhanced_face, (x2 - x1, y2 - y1))
            
            # Save enhanced face
            output_path = Path(output_dir) / f"frame_{frame_count}_face_{idx}.jpg"
            cv2.imwrite(str(output_path), enhanced_face)

        frame_count += 1

    cap.release()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    stacked_model_path = "models/saved_models/stacked_model.pth"
    yolo_model_path = "models/saved_models/yolov8n.pt"
    video_path = "data/external/video.mp4"
    output_dir = "data/output"

    stacked_model = load_stacked_model(stacked_model_path).to(device)
    yolo_model = load_yolo_model(yolo_model_path)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    process_video(video_path, output_dir, stacked_model, yolo_model, device)

if __name__ == "__main__":
    main()