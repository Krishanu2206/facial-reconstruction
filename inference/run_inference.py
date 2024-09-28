import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import os
import argparse
from yolov8_face import YOLOv8_face  # Assuming the YOLOv8_face class is in a file named yolov8_face.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your GAN model components
from src.models import UnetGenerator, RefinementNetwork, StackedRefinementSR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_stacked_model(model_path):
    generator = UnetGenerator(input_nc=3, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False)
    refinement_net = RefinementNetwork()
    stacked_model = StackedRefinementSR(generator, refinement_net)
    stacked_model.load_state_dict(torch.load(model_path), map_location=device)
    stacked_model.eval()
    return stacked_model

def preprocess_for_gan(image):
    # Convert to RGB if necessary
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to 256x256
    image = cv2.resize(image, (256, 256))
    
    # Convert to PyTorch tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

def denormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)

def process_video(video_path, face_model, gan_model, output_dir):
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        boxes, scores, classids, kpts = face_model.detect(frame)
        
        for i, box in enumerate(boxes):
            x, y, w, h = box.astype(int)
            face = frame[y:y+h, x:x+w]
            
            # Preprocess face for GAN
            face_tensor = preprocess_for_gan(face)
            
            # Pass through GAN
            with torch.no_grad():
                enhanced_face_tensor = gan_model(face_tensor)
            
            # Denormalize and convert back to numpy array
            enhanced_face = denormalize(enhanced_face_tensor[0]).permute(1, 2, 0).numpy()
            enhanced_face = (enhanced_face * 255).astype(np.uint8)
            enhanced_face = cv2.cvtColor(enhanced_face, cv2.COLOR_RGB2BGR)
            
            # Save enhanced face
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, f'enhanced_face_{frame_count}_{i}.jpg'), enhanced_face)
        
        # Draw bounding boxes on the frame
        frame_with_detections = face_model.draw_detections(frame, boxes, scores, kpts)
        
        # Display the frame
        cv2.imshow('Face Detection', frame_with_detections)
        
        frame_count += 1
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videopath', type=str, default='data/external/cctv1.mp4', help="video path")
    parser.add_argument('--yolo_modelpath', type=str, default='models/saved_models/yolov8n-face.onnx', help="YOLO onnx filepath")
    parser.add_argument('--gan_modelpath', type=str, default='models/saved_models/gan_model.pth', help="GAN model filepath")
    parser.add_argument('--output_dir', type=str, default='data/output', help="Output directory for enhanced faces")
    parser.add_argument('--confThreshold', default=0.45, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    args = parser.parse_args()

    # Initialize YOLOv8_face object detector
    face_model = YOLOv8_face(args.yolo_modelpath, conf_thres=args.confThreshold, iou_thres=args.nmsThreshold)
    
    # Load GAN model
    gan_model = load_stacked_model(args.gan_modelpath)
    
    # Process video
    process_video(args.videopath, face_model, gan_model, args.output_dir)

if __name__ == "__main__":
    main()