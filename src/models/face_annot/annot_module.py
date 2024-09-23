import cv2
import mediapipe as mp
import numpy as np
import os

mp_face_mesh = mp.solutions.face_mesh

FACIAL_LANDMARKS = {
    'left_eye': [33, 133],
    'right_eye': [362, 263],
    'nose_tip': [1],
    'mouth': [13, 14]
}

def generate_heatmap(image, landmark, size=(256, 256), sigma=15):
    heatmap = np.zeros(size, dtype=np.float32)
    for lm in landmark:
        x = int(lm[0] * size[1])
        y = int(lm[1] * size[0])
        if x >= size[1] or y >= size[0]:
            continue
        heatmap = cv2.circle(heatmap, (x, y), sigma, (1), thickness=-1)
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma)
    heatmap = np.clip(heatmap, 0, 1)
    return heatmap

def process_image(image_path, output_dir, face_mesh):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))  # Resize to 256x256
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(image_rgb)

    heatmaps = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]

            left_eye_heatmap = generate_heatmap(image, [landmarks[i] for i in FACIAL_LANDMARKS['left_eye']])
            right_eye_heatmap = generate_heatmap(image, [landmarks[i] for i in FACIAL_LANDMARKS['right_eye']])
            nose_heatmap = generate_heatmap(image, [landmarks[i] for i in FACIAL_LANDMARKS['nose_tip']])
            mouth_heatmap = generate_heatmap(image, [landmarks[i] for i in FACIAL_LANDMARKS['mouth']])

            heatmaps.extend([left_eye_heatmap, right_eye_heatmap, nose_heatmap, mouth_heatmap])

        heatmaps_stack = np.stack(heatmaps, axis=0) 

        image_normalized = image / 255.0
        rgb_and_heatmaps = np.concatenate([image_normalized.transpose(2, 0, 1), heatmaps_stack], axis=0) 

        file_name = os.path.basename(image_path).split('.')[0]
        output_path = os.path.join(output_dir, f'{file_name}_rgb_heatmaps.npy')
        np.save(output_path, rgb_and_heatmaps)
        print(f"Processed and saved: {output_path}")
    else:
        print(f"No face landmarks detected in: {image_path}")

def process_directory(input_dir, output_dir):
    with mp_face_mesh.FaceMesh(
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5) as face_mesh:

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for file_name in os.listdir(input_dir):
            file_path = os.path.join(input_dir, file_name)
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                process_image(file_path, output_dir, face_mesh)


if __name__ == "__main__":
    input_directory = 'E:/computer vision/facial_reconstruction_project/data/raw/high_quality_images/mugshot_frontal_original_all'  
    output_directory = 'E:/computer vision/facial_reconstruction_project/data/rgb&heatMap/highres'  
    process_directory(input_directory, output_directory)
