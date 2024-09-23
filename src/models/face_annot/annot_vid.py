import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

FACIAL_LANDMARKS = {
    'left_eye': [33, 133], 
    'right_eye': [362, 263], 
    'nose_tip': [1], 
    'mouth': [13, 14]  
}

cap = cv2.VideoCapture(0)

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


with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (256, 256))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(frame_rgb)

        heatmaps = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]

                left_eye_landmarks = [landmarks[i] for i in FACIAL_LANDMARKS['left_eye']]
                right_eye_landmarks = [landmarks[i] for i in FACIAL_LANDMARKS['right_eye']]
                nose_landmarks = [landmarks[i] for i in FACIAL_LANDMARKS['nose_tip']]
                mouth_landmarks = [landmarks[i] for i in FACIAL_LANDMARKS['mouth']]

                left_eye_heatmap = generate_heatmap(frame, left_eye_landmarks)
                right_eye_heatmap = generate_heatmap(frame, right_eye_landmarks)
                nose_heatmap = generate_heatmap(frame, nose_landmarks)
                mouth_heatmap = generate_heatmap(frame, mouth_landmarks)

                heatmaps.extend([left_eye_heatmap, right_eye_heatmap, nose_heatmap, mouth_heatmap])

        if heatmaps:
            heatmaps_stack = np.stack(heatmaps, axis=0) 
            frame_normalized = frame / 255.0  
            rgb_and_heatmaps = np.concatenate([frame_normalized.transpose(2, 0, 1), heatmaps_stack], axis=0)  # Shape: (7, 256, 256)
            
            cv2.imshow('Original Frame', frame)
            cv2.imshow('Heatmaps', np.sum(heatmaps_stack, axis=0))  
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
