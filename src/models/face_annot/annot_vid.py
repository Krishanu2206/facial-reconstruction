import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Indices for eyes, nose, and mouth landmarks based on Mediapipe's Face Mesh model
# You can fine-tune these indices based on your use case and facial feature interest
FACIAL_LANDMARKS = {
    'left_eye': [33, 133],  # Approximate indices for left eye
    'right_eye': [362, 263],  # Approximate indices for right eye
    'nose_tip': [1],  # Approximate index for the tip of the nose
    'mouth': [13, 14]  # Approximate indices for the mouth
}

# Setup video capture (0 for webcam, or provide video file path)
cap = cv2.VideoCapture(0)

# Function to generate heatmap for specific landmark positions
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

# Initialize FaceMesh
with mp_face_mesh.FaceMesh(
    max_num_faces=1,  # Detect only one face
    refine_landmarks=True,  # Refine landmarks around eyes and lips
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to 256x256 if needed (matching your image dimensions)
        frame = cv2.resize(frame, (256, 256))

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect face landmarks
        results = face_mesh.process(frame_rgb)

        # List to store the heatmaps for eyes, nose, and mouth
        heatmaps = []

        # If landmarks are found, extract relevant points
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Convert face landmarks to a list of tuples (x, y) in normalized coordinates
                landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]

                # Generate heatmaps for each facial feature
                left_eye_landmarks = [landmarks[i] for i in FACIAL_LANDMARKS['left_eye']]
                right_eye_landmarks = [landmarks[i] for i in FACIAL_LANDMARKS['right_eye']]
                nose_landmarks = [landmarks[i] for i in FACIAL_LANDMARKS['nose_tip']]
                mouth_landmarks = [landmarks[i] for i in FACIAL_LANDMARKS['mouth']]

                left_eye_heatmap = generate_heatmap(frame, left_eye_landmarks)
                right_eye_heatmap = generate_heatmap(frame, right_eye_landmarks)
                nose_heatmap = generate_heatmap(frame, nose_landmarks)
                mouth_heatmap = generate_heatmap(frame, mouth_landmarks)

                # Add heatmaps to the list
                heatmaps.extend([left_eye_heatmap, right_eye_heatmap, nose_heatmap, mouth_heatmap])

        # Stack the heatmaps into a single numpy array
        if heatmaps:
            heatmaps_stack = np.stack(heatmaps, axis=0)  # Shape: (4, 256, 256)
            # Concatenate the heatmaps with the original RGB image
            frame_normalized = frame / 255.0  # Normalize RGB image to [0, 1]
            rgb_and_heatmaps = np.concatenate([frame_normalized.transpose(2, 0, 1), heatmaps_stack], axis=0)  # Shape: (7, 256, 256)
            print(rgb_and_heatmaps.shape)
            # Visualize the original image and the concatenated result
            cv2.imshow('Original Frame', frame)
            cv2.imshow('Heatmaps', np.sum(heatmaps_stack, axis=0))  # Summing heatmaps for visualization
            
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
