# ARTEMIS STABLE SCRIPT - With Confidence Check & Smoothing
import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
from collections import deque, Counter
import json
import os

# --- CONSTANTS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'custom_yolo', 'best.pt')
ONNX_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'gru_v1', 'model.onnx')
ID_TO_GLOSS_PATH = os.path.join(PROJECT_ROOT, 'models', 'id_to_gloss.json')

# --- MODEL INITIALIZATION ---
print("Loading all models...")
onnx_session = ort.InferenceSession(ONNX_MODEL_PATH)
input_details = onnx_session.get_inputs()[0]
input_name = input_details.name
emotion_classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
with open(ID_TO_GLOSS_PATH, 'r') as f:
    id_to_gloss = json.load(f)
print("✅ All models loaded successfully.")

def extract_and_normalize_keypoints(results, frame_shape):
    h, w, _ = frame_shape
    if results.pose_landmarks:
        nose_landmark = results.pose_landmarks.landmark[0]
        nose_x, nose_y = nose_landmark.x * w, nose_landmark.y * h
        pose = np.array([((res.x * w) - nose_x, (res.y * h) - nose_y) for res in results.pose_landmarks.landmark]).flatten()
        face = np.array([((res.x * w) - nose_x, (res.y * h) - nose_y) for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*2)
        lh = np.array([((res.x * w) - nose_x, (res.y * h) - nose_y) for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*2)
        rh = np.array([((res.x * w) - nose_x, (res.y * h) - nose_y) for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*2)
        face = face[:70]
        return np.concatenate([pose, face, lh, rh])
    else:
        return np.zeros(220)

# --- MAIN APPLICATION LOOP ---
cap = cv2.VideoCapture(1)
print("\n🚀 Starting ARTEMIS Stable Demo... Press 'q' to quit.")

sequence = deque(maxlen=20)
prediction_history = deque(maxlen=10) # For smoothing
current_emotion, smoothed_sign = "...", "..."

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    mp_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_results = holistic.process(mp_image_rgb)

    if mp_results.face_landmarks:
        landmarks = mp_results.face_landmarks.landmark
        x_min, y_min = int(min([lm.x for lm in landmarks]) * frame.shape[1]), int(min([lm.y for lm in landmarks]) * frame.shape[0])
        x_max, y_max = int(max([lm.x for lm in landmarks]) * frame.shape[1]), int(max([lm.y for lm in landmarks]) * frame.shape[0])
        padding = 20
        x1, y1 = max(0, x_min - padding), max(0, y_min - padding)
        x2, y2 = min(frame.shape[1], x_max + padding), min(frame.shape[0], y_max + padding)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size > 0:
            try:
                pil_image = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                emotion_results = emotion_classifier(pil_image)
                current_emotion = emotion_results[0]['label'].capitalize()
            except Exception:
                current_emotion = "N/A"

    keypoints = extract_and_normalize_keypoints(mp_results, frame.shape)
    sequence.append(keypoints)

    current_sign = "..." # Default to nothing
    if len(sequence) == 20:
        input_tensor = np.expand_dims(np.array(sequence), axis=0)
        onnx_input = {input_name: input_tensor.astype(np.float32)}
        onnx_output = onnx_session.run(None, onnx_input)[0]

        # The Confidence Check
        confidence = np.max(np.exp(onnx_output) / np.sum(np.exp(onnx_output)))
        if confidence > 0.6: # Use a 60% confidence threshold
            prediction_index = np.argmax(onnx_output)
            current_sign = id_to_gloss[str(prediction_index)]

        prediction_history.append(current_sign)

        # The Voting System
        if len(prediction_history) == 10:
            most_common = Counter(prediction_history).most_common(1)[0]
            if most_common[1] >= 5: # If the sign is stable
                smoothed_sign = most_common[0]

    # --- VISUALIZATION ---
    mp_drawing.draw_landmarks(frame, mp_results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, None, mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(frame, mp_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(frame, mp_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(frame, mp_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

    cv2.rectangle(frame, (0, 0), (350, 80), (0, 0, 0), -1)
    # Display the final, smoothed sign
    cv2.putText(frame, f'Sign: {smoothed_sign}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Emotion: {current_emotion}', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('ARTEMIS Stable Demo', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print("✅ Demo finished cleanly.")


