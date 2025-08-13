# ARTEMIS FINAL SCRIPT - With Correct Hand Landmark Integration
import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
from collections import deque
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
yolo_model = YOLO(YOLO_MODEL_PATH)
onnx_session = ort.InferenceSession(ONNX_MODEL_PATH)
input_details = onnx_session.get_inputs()[0]
input_name = input_details.name
EXPECTED_FEATURES = input_details.shape[2]
emotion_classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
with open(ID_TO_GLOSS_PATH, 'r') as f:
    id_to_gloss = json.load(f)
print("✅ All models loaded successfully.")

# --- KEYPOINT EXTRACTION ---
def extract_keypoints(results):
    # NEW: We now extract from pose, face, and BOTH hands.
    pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*2)

    # We now assume the 220-D vector includes hands, so we take fewer face landmarks.
    # This is a common structure: all pose, all hands, and a subset of face landmarks.
    face = np.array([[res.x, res.y] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*2)
    face = face[:70] # Take 35 landmarks (70 coords) from the face.

    lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*2)
    rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*2)

    # Concatenate all features: Pose (66) + Face (70) + Left Hand (42) + Right Hand (42) = 220
    combined = np.concatenate([pose, face, lh, rh])
    return combined

# --- MAIN APPLICATION LOOP ---
cap = cv2.VideoCapture(1)
print("\n🚀 Starting ARTEMIS Final Demo... Press 'q' to quit.")

sequence = deque(maxlen=20)
current_emotion = "..."
current_sign = "..."

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    # We can run emotion detection on a crop for better performance
    yolo_results = yolo_model.predict(frame, verbose=False)

    if yolo_results[0].boxes:
        box = yolo_results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # --- Emotion Recognition on the face crop ---
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size > 0:
            try:
                pil_image = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                emotion_results = emotion_classifier(pil_image)
                current_emotion = emotion_results[0]['label'].capitalize()
            except Exception:
                current_emotion = "N/A"

        # --- Sign Recognition ---
        mp_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_results = holistic.process(mp_image_rgb)
        keypoints = extract_keypoints(mp_results)
        sequence.append(keypoints)

        if len(sequence) == 20:
            np_sequence = np.array(sequence)
            input_tensor = np.expand_dims(np_sequence, axis=0)
            onnx_input = {input_name: input_tensor.astype(np.float32)}
            onnx_output = onnx_session.run(None, onnx_input)
            current_sign = id_to_gloss[str(np.argmax(onnx_output[0]))]

        # --- VISUALIZATION (WITH HANDS) ---
        mp_drawing.draw_landmarks(frame, mp_results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, None, mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
        mp_drawing.draw_landmarks(frame, mp_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
        # NEW: Draw left and right hand landmarks
        mp_drawing.draw_landmarks(frame, mp_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
        mp_drawing.draw_landmarks(frame, mp_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

    # --- Display text and indicators ---
    if len(sequence) < 20:
        cv2.putText(frame, 'COLLECTING FRAMES', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, frame.shape[0] - 40), (250, frame.shape[0]), (0,0,0), -1)
    cv2.putText(frame, f'Sign: {current_sign}', (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Emotion: {current_emotion}', (130, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('ARTEMIS Final Demo', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print("✅ Demo finished cleanly.")
