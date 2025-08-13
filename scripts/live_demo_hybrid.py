# FINAL SCRIPT - TensorFlow-FREE Edition
import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
from ultralytics import YOLO
from fer import FER
import json
import os

# --- 1. SETUP & CONSTANTS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'custom_yolo', 'best.pt')
ONNX_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'gru_v1', 'model.onnx')
ID_TO_GLOSS_PATH = os.path.join(PROJECT_ROOT, 'models', 'id_to_gloss.json')

# --- 2. INITIALIZE MODELS ---
print("Loading models...")
yolo_model = YOLO(YOLO_MODEL_PATH)
onnx_session = ort.InferenceSession(ONNX_MODEL_PATH)
input_details = onnx_session.get_inputs()[0]
input_name = input_details.name
EXPECTED_FEATURES = input_details.shape[2]

# Initialize FER without TensorFlow
emotion_detector = FER(mtcnn=False)

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
with open(ID_TO_GLOSS_PATH, 'r') as f:
    id_to_gloss = json.load(f)
print("✅ All models loaded successfully.")

def extract_keypoints(results):
    pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*2)
    face = np.array([[res.x, res.y] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*2)
    face = face[:140]
    combined = np.concatenate([pose, face])
    if len(combined) < EXPECTED_FEATURES:
        padding = np.zeros(EXPECTED_FEATURES - len(combined))
        combined = np.concatenate([combined, padding])
    return combined.reshape(1, EXPECTED_FEATURES)

# --- 3. MAIN APPLICATION LOOP ---
cap = cv2.VideoCapture(0)
print("\n🚀 Starting ARTEMIS Demo... Press 'q' to quit.")
current_emotion = "..."
current_sign = "..."

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    # --- Emotion Recognition ---
    try:
        emotion_result = emotion_detector.detect_emotions(frame)
        if emotion_result:
            top_emotion = max(emotion_result[0]['emotions'], key=emotion_result[0]['emotions'].get)
            current_emotion = top_emotion.capitalize()
        else:
            current_emotion = "N/A"
    except Exception:
        current_emotion = "N/A"

    # --- Sign Recognition ---
    yolo_results = yolo_model.predict(frame, verbose=False)
    if yolo_results[0].boxes:
        box = yolo_results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_results = holistic.process(image_rgb)

        if mp_results.pose_landmarks and mp_results.face_landmarks:
            keypoints = extract_keypoints(mp_results)
            onnx_input = {input_name: keypoints.astype(np.float32)}
            onnx_output = onnx_session.run(None, onnx_input)
            current_sign = id_to_gloss[str(np.argmax(onnx_output[0]))]
            mp_drawing.draw_landmarks(frame, mp_results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, None, mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
            mp_drawing.draw_landmarks(frame, mp_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))

    # --- Display ---
    cv2.rectangle(frame, (0, 0), (350, 80), (0, 0, 0), -1)
    cv2.putText(frame, f'Sign: {current_sign}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Emotion: {current_emotion}', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('ARTEMIS Demo', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print("✅ Demo finished cleanly.")
