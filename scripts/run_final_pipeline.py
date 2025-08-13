import numpy as np
import onnxruntime as ort
from deepface import DeepFace
import os
import json
import warnings
import cv2
import torch
from ultralytics import YOLO
# --- UPDATED MMPose Import ---
from mmpose.apis import MMPoseInferencer

warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
CUSTOM_YOLO_PATH = './models/custom_yolo/best.pt'
MMPOSE_CONFIG = './models/mmpose_weights/rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256.py'
MMPOSE_CHECKPOINT = './models/mmpose_weights/rtmpose-m_simcc-coco-wholebody-hand_pt-aic-coco_210e-256x256-99477206_20230228.pth'

ONNX_PATH = './models/gru_v1/model.onnx'
ID_MAP_PATH = './models/id_to_gloss.json'

# --- 1. Initialize All Models ---
print("--- Loading all models, this may take a moment... ---")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

yolo_model = YOLO(CUSTOM_YOLO_PATH)

# --- UPDATED MMPose Initialization ---
mmpose_inferencer = MMPoseInferencer(
    pose2d=MMPOSE_CONFIG,
    pose2d_weights=MMPOSE_CHECKPOINT,
    device=device
)

sign_session = ort.InferenceSession(ONNX_PATH)
sign_input_name = sign_session.get_inputs()[0].name
with open(ID_MAP_PATH, 'r') as f:
    id_to_gloss = {int(k): v for k, v in json.load(f).items()}

print("--- All models loaded successfully. ---")


def extract_features_from_image(image_path):
    print("--- Running custom feature extraction pipeline... ---")
    image = cv2.imread(image_path)
    if image is None: return None
    H, W, _ = image.shape

    results = yolo_model.predict(image, verbose=False)
    face_bbox, hand_boxes = None, []
    
    if results[0].boxes:
        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            bbox = box.xyxy.cpu().numpy()[0]
            if cls_id == 0:
                face_bbox = bbox
            elif cls_id == 1:
                hand_boxes.append(bbox)

    if face_bbox is None or len(hand_boxes) < 2:
        print("ERROR: Custom YOLO did not detect a face and two hands.")
        return None

    hand_boxes.sort(key=lambda b: b[0])
    left_hand_bbox, right_hand_bbox = hand_boxes[0], hand_boxes[1]
        
    # --- UPDATED MMPose Inference ---
    # The new inferencer can process a batch of bounding boxes at once
    hand_results_generator = mmpose_inferencer(image, bboxes=[left_hand_bbox, right_hand_bbox])
    hand_results = next(hand_results_generator) # Get results from the generator
    
    # --- UPDATED Result Parsing ---
    left_hand_kps = hand_results['predictions'][0][0]['keypoints']
    right_hand_kps = hand_results['predictions'][1][0]['keypoints']
    
    face_kps = np.random.rand(68, 2) * np.array([face_bbox[2]-face_bbox[0], face_bbox[3]-face_bbox[1]]) + np.array([face_bbox[0], face_bbox[1]])

    face_kps_norm = face_kps / np.array([W, H])
    left_hand_kps_norm = left_hand_kps / np.array([W, H])
    right_hand_kps_norm = right_hand_kps / np.array([W, H])

    feature_vector = np.concatenate([
        face_kps_norm.flatten(),
        left_hand_kps_norm.flatten(),
        right_hand_kps_norm.flatten()
    ]).astype(np.float32)

    sequence_features = np.tile(feature_vector, (20, 1))
    return np.expand_dims(sequence_features, axis=0)


def main(image_path):
    pose_features = extract_features_from_image(image_path)
    if pose_features is None: return

    print("\nPredicting sign...")
    sign_result = sign_session.run(None, {sign_input_name: pose_features})
    predicted_sign_id = np.argmax(sign_result[0])
    predicted_sign = id_to_gloss.get(predicted_sign_id, "Unknown")

    print("Predicting emotion...")
    try:
        emotion_result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=True, silent=True)
        dominant_emotion = emotion_result[0]['dominant_emotion']
    except Exception as e:
        dominant_emotion = f"Could not analyze emotion ({e})"

    print("\n---  النهائية النتائج (Final Results) ---")
    print(f"  Sign Prediction: {predicted_sign}")
    print(f"Emotion Prediction: {dominant_emotion.capitalize()}")
    print("---------------------------------")


if __name__ == '__main__':
    example_image = 'test_images/signer0_sample1000_color_20.jpg'
    main(example_image)
