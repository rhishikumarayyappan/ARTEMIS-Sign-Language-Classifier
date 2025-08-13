# generate_wlasl_dataset.py
import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import json

# --- CONFIGURATION ---
VIDEO_PATH = os.path.expanduser('~/Desktop/videos')
JSON_PATH = os.path.expanduser('~/Desktop/WLASL/WLASL_v0.3.json')
OUTPUT_FILE_PATH = os.path.expanduser('~/Desktop/ARTEMIS_Final_Demo/models/features_wlasl.npz')

# --- INITIALIZE MEDIAPIPE ---
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)

def extract_and_normalize_keypoints(results, frame_shape):
    h, w, _ = frame_shape
    if results.pose_landmarks:
        nose = results.pose_landmarks.landmark[0]
        pose = np.array([((res.x * w) - (nose.x*w), (res.y * h) - (nose.y*h)) for res in results.pose_landmarks.landmark]).flatten()
        face = np.array([((res.x * w) - (nose.x*w), (res.y * h) - (nose.y*h)) for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*2)
        lh = np.array([((res.x * w) - (nose.x*w), (res.y * h) - (nose.y*h)) for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*2)
        rh = np.array([((res.x * w) - (nose.x*w), (res.y * h) - (nose.y*h)) for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*2)
        face = face[:70]
        return np.concatenate([pose, face, lh, rh])
    else:
        return np.zeros(220)

# --- MAIN PROCESSING LOGIC ---
def create_feature_pack():
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    features_per_video, labels = [], []
    label_map = {}
    label_counter = 0

    print(f"Starting to process videos from {VIDEO_PATH}...")
    
    for entry in tqdm(data):
        gloss = entry['gloss']
        if gloss not in label_map:
            label_map[gloss] = label_counter
            label_counter += 1
        
        label_index = label_map[gloss]

        for instance in entry['instances']:
            video_id = instance['video_id']
            video_file = os.path.join(VIDEO_PATH, f"{video_id}.mp4")

            if not os.path.exists(video_file):
                continue

            cap = cv2.VideoCapture(video_file)
            video_keypoints = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)
                keypoints = extract_and_normalize_keypoints(results, frame.shape)
                video_keypoints.append(keypoints)
            
            cap.release()
            
            if video_keypoints:
                features_per_video.append(np.array(video_keypoints))
                labels.append(label_index)

    print(f"\nSuccessfully processed {len(features_per_video)} videos.")
    print(f"\nSaving new feature pack to: {OUTPUT_FILE_PATH}")
    np.save(OUTPUT_FILE_PATH, {'features': features_per_video, 'labels': np.array(labels), 'label_map': label_map})
    print("✅ Done.")

if __name__ == "__main__":
    create_feature_pack()
