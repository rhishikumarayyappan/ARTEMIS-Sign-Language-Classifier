# generate_new_dataset.py - Final Version, Handles All Small Classes
import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import re
from collections import Counter

# --- CONFIGURATION ---
IMAGES_DIR = os.path.expanduser('~/Desktop/ARTEMIS_Thesis_Archive/dataset/images/train')
OUTPUT_FILE_PATH = os.path.expanduser('~/Desktop/ARTEMIS_Final_Demo/models/features_v2.npz')

# --- INITIALIZE MEDIAPIPE ---
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5)

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
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith('.jpg')]
    features, labels = [], []

    for image_name in tqdm(image_files, desc="Processing Images"):
        try:
            label = int(re.search(r'_color_(\d+)\.jpg', image_name).group(1))
        except (AttributeError, ValueError):
            continue

        image_path = os.path.join(IMAGES_DIR, image_name)
        frame = cv2.imread(image_path)
        if frame is None: continue

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        keypoints = extract_and_normalize_keypoints(results, frame.shape)
        features.append(keypoints)
        labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    print(f"\nSuccessfully processed {len(features)} images.")

    # --- THE FINAL FIX: Use a robust threshold of 3 ---
    print("Filtering out classes that are too small for splitting...")
    label_counts = Counter(labels)
    # A class must have at least 3 samples to be split into train/val/test
    small_classes = {label for label, count in label_counts.items() if count < 3}

    if small_classes:
        print(f"Found and removed {len(small_classes)} classes with fewer than 3 samples.")
        mask = np.array([label not in small_classes for label in labels])
        features = features[mask]
        labels = labels[mask]
        print(f"Dataset size after filtering: {len(features)} images.")
    else:
        print("No small classes found.")

    # --- Splitting the data ---
    print("\nSplitting data into training, validation, and test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"\nSaving new feature pack to: {OUTPUT_FILE_PATH}")
    np.savez(OUTPUT_FILE_PATH, 
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)
    print("✅ Done.")

if __name__ == "__main__":
    create_feature_pack()
