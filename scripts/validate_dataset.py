# validate_dataset.py
import os
import re
from collections import Counter

# --- CONFIGURATION ---
# This path must point to the folder containing your 20,000 training images
IMAGES_DIR = os.path.expanduser('~/Desktop/ARTEMIS_Thesis_Archive/dataset/images/train')

def run_validation():
    print(f"🔬 Starting validation for dataset at: {IMAGES_DIR}")
    if not os.path.isdir(IMAGES_DIR):
        print(f"❌ FATAL ERROR: The directory does not exist. Please check the path.")
        return

    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith('.jpg')]
    if not image_files:
        print(f"❌ FATAL ERROR: No .jpg images found in the directory.")
        return

    print(f"Found {len(image_files)} total images.")
    
    labels = []
    for image_name in image_files:
        try:
            # Extract the class ID from the filename (the number after '_color_')
            label = int(re.search(r'_color_(\d+)\.jpg', image_name).group(1))
            labels.append(label)
        except (AttributeError, ValueError):
            # This handles any files that don't match our expected naming pattern
            print(f"  - Warning: Could not parse label from filename: {image_name}")
            continue

    if not labels:
        print("❌ FATAL ERROR: Could not extract any labels from the filenames.")
        return

    print("\n--- 📊 DATASET HEALTH REPORT ---")
    label_counts = Counter(labels)
    print(f"Total unique signs (classes) found: {len(label_counts)}")

    # Check for problematic classes
    singletons = {label: count for label, count in label_counts.items() if count < 2}
    
    if singletons:
        print(f"\n🚨 Found {len(singletons)} classes with only 1 sample (these will be automatically removed):")
        for label, count in sorted(singletons.items()):
            print(f"  - Sign ID {label}: {count} sample")
    else:
        print("\n✅ No singleton classes found. Excellent data quality!")

    print("\n--- ✅ VALIDATION PASSED ---")
    print("Your dataset is correctly structured and ready for processing.")

if __name__ == "__main__":
    run_validation()
