import os
import sys

# A comprehensive script to check for all required files and libraries
# before running the final ARTEMIS demo.

# --- Define Paths ---
# Use absolute paths from the script's location for robustness
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CRITICAL_FILES = [
    os.path.join(PROJECT_ROOT, 'models', 'custom_yolo', 'best.pt'),
    os.path.join(PROJECT_ROOT, 'models', 'gru_v1', 'model.onnx'),
    os.path.join(PROJECT_ROOT, 'models', 'id_to_gloss.json'),
    os.path.join(PROJECT_ROOT, 'pyproject.toml') # The poetry config file
]

# --- Sanity Check Function ---
def run_system_check():
    """Runs a full check of the project's files and libraries."""
    print("🚀 Starting ARTEMIS System Sanity Check...")
    all_ok = True

    # 1. Check for critical files
    print("\n--- Checking for critical files ---")
    for file_path in CRITICAL_FILES:
        if os.path.exists(file_path):
            print(f"✅ Found: {os.path.relpath(file_path, PROJECT_ROOT)}")
        else:
            print(f"❌ MISSING: {os.path.relpath(file_path, PROJECT_ROOT)}")
            all_ok = False
            
    # 2. Check for critical library imports
    print("\n--- Checking library imports ---")
    try:
        import cv2; print("✅ Imported: cv2 (OpenCV)")
        import numpy; print("✅ Imported: numpy")
        import onnxruntime; print("✅ Imported: onnxruntime")
        import mediapipe; print("✅ Imported: mediapipe")
        import deepface; print("✅ Imported: deepface")
        from ultralytics import YOLO; print("✅ Imported: ultralytics (YOLO)")
        print("All libraries imported successfully.")
    except ImportError as e:
        print(f"❌ FAILED to import a critical library: {e}")
        print("Please run 'poetry install' to fix dependencies.")
        all_ok = False

    # 3. Final Verdict
    print("\n--------------------")
    print("--- VERDICT ---")
    print("--------------------")
    if all_ok:
        print("🎉 System Sanity Check PASSED. All files and libraries are in place.")
        print("You are ready to run the demo!")
    else:
        print("🔥 System Sanity Check FAILED. Please resolve the missing files or failed imports above.")
        sys.exit(1) # Exit with an error code

if __name__ == "__main__":
    run_system_check()
