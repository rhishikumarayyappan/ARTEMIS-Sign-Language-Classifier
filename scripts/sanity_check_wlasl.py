# sanity_check_wlasl.py
import os
import json

# --- CONFIGURATION ---
# These paths point to the WLASL data we downloaded to the Desktop
VIDEO_PATH = os.path.expanduser('~/Desktop/videos')
JSON_PATH = os.path.expanduser('~/Desktop/WLASL/WLASL_v0.3.json')
SCRIPT_TO_CHECK = os.path.expanduser('~/Desktop/ARTEMIS_Final_Demo/scripts/generate_wlasl_dataset.py')

def run_wlasl_check():
    """Performs a sanity check for the WLASL data generation phase."""
    print("🚀 Starting WLASL Sanity Check...")
    all_ok = True

    # 1. Check for required folders and files
    print("\n--- Checking for required WLASL files and folders ---")
    paths_to_check = {
        "WLASL videos folder": VIDEO_PATH,
        "WLASL JSON label file": JSON_PATH,
        "Data generation script": SCRIPT_TO_CHECK
    }
    for name, path in paths_to_check.items():
        if os.path.exists(path):
            print(f"✅ Found: {name}")
        else:
            print(f"❌ MISSING: {name} at path '{path}'")
            all_ok = False

    # 2. Check video count
    if os.path.isdir(VIDEO_PATH):
        video_count = len([f for f in os.listdir(VIDEO_PATH) if f.endswith('.mp4')])
        print(f"\n--- Checking video count ---")
        print(f"✅ Found {video_count} downloaded videos.")
        if video_count < 1000:
            print("   - Warning: This seems low. The download may have been interrupted.")
            all_ok = False

    # 3. Final Verdict
    print("\n--------------------")
    print("--- VERDICT ---")
    print("--------------------")
    if all_ok:
        print("🎉 Sanity Check PASSED. The WLASL dataset is ready for processing.")
    else:
        print("🔥 Sanity Check FAILED. Please resolve the missing items listed above.")

if __name__ == "__main__":
    run_wlasl_check()
