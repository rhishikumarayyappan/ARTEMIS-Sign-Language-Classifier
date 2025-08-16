#!/usr/bin/env bash
set -euo pipefail
ID="${1:-}"
if [ -z "$ID" ]; then
  echo "Usage: ./run_one.sh <sample_id>"; exit 1
fi
python3 "$(dirname "$0")/tools/generate_final_demo.py" \
  --sample_id "$ID" \
  --features_dir "$HOME/Desktop/wlasl_mediapipe_features_rel" \
  --videos_dir   "$HOME/Desktop/WLASL/start_kit/videos" \
  --model        "$(dirname "$0")/models/model_v2.onnx" \
  --labels       "$(dirname "$0")/models/label2gloss.json" \
  --emotion_csv  "$(dirname "$0")/logs/emotion_dataset_robust.csv"
