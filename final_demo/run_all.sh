#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
while IFS= read -r ID; do
  [ -n "$ID" ] || continue
  echo "== Running demo for ID: $ID =="
  python3 "$HERE/tools/generate_final_demo.py" \
    --sample_id "$ID" \
    --features_dir "$HOME/Desktop/wlasl_mediapipe_features_rel" \
    --videos_dir   "$HOME/Desktop/WLASL/start_kit/videos" \
    --model        "$HERE/models/model_v2.onnx" \
    --labels       "$HERE/models/label2gloss.json" \
    --emotion_csv  "$HERE/logs/emotion_dataset_robust.csv"
done < "$HERE/curated/replay_candidates.txt"
