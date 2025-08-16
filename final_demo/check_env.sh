#!/usr/bin/env bash
set -euo pipefail
echo "== Final Demo Environment Check =="
python3 -c "import sys; print('Python', sys.version.split()[0])"
python3 -c "import numpy, onnxruntime, cv2; print('OK: numpy/onnxruntime/opencv')" 2>/dev/null || \
  python3 -m pip install -r requirements.txt
for p in \
  final_demo/tools/generate_final_demo.py \
  final_demo/models/model_v2.onnx \
  final_demo/models/label2gloss.json \
  final_demo/logs/emotion_dataset_robust.csv \
  final_demo/curated/replay_candidates.txt; do
  [ -e "$p" ] && echo "[OK] $p" || echo "[MISSING] $p"
done
[ -e "$HOME/Desktop/wlasl_mediapipe_features_rel" ] && echo "[OK] features path" || echo "[WARN] features missing"
[ -e "$HOME/Desktop/WLASL/start_kit/videos" ]       && echo "[OK] videos path"   || echo "[WARN] videos missing"
