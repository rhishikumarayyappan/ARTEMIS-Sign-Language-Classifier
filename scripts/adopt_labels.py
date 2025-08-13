import argparse
import json
import os
import torch
import pandas as pd
import numpy as np

# --- Re-define our exact model architecture here for consistency ---
class ASLClassifier(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

def main(args):
    # --- Setup and validation ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    assert os.path.isdir(args.model_dir), f"Model directory not found: {args.model_dir}"
    model_path = os.path.join(args.model_dir, 'model.pt') # The renamed model file
    id_map_path = os.path.join(args.model_dir, 'id_to_gloss.json')
    assert os.path.exists(model_path), f"Model file not found: {model_path}"
    assert os.path.exists(id_map_path), f"ID map file not found: {id_map_path}"

    # --- Load model and mappings ---
    print("Loading model and label map...")
    with open(id_map_path, 'r') as f:
        id_to_gloss = {int(k): v for k, v in json.load(f).items()}
    
    gloss_to_id = {v: k for k, v in id_to_gloss.items()}
    num_classes = len(id_to_gloss)
    
    # Instantiate the *correct* model and load its weights
    model = ASLClassifier(input_size=220, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # --- Load data ---
    print("Loading features, labels, and cluster assignments...")
    features_df = pd.read_parquet(args.feature_pack)
    labels_df = pd.read_csv(args.labels_csv)
    clusters_df = pd.read_csv(args.cluster_assignments)

    # Find the target gloss ID
    if args.target_gloss not in gloss_to_id:
        raise ValueError(f"Target gloss '{args.target_gloss}' not found in model's vocabulary.")
    target_label_id = gloss_to_id[args.target_gloss]

    # --- Find candidate frames ---
    # Merge dataframes to link frames to their cluster
    merged_df = pd.merge(features_df[['img_stem']], clusters_df, on='img_stem')
    
    # Select frames from the target cluster that are currently unlabeled
    candidate_stems = merged_df[
        (merged_df['cluster_id_km'] == args.cluster_id) &
        (merged_df['img_stem'].isin(labels_df[labels_df['label_id'] == -1]['img_stem']))
    ]['img_stem']
    
    candidate_features = features_df[features_df['img_stem'].isin(candidate_stems)]
    
    if candidate_features.empty:
        print("No new unlabeled frames found in the target cluster. Nothing to do.")
        return

    print(f"Found {len(candidate_features)} unlabeled candidate frames in cluster {args.cluster_id}.")

    # --- Run inference ---
    feature_cols = [f'x{i}' for i in range(220)]
    X_candidates = torch.tensor(candidate_features[feature_cols].values, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_candidates)
        probabilities = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probabilities, dim=1)

    # --- Adopt high-confidence labels ---
    adopted_count = 0
    stems_to_label = []
    for idx, stem in enumerate(candidate_features['img_stem']):
        if predictions[idx].item() == target_label_id and confidences[idx].item() >= args.conf_threshold:
            stems_to_label.append(stem)
            adopted_count += 1
            
    if adopted_count == 0:
        print("No frames met the confidence threshold. No labels were changed.")
        return

    print(f"Adopting {adopted_count} new labels for '{args.target_gloss}' with confidence >= {args.conf_threshold}.")
    
    # Update the main labels.csv file
    labels_df.loc[labels_df['img_stem'].isin(stems_to_label), 'label_id'] = target_label_id
    labels_df.loc[labels_df['img_stem'].isin(stems_to_label), 'gloss'] = args.target_gloss
    
    labels_df.to_csv(args.labels_csv, index=False)
    print(f"Successfully updated {args.labels_csv}.")


if __name__ == "__main__":
    # Use environment variables for simplicity, as planned
    parser = argparse.ArgumentParser(description="Robust label adoption script.")
    parser.add_argument('--model_dir', type=str, default=os.getenv('MODEL_DIR'), help="Directory containing model.pt and id_to_gloss.json")
    parser.add_argument('--cluster_id', type=int, default=int(os.getenv('CLUSTER_ID')), help="The cluster ID to target.")
    parser.add_argument('--target_gloss', type=str, default=os.getenv('TARGET_GLOSS'), help="The gloss to assign.")
    parser.add_argument('--conf_threshold', type=float, default=float(os.getenv('CONF_THRESHOLD')), help="Confidence threshold for adoption.")
    parser.add_argument('--feature_pack', type=str, default='/root/features/feature_pack_v1.parquet')
    parser.add_argument('--labels_csv', type=str, default='/root/labels/labels.csv')
    parser.add_argument('--cluster_assignments', type=str, default='/root/analysis/frame_cluster_assignments_v2.csv')

    args = parser.parse_args()
    main(args)
