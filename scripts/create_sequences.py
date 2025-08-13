import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from collections import Counter

# --- Configuration ---
FEATURE_PACK_PATH = '/root/features/feature_pack_v1.parquet'
OUTPUT_PATH = '/root/features/sequences.npz'
SEQUENCE_LENGTH = 20

def create_sequences_final_sorted():
    print("--- Starting Chronologically-Sorted Sequence Creation ---")

    # 1. Load data
    try:
        data_df = pd.read_parquet(FEATURE_PACK_PATH)
    except FileNotFoundError as e:
        print(f"FATAL: Could not find feature pack. {e}")
        return
    print("Data loaded successfully.")

    # 2. Prepare label and grouping columns
    data_df['label_id'] = data_df['label_id'].fillna(-1).astype(int)
    data_df['signer_id'] = data_df['img_stem'].apply(lambda x: x.split('_')[0])
    
    # --- NEW, CRUCIAL STEP: Extract frame number for sorting ---
    # This handles potential non-numeric parts by finding the last numeric part of the stem
    def get_frame_num(stem):
        try:
            return int(stem.split('_')[-1])
        except ValueError:
            return -1 # Return -1 for stems that don't end in a number
            
    data_df['frame_num'] = data_df['img_stem'].apply(get_frame_num)
    data_df = data_df[data_df['frame_num'] != -1] # Remove any rows that couldn't be parsed

    # 3. Group by signer and create sequences
    all_sequences = []
    all_labels = []
    feature_cols = [f'x{i}' for i in range(220)]
    
    for signer, group in tqdm(data_df.groupby('signer_id'), desc="Processing signers"):
        # --- NEW, CRUCIAL STEP: Sort the group by frame number ---
        sorted_group = group.sort_values(by='frame_num')
        
        video_features = sorted_group[feature_cols].values
        video_labels = sorted_group['label_id'].values
        
        # Slide a window across the chronologically sorted frames
        for i in range(len(video_features) - SEQUENCE_LENGTH + 1):
            sequence = video_features[i: i + SEQUENCE_LENGTH]
            labels_in_window = video_labels[i: i + SEQUENCE_LENGTH]
            
            valid_labels = [label for label in labels_in_window if label != -1]
            
            if valid_labels:
                most_common_label = Counter(valid_labels).most_common(1)[0][0]
                all_sequences.append(sequence)
                all_labels.append(most_common_label)

    if not all_sequences:
        print("FATAL: Still no sequences were created. The data might be more sparse than expected.")
        return

    # 4. Save the processed data
    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    print(f"\nCreated {X.shape[0]} sequences.")
    print(f"Shape of feature data (X): {X.shape}")
    print(f"Shape of label data (y): {y.shape}")

    np.savez_compressed(OUTPUT_PATH, X=X, y=y)
    
    print(f"\n✅ Successfully saved sequential data to: {OUTPUT_PATH}")

if __name__ == '__main__':
    create_sequences_final_sorted()
