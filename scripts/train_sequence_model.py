import argparse
import json
import logging
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Model Definition (GRU Sequence Model) ---
class SequenceClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        # GRU layer processes the sequence of features
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, # Expects input shape (batch, seq, feature)
            dropout=0.2 if num_layers > 1 else 0
        )
        # Final linear layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # gru_out shape: (batch, seq_len, hidden_size)
        # h_n shape: (num_layers, batch, hidden_size)
        gru_out, h_n = self.gru(x)
        
        # We take the output of the last time step for classification
        last_time_step_out = gru_out[:, -1, :]
        
        out = self.fc(last_time_step_out)
        return out

# --- 2. Dataset Definition ---
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --- 3. Main Training Function ---
def main(args):
    # --- Setup ---
    os.makedirs(args.outdir, exist_ok=True)
    log_file = os.path.join(args.outdir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Data Loading ---
    logging.info(f"Loading sequence data from {args.sequence_pack}")
    data = np.load(args.sequence_pack)
    X, y = data['X'], data['y']
    
    # We still need the gloss map from the previous run
    id_map_path = '/root/models/mlp_resume/id_to_gloss.json'
    with open(id_map_path, 'r') as f:
        id_to_gloss = {int(k): v for k, v in json.load(f).items()}

    # --- Data Splitting ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    # --- Model Initialization ---
    model = SequenceClassifier(
        input_size=220, 
        hidden_size=256, 
        num_layers=2, 
        num_classes=len(id_to_gloss)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # --- Training Loop (Identical to before) ---
    best_val_acc = 0.0
    logging.info("Starting sequence model training...")
    for epoch in range(args.epochs):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = correct / total
        logging.info(f"Epoch {epoch+1}/{args.epochs}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.outdir, 'best_model.pth'))
            logging.info(f"  -> New best model saved.")

    logging.info("Training finished.")
    logging.info(f"Best Val Acc: {best_val_acc:.4f}")

    # --- Final Evaluation ---
    logging.info("\nClassification Report on Validation Set:")
    class_names = [id_to_gloss[i] for i in range(len(id_to_gloss))]
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    logging.info(f"\n{report}")
    
# --- 4. Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GRU sequence model.")
    parser.add_argument('--sequence_pack', type=str, required=True, help="Path to the sequences.npz file.")
    parser.add_argument('--outdir', type=str, required=True, help="Directory to save model and results.")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
