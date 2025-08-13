import argparse
import json
import logging
import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Model Definition ---
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

# --- 2. Dataset Definition ---
class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --- 3. Main Training Function ---
def main(args):
    # --- Proactive Setup ---
    os.makedirs(args.outdir, exist_ok=True)

    # Configure logging to file and console
    log_file = os.path.join(args.outdir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Starting run with args: {args}")

    # --- Robust Data Loading & Validation ---
    logging.info("Loading data...")
    try:
        df = pd.read_parquet(args.feature_pack)
    except FileNotFoundError:
        logging.error(f"FATAL: Input file not found at {args.feature_pack}")
        return

    assert 'label_id' in df.columns, "FATAL: 'label_id' column not found in feature pack."
    assert 'gloss' in df.columns, "FATAL: 'gloss' column not found in feature pack."
    
    labeled_df = df[df['label_id'] >= 0].copy()
    
    assert not labeled_df.empty, "FATAL: No labeled data found. Cannot train."
    logging.info(f"Found {len(labeled_df)} labeled samples.")

    # --- Data Preparation ---
    gloss_to_id = {gloss: id for id, gloss in enumerate(labeled_df['gloss'].unique())}
    id_to_gloss = {id: gloss for gloss, id in gloss_to_id.items()}
    labeled_df['label_id'] = labeled_df['gloss'].map(gloss_to_id)
    
    with open(os.path.join(args.outdir, 'id_to_gloss.json'), 'w') as f:
        json.dump(id_to_gloss, f, indent=2)
    logging.info(f"Found {len(gloss_to_id)} classes: {list(gloss_to_id.keys())}")

    feature_cols = [f'x{i}' for i in range(220)]
    X = labeled_df[feature_cols].values
    y = labeled_df['label_id'].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_dataset = PoseDataset(X_train, y_train)
    val_dataset = PoseDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    # --- Model Initialization ---
    num_classes = len(gloss_to_id)
    model = ASLClassifier(input_size=X.shape[1], num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # --- Training Loop ---
    best_val_acc = 0.0
    logging.info("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(features)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        # Validation loop
        model.eval()
        correct = 0
        total = 0
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
        logging.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.outdir, 'best_model.pth'))
            logging.info(f"  -> New best model saved with Val Acc: {best_val_acc:.4f}")

        # Periodic checkpointing
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, os.path.join(args.outdir, f'checkpoint_epoch_{epoch+1}.pth'))
            logging.info(f"  -> Saved periodic checkpoint at epoch {epoch+1}")

    logging.info("Training finished.")
    logging.info(f"Best Val Acc: {best_val_acc:.4f}")

    # --- Final Evaluation ---
    logging.info("\nClassification Report on Validation Set:")
    class_names = [id_to_gloss[i] for i in range(num_classes)]
    report = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    )
    logging.info(f"\n{report}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(args.outdir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    logging.info(f"Confusion matrix saved to {cm_path}")

# --- 4. Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a robust MLP on pose features.")
    parser.add_argument('--feature_pack', type=str, required=True, help="Path to the feature_pack_v1.parquet file.")
    parser.add_argument('--outdir', type=str, required=True, help="Directory to save model and results.")
    parser.add_argument('--batch_size', type=int, default=256, help="Training batch size.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate.")
    args = parser.parse_args()
    main(args)
