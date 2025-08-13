import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import argparse, json, logging, os, sys, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. TCN Model Definition ---
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, num_classes, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], num_classes)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.network(x)
        out = self.linear(out[:, :, -1])
        return out

# --- 2. Dataset Definition ---
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

# --- 3. Main Training Function ---
def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    log_file = os.path.join(args.outdir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    data = np.load(args.sequence_pack)
    X, y = data['X'], data['y']
    id_map_path = '/root/models/mlp_resume/id_to_gloss.json'
    with open(id_map_path, 'r') as f:
        id_to_gloss = {int(k): v for k, v in json.load(f).items()}

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    model = TCN(
        input_size=220, 
        num_classes=len(id_to_gloss), 
        num_channels=[256, 128, 64],
        kernel_size=3, 
        dropout=0.25
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_acc = 0.0
    logging.info("Starting TCN model training...")
    for epoch in range(args.epochs):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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

    logging.info("Training finished. Best Val Acc: {best_val_acc:.4f}")
    class_names = [id_to_gloss[i] for i in range(len(id_to_gloss))]
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    logging.info(f"\nClassification Report:\n{report}")

# --- 4. Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TCN sequence model.")
    parser.add_argument('--sequence_pack', type=str, required=True, help="Path to the sequences.npz file.")
    parser.add_argument('--outdir', type=str, required=True, help="Directory to save model and results.")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
