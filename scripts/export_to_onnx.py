import torch
import torch.nn as nn
import os
import json

# --- 1. Define the NEW SequenceClassifier class ---
class SequenceClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        gru_out, h_n = self.gru(x)
        last_time_step_out = gru_out[:, -1, :]
        out = self.fc(last_time_step_out)
        return out

def export_model():
    # --- 2. Update the model directory ---
    model_dir = '/root/models/gru_v1'
    # We need the old id_map, as the classes are the same
    id_map_path = '/root/models/mlp_resume/id_to_gloss.json' 

    model_path = os.path.join(model_dir, 'best_model.pth')
    onnx_path = os.path.join(model_dir, 'model.onnx')

    print("--- Starting Sequence Model ONNX Export ---")

    with open(id_map_path, 'r') as f:
        num_classes = len(json.load(f))

    model = SequenceClassifier(
        input_size=220, 
        hidden_size=256, 
        num_layers=2, 
        num_classes=num_classes
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print("Successfully loaded trained sequence model weights.")

    # --- 3. Update the dummy input shape ---
    # Shape is (batch_size, sequence_length, num_features)
    dummy_input = torch.randn(1, 20, 220)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}
    )
    print(f"\n✅ Sequence model successfully exported to: {onnx_path}")

if __name__ == '__main__':
    export_model()
