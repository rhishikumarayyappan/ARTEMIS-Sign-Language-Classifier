import os
import numpy as np
import onnx
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader

# --- 1. Create a Calibration Data Reader ---
# This class feeds sample data to the quantization tool
class SequenceDataReader(CalibrationDataReader):
    def __init__(self, data_path, batch_size=32):
        self.data = np.load(data_path)['X']
        self.batch_size = batch_size
        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            # Create an enumerator that yields batches of data
            self.enum_data = iter(
                [self.data[i:i + self.batch_size] for i in range(0, len(self.data), self.batch_size)]
            )
        
        # Yield the next batch
        try:
            return {'input': next(self.enum_data)}
        except StopIteration:
            return None # No more data

def quantize_onnx_model_static():
    model_dir = '/root/models/gru_v1'
    data_path = '/root/features/sequences.npz'
    onnx_model_path = os.path.join(model_dir, 'model.onnx')
    quantized_model_path = os.path.join(model_dir, 'model.quant.onnx')

    if not os.path.exists(onnx_model_path):
        print(f"FATAL: ONNX model not found at {onnx_model_path}.")
        return

    print("--- Starting STATIC ONNX Quantization ---")
    
    # 2. Create an instance of our data reader
    calibration_data_reader = SequenceDataReader(data_path)

    # 3. Perform static quantization
    quantize_static(
        model_input=onnx_model_path,
        model_output=quantized_model_path,
        calibration_data_reader=calibration_data_reader,
        weight_type=QuantType.QInt8, # Quantize weights to INT8
        activation_type=QuantType.QInt8 # Also quantize activations to INT8
    )

    original_size = os.path.getsize(onnx_model_path) / 1024  # in KB
    quantized_size = os.path.getsize(quantized_model_path) / 1024 # in KB

    print("\n✅ Static Quantization Complete!")
    print(f"Quantized model saved to: {quantized_model_path}")
    print(f"Original model size: {original_size:.2f} KB")
    print(f"Quantized model size: {quantized_size:.2f} KB")
    print(f"Size reduction: {((original_size - quantized_size) / original_size) * 100:.2f}%")

if __name__ == '__main__':
    quantize_onnx_model_static()
