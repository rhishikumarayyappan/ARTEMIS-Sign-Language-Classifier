import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_onnx_model():
    model_dir = '/root/models/gru_v1'
    onnx_model_path = os.path.join(model_dir, 'model.onnx')
    quantized_model_path = os.path.join(model_dir, 'model.quant.onnx')

    if not os.path.exists(onnx_model_path):
        print(f"FATAL: ONNX model not found at {onnx_model_path}. Please run the export script first.")
        return

    print("--- Starting ONNX Quantization ---")
    print(f"Input model: {onnx_model_path}")

    # Perform dynamic quantization
    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=quantized_model_path,
        weight_type=QuantType.QInt8 # Use 8-bit integers for weights
    )

    original_size = os.path.getsize(onnx_model_path) / 1024  # in KB
    quantized_size = os.path.getsize(quantized_model_path) / 1024 # in KB

    print("\n✅ Quantization Complete!")
    print(f"Quantized model saved to: {quantized_model_path}")
    print(f"Original model size: {original_size:.2f} KB")
    print(f"Quantized model size: {quantized_size:.2f} KB")
    print(f"Size reduction: {((original_size - quantized_size) / original_size) * 100:.2f}%")

if __name__ == '__main__':
    quantize_onnx_model()

