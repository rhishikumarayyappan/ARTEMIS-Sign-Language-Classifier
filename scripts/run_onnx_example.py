import numpy as np, onnxruntime as ort

# Create dummy input (batch=1, 32x150) — adjust shape if your model expects (32,150) only
x = np.random.rand(1, 32, 150).astype("float32")

sess = ort.InferenceSession("models/gru_v2.onnx", providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
out = sess.get_outputs()[0].name
logits = sess.run([out], {inp: x})[0]  # shape: (1, num_classes)

# Temperature scaling + abstention rule (T=0.60, tau=0.89)
T = 0.60
tau = 0.89
p = np.exp(logits / T) / np.exp(logits / T).sum(axis=-1, keepdims=True)
conf = float(p.max())
label = int(p.argmax())

print(f"Top-1 label: {label}, confidence: {conf:.3f}, decision: {'EMIT' if conf>=tau else 'ABSTAIN'}")
