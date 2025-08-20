import time, numpy as np, onnxruntime as ort

sess = ort.InferenceSession("models/gru_v2.onnx", providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
out = sess.get_outputs()[0].name

x = np.random.rand(1, 32, 150).astype("float32")

# warmup
for _ in range(5):
    _ = sess.run([out], {inp: x})

N = 50
t0 = time.perf_counter()
for _ in range(N):
    _ = sess.run([out], {inp: x})
t1 = time.perf_counter()

avg_ms = (t1 - t0) * 1000.0 / N
print(f"Avg model-only latency over {N} runs: {avg_ms:.2f} ms")
