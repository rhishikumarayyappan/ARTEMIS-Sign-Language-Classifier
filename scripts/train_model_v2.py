# train_model_v2.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# --- CONFIGURATION ---
FEATURE_FILE_PATH = os.path.expanduser('~/Desktop/ARTEMIS_Final_Demo/models/features_v2.npz')
MODEL_SAVE_PATH = os.path.expanduser('~/Desktop/ARTEMIS_Final_Demo/models/model_v2.h5')
ONNX_MODEL_SAVE_PATH = os.path.expanduser('~/Desktop/ARTEMIS_Final_Demo/models/model_v2.onnx')

# --- LOAD DATA ---
print("Loading pre-processed data...")
data = np.load(FEATURE_FILE_PATH)
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

# Reshape data for GRU: (samples, timesteps, features)
# Our data is per-frame, so we treat it as 1 timestep for training
X_train = np.expand_dims(X_train, axis=1)
X_val = np.expand_dims(X_val, axis=1)
X_test = np.expand_dims(X_test, axis=1)

# Get number of classes from the labels
num_classes = len(np.unique(y_train))
print(f"Data loaded. Found {num_classes} classes.")

# --- BUILD THE GRU MODEL ---
print("Building the GRU model...")
model = Sequential([
    GRU(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    GRU(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- TRAIN THE MODEL WITH SMART CALLBACKS ---
print("\nStarting model training...")
# This will save only the best model based on validation accuracy
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# This will stop training if the model doesn't improve for 5 epochs
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, mode='max')

history = model.fit(X_train, y_train,
                    epochs=50, # Set a high number, early stopping will find the best
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=[checkpoint, early_stopping])

# --- EVALUATE AND CONVERT ---
print("\nTraining complete. Evaluating the best model on the test set...")
# Load the best saved model
best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

print(f"\nConverting the best model to ONNX format at: {ONNX_MODEL_SAVE_PATH}")
# The 'opset' is important for compatibility
os.system(f'python -m tf2onnx.convert --saved-model {MODEL_SAVE_PATH} --output {ONNX_MODEL_SAVE_PATH} --opset 13')

print("\n✅ All tasks complete. You are ready to update the live demo script.")
