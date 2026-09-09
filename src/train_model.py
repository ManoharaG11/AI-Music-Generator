import numpy as np
import pickle
from pathlib import Path

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint


# ==========================================
# CONFIGURATION
# ==========================================

X_FILE = Path("data/X.npy")
Y_FILE = Path("data/y.npy")
NOTE_TO_INT_FILE = Path("data/note_to_int.pkl")

MODEL_FOLDER = Path("models")
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_FOLDER / "music_generator.keras"

SEQUENCE_LENGTH = 50

# Smaller model = faster training
EMBEDDING_DIM = 64
LSTM_UNITS = 64

# Fast prototype training
EPOCHS = 3
BATCH_SIZE = 256

# Train on only part of the dataset
MAX_SAMPLES = 30000


# ==========================================
# LOAD DATA
# ==========================================

print("Loading training data...")

X = np.load(X_FILE)
y = np.load(Y_FILE)

with open(NOTE_TO_INT_FILE, "rb") as file:
    note_to_int = pickle.load(file)

vocabulary_size = len(note_to_int)

print(f"Full dataset samples : {len(X)}")
print(f"Vocabulary size      : {vocabulary_size}")


# ==========================================
# USE SMALL TRAINING SUBSET
# ==========================================

X = X[:MAX_SAMPLES]
y = y[:MAX_SAMPLES]

print(f"Samples used for training : {len(X)}")
print(f"Sequence length           : {X.shape[1]}")


# ==========================================
# PREPARE DATA
# ==========================================

X = X.astype(np.int32)
y = y.astype(np.int32)


# ==========================================
# BUILD LSTM MODEL
# ==========================================

print("\nBuilding LSTM model...")

model = Sequential([

    Embedding(
        input_dim=vocabulary_size,
        output_dim=EMBEDDING_DIM
    ),

    LSTM(
        LSTM_UNITS
    ),

    Dense(
        vocabulary_size,
        activation="softmax"
    )
])


# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)


# ==========================================
# MODEL SUMMARY
# ==========================================

print("\n===================================")
print("MODEL ARCHITECTURE")
print("===================================")

model.summary()


# ==========================================
# SAVE BEST MODEL
# ==========================================

checkpoint = ModelCheckpoint(
    MODEL_FILE,
    monitor="loss",
    save_best_only=True,
    verbose=1
)


# ==========================================
# TRAIN
# ==========================================

print("\n===================================")
print("STARTING FAST LSTM TRAINING")
print("===================================")

history = model.fit(
    X,
    y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint],
    shuffle=True
)


# ==========================================
# SAVE MODEL
# ==========================================

model.save(MODEL_FILE)


print("\n===================================")
print("LSTM TRAINING COMPLETE")
print("===================================")

print(f"Model saved to: {MODEL_FILE}")
