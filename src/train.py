import pickle
import numpy as np
from pathlib import Path

# ==============================
# CONFIGURATION
# ==============================

DATA_FILE = Path("data/notes.pkl")
MODEL_FOLDER = Path("models")

SEQUENCE_LENGTH = 50

# ==============================
# LOAD DATA
# ==============================

print("Loading music dataset...")

with open(DATA_FILE, "rb") as file:
    notes = pickle.load(file)

print(f"Total notes/chords: {len(notes)}")

# ==============================
# CREATE VOCABULARY
# ==============================

unique_notes = sorted(set(notes))

print(f"Unique notes/chords: {len(unique_notes)}")

note_to_int = {
    note: number
    for number, note in enumerate(unique_notes)
}

int_to_note = {
    number: note
    for note, number in note_to_int.items()
}

# ==============================
# CREATE SEQUENCES
# ==============================

network_input = []
network_output = []

print("Creating training sequences...")

for i in range(len(notes) - SEQUENCE_LENGTH):

    sequence = notes[i:i + SEQUENCE_LENGTH]
    target = notes[i + SEQUENCE_LENGTH]

    network_input.append(
        [note_to_int[n] for n in sequence]
    )

    network_output.append(
        note_to_int[target]
    )

# Convert to NumPy arrays
X = np.array(network_input)
y = np.array(network_output)

print("\n===================================")
print("DATASET PREPARATION COMPLETE")
print("===================================")

print(f"Training samples : {len(X)}")
print(f"Sequence length  : {X.shape[1]}")
print(f"Vocabulary size  : {len(unique_notes)}")

# ==============================
# SAVE PREPARED DATA
# ==============================

MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

np.save("data/X.npy", X)
np.save("data/y.npy", y)

with open("data/note_to_int.pkl", "wb") as file:
    pickle.dump(note_to_int, file)

with open("data/int_to_note.pkl", "wb") as file:
    pickle.dump(int_to_note, file)

print("\nSaved files:")

print("data/X.npy")
print("data/y.npy")
print("data/note_to_int.pkl")
print("data/int_to_note.pkl")