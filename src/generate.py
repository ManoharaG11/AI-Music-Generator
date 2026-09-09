import numpy as np
import pickle
from pathlib import Path

from tensorflow.keras.models import load_model
from music21 import stream, note, chord, tempo


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_FILE = Path("models/music_generator.keras")

NOTE_TO_INT_FILE = Path("data/note_to_int.pkl")
INT_TO_NOTE_FILE = Path("data/int_to_note.pkl")

OUTPUT_FOLDER = Path("outputs")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_FOLDER / "generated_music.mid"

SEQUENCE_LENGTH = 50
GENERATE_NOTES = 200


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading trained LSTM model...")

model = load_model(MODEL_FILE)

print("Model loaded successfully.")


# ==========================================
# LOAD NOTE MAPPINGS
# ==========================================

with open(NOTE_TO_INT_FILE, "rb") as file:
    note_to_int = pickle.load(file)

with open(INT_TO_NOTE_FILE, "rb") as file:
    int_to_note = pickle.load(file)


# ==========================================
# LOAD ORIGINAL NOTES
# ==========================================

with open("data/notes.pkl", "rb") as file:
    notes = pickle.load(file)


# ==========================================
# CREATE RANDOM SEED
# ==========================================

print("\nSelecting starting musical sequence...")

start = np.random.randint(0, len(notes) - SEQUENCE_LENGTH)

pattern = notes[start:start + SEQUENCE_LENGTH]

print(f"Starting position: {start}")


# ==========================================
# GENERATE NEW NOTES
# ==========================================

print("\nGenerating music...")

generated_notes = []

for i in range(GENERATE_NOTES):

    # Convert current sequence to numbers
    input_sequence = [
        note_to_int[n]
        for n in pattern
    ]

    input_sequence = np.array(input_sequence)
    input_sequence = np.reshape(
        input_sequence,
        (1, SEQUENCE_LENGTH)
    )

    # Predict next note
    prediction = model.predict(
        input_sequence,
        verbose=0
    )

    predicted_index = np.argmax(prediction)

    predicted_note = int_to_note[predicted_index]

    # Add predicted note
    generated_notes.append(predicted_note)

    # Move sequence forward
    pattern.append(predicted_note)
    pattern = pattern[1:]


    if (i + 1) % 25 == 0:
        print(f"Generated {i + 1}/{GENERATE_NOTES} notes...")


# ==========================================
# CONVERT NOTES TO MIDI
# ==========================================

print("\nConverting generated music to MIDI...")

music_stream = stream.Stream()

music_stream.append(
    tempo.MetronomeMark(number=100)
)

for pattern_note in generated_notes:

    # Chord
    if "." in pattern_note:

        try:
            chord_notes = [
                int(pitch)
                for pitch in pattern_note.split(".")
            ]

            new_chord = chord.Chord(chord_notes)
            new_chord.quarterLength = 0.5

            music_stream.append(new_chord)

        except:
            pass

    # Single note
    else:

        try:
            new_note = note.Note(pattern_note)
            new_note.quarterLength = 0.5

            music_stream.append(new_note)

        except:
            pass


# ==========================================
# SAVE MIDI
# ==========================================

music_stream.write(
    "midi",
    fp=OUTPUT_FILE
)

print("\n===================================")
print("AI MUSIC GENERATION COMPLETE")
print("===================================")

print(f"Generated notes : {len(generated_notes)}")
print(f"MIDI file       : {OUTPUT_FILE}")