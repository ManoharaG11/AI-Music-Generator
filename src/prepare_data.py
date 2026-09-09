from music21 import converter, instrument, note, chord
from pathlib import Path
import pickle

MIDI_FOLDER = Path("data/midi")
OUTPUT_FILE = Path("data/notes.pkl")

notes = []

midi_files = list(MIDI_FOLDER.glob("*.mid")) + list(MIDI_FOLDER.glob("*.midi"))

print(f"Found {len(midi_files)} MIDI files.")

for midi_file in midi_files:
    print(f"Processing: {midi_file.name}")

    try:
        midi = converter.parse(midi_file)

        parts = instrument.partitionByInstrument(midi)

        if parts:
            elements = parts.parts[0].recurse()
        else:
            elements = midi.flat.notes

        for element in elements:

            if isinstance(element, note.Note):
                notes.append(str(element.pitch))

            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(n) for n in element.normalOrder))

    except Exception as e:
        print(f"Could not process {midi_file.name}: {e}")

# Save extracted notes
with open(OUTPUT_FILE, "wb") as file:
    pickle.dump(notes, file)

print("\n===================================")
print("MIDI PREPROCESSING COMPLETE")
print("===================================")
print(f"MIDI files processed : {len(midi_files)}")
print(f"Total notes/chords   : {len(notes)}")
print(f"Saved dataset        : {OUTPUT_FILE}")

print("\nFirst 20 examples:")
print(notes[:20])