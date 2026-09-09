# 🎵 AI Music Generator

An AI-powered music generation system that learns musical patterns from MIDI files using a Long Short-Term Memory (LSTM) neural network and generates new musical sequences automatically.

## 📌 Project Overview

Music is sequential data where the current note is influenced by previous notes. This project uses an LSTM-based deep learning model to learn patterns from a collection of MIDI files and generate new musical sequences.

The generated sequence is converted back into a MIDI file, which can be downloaded and played using a MIDI-compatible music player.

## 🎯 Problem Statement

Creating original musical compositions manually can be time-consuming. Traditional rule-based systems have difficulty learning complex musical patterns.

The goal of this project is to build an AI system capable of learning patterns from existing MIDI music and generating new musical sequences.

## 💡 Solution

The system follows this pipeline:

MIDI Dataset
↓
Music Preprocessing using music21
↓
Note and Chord Extraction
↓
Numerical Encoding
↓
Sequence Creation
↓
LSTM Neural Network
↓
Next Note Prediction
↓
Generated Musical Sequence
↓
MIDI File
↓
Download / Play

## 🧠 How the LSTM Works

The model receives a sequence of previous musical events and learns to predict the next event.

For example:

Previous 50 notes/chords → LSTM → Next musical event

The generated event is then added back to the sequence and used to predict the following event.

This process is repeated to create a complete musical sequence.

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* music21
* NumPy
* Pandas
* Streamlit
* LSTM Neural Networks
* MIDI

## 📂 Project Structure

```text
AI_Music_Generator/
│
├── data/
│   ├── midi/
│   ├── notes.pkl
│   ├── X.npy
│   ├── y.npy
│   ├── note_to_int.pkl
│   └── int_to_note.pkl
│
├── models/
│   └── music_generator.keras
│
├── outputs/
│   └── generated_music.mid
│
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   ├── train_model.py
│   └── generate.py
│
├── app.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation

Clone the repository and open the project directory.

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

### Step 1 — Prepare MIDI Data

Place MIDI files inside:

```text
data/midi/
```

Then run:

```bash
python src/prepare_data.py
```

### Step 2 — Create Training Sequences

```bash
python src/train.py
```

### Step 3 — Train the LSTM Model

```bash
python src/train_model.py
```

The trained model will be saved in:

```text
models/music_generator.keras
```

### Step 4 — Generate MIDI

```bash
python src/generate.py
```

The generated music will be saved as:

```text
outputs/generated_music.mid
```

### Step 5 — Launch Streamlit Application

```bash
streamlit run app.py
```

The application provides an interface for generating and downloading new MIDI music.

## 📊 Dataset

The model was trained using a collection of MIDI piano files.

During preprocessing, the project processed:

* MIDI files: 140
* Musical events extracted: 315,342
* Sequence length: 50
* Training samples prepared: 315,292
* Vocabulary size: 1,657

For the lightweight training configuration, 30,000 training sequences were used to reduce training time.

## 🤖 Model

The project uses:

* Embedding layer
* LSTM layer
* Dense output layer
* Softmax activation

The model predicts the probability of each possible musical event and selects the next event to continue the generated sequence.

## 🌐 Streamlit Application

The Streamlit interface allows users to:

* Select the number of musical events
* Generate new music
* Download the generated MIDI file

## 🚀 Future Improvements

Possible improvements include:

* Temperature-based probabilistic sampling
* Larger and more diverse datasets
* Multiple LSTM layers
* Better handling of note duration and timing
* Instrument selection
* WAV/MP3 audio generation
* Real-time music generation
* Transformer-based music generation


## 👨‍💻 Author

**G Manohara**

Computer Science and Business Systems
Dr. Ambedkar Institute of Technology, Bengaluru
