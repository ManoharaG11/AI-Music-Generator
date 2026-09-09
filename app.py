import streamlit as st
import numpy as np
import pickle
from pathlib import Path
from tensorflow.keras.models import load_model
from music21 import stream, note, chord, tempo


# ============================================================
# PROJECT PATHS
# ============================================================

MODEL_FILE = Path("models/music_generator.keras")
NOTE_TO_INT_FILE = Path("data/note_to_int.pkl")
INT_TO_NOTE_FILE = Path("data/int_to_note.pkl")
NOTES_FILE = Path("data/notes.pkl")

OUTPUT_FOLDER = Path("outputs")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_FOLDER / "generated_music.mid"

SEQUENCE_LENGTH = 50


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Music Generator",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- MAIN BACKGROUND ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(0, 220, 255, 0.12),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(150, 70, 255, 0.14),
                transparent 30%
            ),
            radial-gradient(
                circle at 50% 100%,
                rgba(0, 180, 255, 0.08),
                transparent 35%
            ),
            #080b14;
    }

    /* ---------- GENERAL TEXT ---------- */

    h1, h2, h3 {
        letter-spacing: -0.5px;
    }

    /* ---------- HERO TITLE ---------- */

    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 5px;
        background: linear-gradient(
            90deg,
            #00e5ff,
            #7c4dff,
            #ff4ecd
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        text-align: center;
        color: #aeb7c8;
        font-size: 1.05rem;
        margin-bottom: 30px;
    }

    /* ---------- METRIC CARDS ---------- */

    div[data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                rgba(20, 27, 45, 0.95),
                rgba(11, 15, 28, 0.95)
            );
        border: 1px solid rgba(0, 220, 255, 0.18);
        border-radius: 16px;
        padding: 18px;
        box-shadow:
            0 8px 30px rgba(0, 0, 0, 0.25);
        transition: 0.25s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: rgba(0, 220, 255, 0.45);
        box-shadow:
            0 10px 35px rgba(0, 220, 255, 0.10);
    }

    /* ---------- INPUTS ---------- */

    div[data-baseweb="select"] > div {
        background-color: #101625;
        border-color: #27334d;
        border-radius: 10px;
    }

    div[data-testid="stSlider"] {
        padding-top: 10px;
        padding-bottom: 10px;
    }

    /* ---------- BUTTON ---------- */

    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        font-weight: 700;
        background: linear-gradient(
            90deg,
            #00c6ff,
            #7c4dff
        );
        color: white;
        box-shadow:
            0 8px 25px rgba(80, 80, 255, 0.22);
        transition: all 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow:
            0 12px 30px rgba(0, 210, 255, 0.25);
    }

    /* ---------- DOWNLOAD BUTTON ---------- */

    .stDownloadButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 700;
        border: 1px solid rgba(0, 220, 255, 0.35);
        background: rgba(0, 220, 255, 0.08);
        color: #e8fbff;
    }

    /* ---------- INFO BOX ---------- */

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* ---------- CODE BLOCK ---------- */

    pre {
        border-radius: 14px !important;
        border: 1px solid rgba(124, 77, 255, 0.22) !important;
        background: #0b101d !important;
    }

    /* ---------- DIVIDERS ---------- */

    hr {
        border-color: rgba(255, 255, 255, 0.08);
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #737d91;
        padding: 25px 0 10px 0;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL ONLY ONCE
# ============================================================

@st.cache_resource
def load_music_model():
    return load_model(MODEL_FILE)


@st.cache_data
def load_music_data():

    with open(NOTE_TO_INT_FILE, "rb") as file:
        note_to_int = pickle.load(file)

    with open(INT_TO_NOTE_FILE, "rb") as file:
        int_to_note = pickle.load(file)

    with open(NOTES_FILE, "rb") as file:
        notes = pickle.load(file)

    return note_to_int, int_to_note, notes


# ============================================================
# GENERATE MUSIC
# ============================================================

def generate_music(
    model,
    note_to_int,
    int_to_note,
    notes,
    number_of_notes
):

    start = np.random.randint(
        0,
        len(notes) - SEQUENCE_LENGTH
    )

    pattern = notes[
        start:start + SEQUENCE_LENGTH
    ]

    generated_notes = []

    for _ in range(number_of_notes):

        input_sequence = [
            note_to_int[item]
            for item in pattern
        ]

        input_sequence = np.array(
            input_sequence
        )

        input_sequence = np.reshape(
            input_sequence,
            (1, SEQUENCE_LENGTH)
        )

        prediction = model.predict(
            input_sequence,
            verbose=0
        )

        predicted_index = np.argmax(
            prediction
        )

        predicted_note = int_to_note[
            predicted_index
        ]

        generated_notes.append(
            predicted_note
        )

        pattern.append(
            predicted_note
        )

        pattern = pattern[1:]

    return generated_notes


# ============================================================
# CONVERT NOTES TO MIDI
# ============================================================

def create_midi(generated_notes):

    music_stream = stream.Stream()

    music_stream.append(
        tempo.MetronomeMark(number=100)
    )

    for pattern_note in generated_notes:

        if "." in pattern_note:

            try:

                chord_notes = [
                    int(pitch)
                    for pitch in pattern_note.split(".")
                ]

                new_chord = chord.Chord(
                    chord_notes
                )

                new_chord.quarterLength = 0.5

                music_stream.append(
                    new_chord
                )

            except Exception:
                pass

        else:

            try:

                new_note = note.Note(
                    pattern_note
                )

                new_note.quarterLength = 0.5

                music_stream.append(
                    new_note
                )

            except Exception:
                pass

    music_stream.write(
        "midi",
        fp=OUTPUT_FILE
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    '<div class="hero-title">🎵 AI Music Generator</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-subtitle">
        Compose original musical sequences using an LSTM-powered
        deep learning model.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT STATS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🧠 AI Model",
        "LSTM"
    )

with col2:
    st.metric(
        "🎼 Sequence Length",
        "50"
    )

with col3:
    st.metric(
        "🎹 Output Format",
        "MIDI"
    )

with col4:
    st.metric(
        "⚡ Generation",
        "Real-Time"
    )


st.write("")


# ============================================================
# MAIN WORKSPACE
# ============================================================

left, right = st.columns(
    [1.15, 1],
    gap="large"
)


# ============================================================
# LEFT — GENERATION CONTROLS
# ============================================================

with left:

    st.subheader("🎛️ Generation Studio")

    st.write(
        "Configure how much music the AI should compose."
    )

    number_of_notes = st.slider(
        "Number of musical events",
        min_value=50,
        max_value=500,
        value=200,
        step=50
    )

    st.caption(
        f"🎼 The model will generate approximately "
        f"**{number_of_notes} musical events**."
    )

    st.write("")

    if st.button(
        "🎶 Generate Music",
        type="primary"
    ):

        if not MODEL_FILE.exists():

            st.error(
                "❌ Trained model not found."
            )

            st.stop()

        with st.spinner(
            "🧠 Loading trained LSTM model..."
        ):

            model = load_music_model()

            note_to_int, int_to_note, notes = (
                load_music_data()
            )

        with st.spinner(
            "🎼 Composing your music..."
        ):

            generated_notes = generate_music(
                model,
                note_to_int,
                int_to_note,
                notes,
                number_of_notes
            )

            create_midi(
                generated_notes
            )

        st.session_state[
            "generated_notes"
        ] = generated_notes

        st.session_state[
            "music_generated"
        ] = True


# ============================================================
# RIGHT — MODEL INFORMATION
# ============================================================

with right:

    st.subheader("🧠 About the AI")

    st.info(
        """
        **LSTM Neural Network**

        The model learns patterns from MIDI note sequences
        and predicts the next musical event based on the
        previous sequence.

        **Pipeline**

        MIDI → Notes → Sequences → LSTM → Prediction → MIDI
        """
    )

    st.write("")

    st.markdown(
        "**Technologies**"
    )

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.caption("🐍 Python")

    with tech2:
        st.caption("🧠 TensorFlow")

    with tech3:
        st.caption("🎼 music21")


# ============================================================
# GENERATED MUSIC SECTION
# ============================================================

if st.session_state.get(
    "music_generated",
    False
):

    generated_notes = st.session_state[
        "generated_notes"
    ]

    st.divider()

    st.subheader(
        "🎧 Your AI-Generated Music"
    )

    result_col1, result_col2 = st.columns(
        [1.4, 1],
        gap="large"
    )

    with result_col1:

        st.success(
            f"🎉 Music generated successfully! "
            f"{len(generated_notes)} musical events created."
        )

        st.write(
            "**Generated Sequence Preview**"
        )

        st.code(
            " → ".join(
                generated_notes[:50]
            ),
            language="text"
        )

    with result_col2:

        st.write(
            "**🎵 MIDI Output**"
        )

        st.write(
            "Your composition is ready."
        )

        with open(
            OUTPUT_FILE,
            "rb"
        ) as file:

            st.download_button(
                label="⬇️ Download Generated MIDI",
                data=file,
                file_name="generated_music.mid",
                mime="audio/midi"
            )

        st.caption(
            "Open the downloaded `.mid` file "
            "with any MIDI-compatible music player."
        )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.subheader("⚙️ How It Works")

step1, step2, step3, step4 = st.columns(4)

with step1:

    st.markdown("### 01 🎼")

    st.write("**MIDI Dataset**")

    st.caption(
        "Musical notes and chords are extracted "
        "from MIDI files."
    )

with step2:

    st.markdown("### 02 🧩")

    st.write("**Sequence Learning**")

    st.caption(
        "Note sequences are converted into "
        "training patterns."
    )

with step3:

    st.markdown("### 03 🧠")

    st.write("**LSTM Prediction**")

    st.caption(
        "The neural network learns musical "
        "patterns and predicts the next event."
    )

with step4:

    st.markdown("### 04 🎵")

    st.write("**MIDI Generation**")

    st.caption(
        "Predicted events are converted back "
        "into a playable MIDI composition."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🎵 AI Music Generator &nbsp;•&nbsp;
        Built with Python, TensorFlow, Keras, music21 & Streamlit
        <br><br>
        © 2026 · Developed by G MANOHARA
    </div>
    """,
    unsafe_allow_html=True
)