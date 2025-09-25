import librosa
import numpy as np
import sounddevice as sd
import time

# --- Load audio ---
filename = "test.mp3"
y, sr = librosa.load(filename, sr=None)

# Frame settings
hop_length = 2048
frame_duration = hop_length / sr

# --- Precompute pitch and chroma ---
pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length)
chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

print(f"timestamp, melody, chord")

# --- Playback and analysis ---
def callback(outdata, frames, time_info, status):
    global frame
    if status:
        print(status)

    # Send chunk to sound output
    start = frame * hop_length
    end = start + frames
    chunk = y[start:end]

    if len(chunk) < frames:
        outdata[:len(chunk), 0] = chunk
        raise sd.CallbackStop()
    else:
        outdata[:, 0] = chunk

    # --- Melody ---
    idx = magnitudes[:, frame].argmax()
    pitch = pitches[idx, frame]
    if pitch > 0:
        note = librosa.hz_to_note(pitch)
    else:
        note = "-"

    # --- Chord root guess ---
    chord_vector = chroma[:, frame]
    root = note_names[np.argmax(chord_vector)]

    chord_vector = chroma[:, frame]
    topN = 5 # five fingers
    top_idx = np.argsort(chord_vector)[::-1][:topN]  # sort descending
    top_notes = [(note_names[i], float(chord_vector[i])) for i in top_idx]

    # Print in CSV-like format
    print(f"{frame*frame_duration:5.2f},{note:3}," +
          " ".join([f"{n}:{v:.2f}" for n, v in top_notes]))


    frame += 1

# Initialize frame index
frame = 0

# --- Play + analyze ---
with sd.OutputStream(channels=1, samplerate=sr, callback=callback, blocksize=hop_length):
    sd.sleep(int(len(y) / sr * 1000))
