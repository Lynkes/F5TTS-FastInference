import os
import re
import toml
import wave
import ffmpeg
import tempfile
import numpy as np
from ollama import Client
from tkinter import filedialog, Tk
from faster_whisper import WhisperModel
import torch

# Configurações
MIN_DUR = 5
MAX_DUR = 15
SILENCE_DURATION = 0.13
ENERGY_SILENCE = 10         # valor empírico para silêncio leve
ENERGY_ZERO_DB = 1.0       # silêncio total (~0 dB)
FRAME_SIZE = 32  # resolução do sample (quanto menor, mais detalhado
OUTPUT_DIR = "output_chunksV2"

# Cliente do Ollama
ollama_client = Client(host='http://localhost:11434')

# Selecionar arquivo de áudio
Tk().withdraw()
print("Select an audio file")
audio_path = filedialog.askopenfilename(title="Select an audio file")
if not audio_path:
    raise Exception("No file selected.")

# Preparar caminho do WAV mono 16kHz
basename = os.path.splitext(os.path.basename(audio_path))[0]
os.makedirs(OUTPUT_DIR, exist_ok=True)
wav_path = os.path.join(tempfile.gettempdir(), f"{basename}_mono.wav")
ffmpeg.input(audio_path).output(wav_path, ac=1, ar=16000).overwrite_output().run()

# Carregar modelo Whisper
model = WhisperModel("large-v3", device="cuda" if torch.cuda.is_available() else "cpu")

def transcribe_with_faster_whisper(path):
    segments, _ = model.transcribe(path, language="en", task="transcribe")
    return " ".join([seg.text for seg in segments])

def get_frame_energies(path, frame_size=FRAME_SIZE):
    with wave.open(path, 'rb') as wf:
        n_frames = wf.getnframes()
        framerate = wf.getframerate()
        duration = n_frames / framerate

        energies = []
        while True:
            frame = wf.readframes(frame_size)
            if not frame:
                break
            samples = np.frombuffer(frame, dtype=np.int16)
            energy = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
            energies.append(energy)
        return energies, framerate, duration

# Análise de energia por frame
energies, framerate, duration = get_frame_energies(wav_path)
frame_duration = FRAME_SIZE / framerate
silence_frames = int(SILENCE_DURATION / frame_duration)

cuts = [0.0]
last_cut = 0.0
i = 0

print("\n📊 Energy levels:")

while i < len(energies):
    t = i * frame_duration
    energy = energies[i]
    #print(f"[{t:.2f}s] Energy: {energy:.2f}")

    time_since_last_cut = t - last_cut

    if time_since_last_cut >= MIN_DUR:
        window = energies[i:i+silence_frames]
        if len(window) == silence_frames and all(e < ENERGY_SILENCE for e in window):
            cuts.append(t)
            last_cut = t
            i += silence_frames
            continue

    if time_since_last_cut >= MAX_DUR:
        future_window = energies[i:i+silence_frames]
        if len(future_window) == silence_frames and all(e < ENERGY_ZERO_DB for e in future_window):
            cuts.append(t)
            last_cut = t
            i += silence_frames
            continue

    i += 1

if cuts[-1] < duration:
    cuts.append(duration)

# Processar e salvar os cortes
for i in range(len(cuts) - 1):
    start = cuts[i]
    end = cuts[i + 1]
    duration_chunk = end - start

    if duration_chunk < 1:
        continue

    temp_out = os.path.join(tempfile.gettempdir(), f"chunk_{i}.wav")
    ffmpeg.input(wav_path, ss=start, t=duration_chunk).output(temp_out).overwrite_output().run()

    text = transcribe_with_faster_whisper(temp_out)
    if not text.strip():
        continue

    prompt = f"""
Analyze the following spoken audio segment. It is part of a dataset for training an AI to detect emotional expression in speech.

1. Is this segment relevant for training emotional speech detection? Answer with "yes" or "no".
2. If yes, what is the predominant emotion in one word? (e.g., happy, sad, anger, neutral, excited)

Transcript:
\"\"\"{text}\"\"\"

Respond in the following format:

Relevant: yes or no  
Emotion: one_word_emotion_or_blank
"""
    response = ollama_client.chat(model="llama3:instruct", messages=[{"role": "user", "content": prompt}])
    content = response['message']['content'].strip().lower()

    is_relevant = "yes" in content
    match = re.search(r"emotion\s*[:\-]?\s*(\w+)", content.replace("\n", " "), re.IGNORECASE)
    emotion = match.group(1) if match else "neutral"

    if not is_relevant:
        continue

    emotion_folder = os.path.join(OUTPUT_DIR, emotion)
    os.makedirs(emotion_folder, exist_ok=True)

    prefix = "ideal_" if 5 <= duration_chunk <= 10 else "clip_"
    filename = f"{prefix}{i:03}_{int(duration_chunk)}s.wav"
    output_path = os.path.join(emotion_folder, filename)

    ffmpeg.input(wav_path, ss=start, t=duration_chunk).output(output_path).overwrite_output().run()

    toml_data = {
        "audio": filename,
        "start": start,
        "end": end,
        "duration": duration_chunk,
        "emotion": emotion,
        "transcription": text,
        "relevant": True
    }
    with open(output_path.replace(".wav", ".toml"), "w", encoding="utf-8") as f:
        toml.dump(toml_data, f)

    print(f"✅ Saved: {output_path} ({emotion}, {duration_chunk:.2f}s)")

print("\n🎯 Finished splitting and labeling audio.")
