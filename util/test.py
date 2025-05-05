import os
import ffmpeg
import tempfile
import wave
from tkinter import filedialog, Tk

# Configurações
MIN_DUR = 4           # segundos
MAX_DUR = 35          # segurança: cortar mesmo que não encontre silêncio
SILENCE_DURATION = 0.5  # segundos de silêncio para considerar um ponto de corte
ENERGY_SILENCE = 4     # energia abaixo disso é considerado silêncio
OUTPUT_DIR = "output_chunks"

# Tkinter para selecionar arquivo
Tk().withdraw()
audio_path = filedialog.askopenfilename(title="Selecione o arquivo de áudio")
if not audio_path:
    raise Exception("Nenhum arquivo selecionado.")

basename = os.path.splitext(os.path.basename(audio_path))[0]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Converte para wav mono temporário
wav_path = os.path.join(tempfile.gettempdir(), f"{basename}_mono.wav")
ffmpeg.input(audio_path).output(wav_path, ac=1, ar=16000).overwrite_output().run()

# Extrai energia por frames
def get_frame_energies(path, frame_size=1024):
    with wave.open(path, 'rb') as wf:
        n_frames = wf.getnframes()
        framerate = wf.getframerate()
        duration = n_frames / framerate

        energies = []
        while True:
            frame = wf.readframes(frame_size)
            if not frame:
                break
            energy = sum(abs(b - 128) for b in frame) / len(frame)
            energies.append(energy)
        return energies, framerate, duration

energies, framerate, duration = get_frame_energies(wav_path)
frame_duration = 1024 / framerate
silence_frames = int(SILENCE_DURATION / frame_duration)

# Detectar cortes baseados em silêncio
cuts = [0.0]
last_cut = 0.0

i = 0
while i < len(energies):
    t = i * frame_duration

    # passou o mínimo?
    if t - last_cut >= MIN_DUR:
        window = energies[i:i+silence_frames]
        if len(window) == silence_frames:
            if all(e < ENERGY_SILENCE for e in window):
                cuts.append(t)
                last_cut = t
                i += silence_frames  # pula o silêncio
                continue

    # força corte se passou demais sem silêncio
    if t - last_cut >= MAX_DUR:
        cuts.append(t)
        last_cut = t
    i += 1

if cuts[-1] < duration:
    cuts.append(duration)

# Cortar com ffmpeg
for i in range(len(cuts) - 1):
    start = cuts[i]
    end = cuts[i + 1]
    duration_chunk = end - start

    if duration_chunk < 1:
        continue

    prefix = "OK_" if 5 <= duration_chunk <= 20 else "CUT_"
    output_path = os.path.join(OUTPUT_DIR, f"{prefix}chunk_{i:03}_{int(duration_chunk)}s.wav")

    ffmpeg.input(audio_path, ss=start, t=duration_chunk)\
        .output(output_path).overwrite_output().run()

    print(f"✅ Gerado: {output_path}")

print("\n🎯 Cortes finalizados com base em fim de fala (silêncio).")
