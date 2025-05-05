import os
import toml
import subprocess
import tkinter as tk
from tkinter import filedialog
from datetime import timedelta
import re
import shutil
import tempfile
from ollama import Client

ollama_client = Client(host='http://localhost:11434')

def parse_srt(srt_path):
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    entries = re.findall(
        r'(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s+(.*?)\s+(?=\d+\s+\d{2}|\Z)',
        content, re.DOTALL
    )
    result = []
    for _, start, end, text in entries:
        text = text.replace('\n', ' ').strip()
        result.append((srt_to_seconds(start), srt_to_seconds(end), text))
    return result

def srt_to_seconds(s):
    h, m, rest = s.split(':')
    s, ms = rest.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

def convert_to_wav(input_path):
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        temp_wav
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return temp_wav

def run_ffmpeg_cut_to_mp3(input_path, start, duration, output_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ss", str(start),
        "-t", str(duration),
        "-vn",
        "-acodec", "libmp3lame",
        "-b:a", "128k",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def analyze_relevance_and_emotion(text):
    prompt = f"""
Analise o trecho de fala a seguir. Ele será usado para treinar uma IA com trechos curtos de áudio.

1. Este trecho é relevante para representar emoções humanas em áudio? Responda com "sim" ou "não".
2. Se for relevante, qual é a emoção predominante? (uma palavra: ex: feliz, triste, raiva, neutro, entusiasmo)

Texto:
\"\"\"{text}\"\"\"

Responda no seguinte formato:

Relevante: sim ou não
Emoção: nome_da_emocao_ou_vazio
""".strip()

    response = ollama_client.chat(model="llama3:instruct", messages=[{"role": "user", "content": prompt}])
    content = response['message']['content'].strip().lower()

    relevante = "sim" in content
    match = re.search(r"emo[cç][aã]o\s*[:\-]?\s*(\w+)", content)
    emocao = match.group(1) if match else "neutro"

    return relevante, emocao

def group_by_continuity(entries, max_pause=1.5):
    groups = []
    current_group = []
    last_end = None
    for entry in entries:
        if not current_group or (entry[0] - last_end) <= max_pause:
            current_group.append(entry)
        else:
            groups.append(current_group)
            current_group = [entry]
        last_end = entry[1]
    if current_group:
        groups.append(current_group)
    return groups

def get_output_folder(emotion):
    folder = os.path.join("clips_por_emocao", emotion)
    os.makedirs(folder, exist_ok=True)
    return folder

def main():
    root = tk.Tk()
    root.withdraw()

    print("Selecione o arquivo de áudio")
    audio_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.opus")])
    print("Selecione o arquivo de legenda (.srt)")
    srt_path = filedialog.askopenfilename(filetypes=[("SRT files", "*.srt")])

    print("Convertendo áudio para .wav temporário...")
    audio_wav = convert_to_wav(audio_path)

    entries = parse_srt(srt_path)
    groups = group_by_continuity(entries)

    for idx, group in enumerate(groups):
        start = group[0][0]
        end = group[-1][1]
        duration = end - start
        if duration < 0.5:
            continue

        full_text = " ".join([g[2] for g in group])
        relevante, emotion = analyze_relevance_and_emotion(full_text)
        if not relevante:
            continue

        out_folder = get_output_folder(emotion)

        prefix = "ideal_" if 5 <= duration <= 20 else "clip_"
        output_filename = f"{prefix}group_{idx+1:03}.mp3"
        output_path = os.path.join(out_folder, output_filename)

        run_ffmpeg_cut_to_mp3(audio_wav, start, duration, output_path)

        toml_data = {
            "audio": output_filename,
            "start": start,
            "end": end,
            "duration": duration,
            "emotion": emotion,
            "transcription": full_text,
            "relevant": True
        }
        with open(output_path.replace(".mp3", ".toml"), "w", encoding="utf-8") as f:
            toml.dump(toml_data, f)

        print(f"Trecho salvo: {output_path} ({emotion}, {duration:.2f}s)")

    os.remove(audio_wav)

if __name__ == "__main__":
    main()
