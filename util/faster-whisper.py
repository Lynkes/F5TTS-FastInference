from faster_whisper import WhisperModel
import os
import tkinter as tk
from tkinter import filedialog
from colorama import Fore, Style

model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

SUPPORTED_AUDIO_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm', '.aac', '.wma', '.mp4')

def format_duration(seconds):
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"

def select_folder():
    folder_path = filedialog.askdirectory(title="Selecione a Pasta com os Áudios")
    if folder_path:
        print(Style.BRIGHT + Fore.GREEN)
        print(f"Pasta selecionada: {folder_path}")
        return folder_path
    else:
        print(Style.BRIGHT + Fore.RED)
        print("Nenhuma pasta selecionada.")
        return None

def process_files(folder_path):
    if not folder_path:
        return

    files = os.listdir(folder_path)
    audio_files = [file for file in files if file.lower().endswith(SUPPORTED_AUDIO_EXTENSIONS)]

    if not audio_files:
        print(Style.BRIGHT + Fore.YELLOW)
        print("Nenhum arquivo de áudio compatível encontrado na pasta.")
        return

    for audio_file in audio_files:
        audio_path = os.path.join(folder_path, audio_file)
        print(Style.BRIGHT + Fore.MAGENTA)
        print(f"Processando arquivo: {audio_path}")

        segments, info = model.transcribe(audio_path, beam_size=5)
        print(Style.BRIGHT + Fore.GREEN)
        print("Idioma detectado: '%s' (probabilidade: %f)" % (info.language, info.language_probability))

        filename, _ = os.path.splitext(os.path.basename(audio_path))
        output_file_path = os.path.join(folder_path, f"{filename}.srt")

        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for segment in segments:
                output_file.write(f"{segment.id}\n")
                output_file.write(f"{format_duration(segment.start)} --> {format_duration(segment.end)}\n")
                output_file.write(f"{segment.text.strip()}\n\n")
                print(Style.BRIGHT + Fore.WHITE)
                print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    print(Style.BRIGHT + Fore.GREEN)
    print("TODOS os arquivos .srt foram gerados.")
    print(Style.BRIGHT + Fore.RESET)

def ui():
    root = tk.Tk()
    root.title("Gerador de Legendas para Áudios")

    select_button = tk.Button(root, text="Selecionar Pasta e Gerar Legendas", command=lambda: process_files(select_folder()))
    select_button.pack(padx=20, pady=20)

    root.mainloop()

# Inicia a interface
ui()
