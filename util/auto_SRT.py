from faster_whisper import WhisperModel
import os
import tkinter as tk
from tkinter import ttk, filedialog
from colorama import Fore, Style

model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

def format_duration(seconds):
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"

def select_file():
    file_path = filedialog.askopenfilename(
        title="Select Audio or Video File",
        filetypes=[("Audio/Video Files", "*.mp3 *.mp4 *.wav *.m4a *.aac *.flac *.mov *.avi")]
    )
    if file_path:
        print(Style.BRIGHT + Fore.GREEN)
        print(f"Selected File: {file_path}")
        return file_path
    else:
        print(Style.BRIGHT + Fore.RED)
        print("No file selected.")
        return None

def process_file(file_path):
    if not file_path:
        return

    print(Style.BRIGHT + Fore.MAGENTA)
    print(f"Processing file: {file_path}")

    segments, info = model.transcribe(file_path, beam_size=5)
    print(Style.BRIGHT + Fore.GREEN)
    print(f"Detected language: {info.language} (probability {info.language_probability:.2f})")

    filename, _ = os.path.splitext(os.path.basename(file_path))
    output_file_path = os.path.join(os.path.dirname(file_path), f"{filename}.srt")

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for segment in segments:
            output_file.write(f"{segment.id}\n")
            output_file.write(f"{format_duration(segment.start)} --> {format_duration(segment.end)}\n")
            output_file.write(f"{segment.text}\n\n")
            print(Style.BRIGHT + Fore.WHITE)
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

    print(Style.BRIGHT + Fore.GREEN)
    print(f"SRT file generated: {output_file_path}")
    print(Style.RESET_ALL)

def ui():
    root = tk.Tk()
    root.title("Subtitle Generator")

    select_button = tk.Button(root, text="Select Audio/Video File", command=lambda: process_file(select_file()))
    select_button.pack(padx=20, pady=20)

    root.mainloop()

# Launch UI
ui()
