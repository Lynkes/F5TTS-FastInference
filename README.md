# 🎙️ F5TTS (Modified)

**F5TTS** has been modified to **stream audio as it's generated**, instead of waiting for the entire inference to complete. This allows for much faster and more interactive voice feedback.

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install dependencies

#### Install PyTorch with CUDA 12.4 support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### Install the remaining requirements:

```bash
pip install -r requirements.txt
```

---

## 🎞️ Install FFmpeg

FFmpeg is required for audio handling.

Download it from:  
🔗 [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)

- Extract the files
- Add the `bin/` folder to your system’s `PATH` environment variable

---

## ⚙️ CUDA Toolkit DLLs

To ensure GPU acceleration works properly, additional DLLs might be needed.

1. Download the precompiled **CUDA DLLs** from:  
   🔗 [https://github.com/Purfview/whisper-standalone-win/releases/tag/libs](https://github.com/Purfview/whisper-standalone-win/releases/tag/libs)

2. Extract the contents directly into the root folder of the cloned repository.

---

## ✅ You're Ready!

Run the main script and F5TTS will now generate and play speech **on-the-fly**, chunk by chunk — no more waiting for the full audio to render before playback.

---
