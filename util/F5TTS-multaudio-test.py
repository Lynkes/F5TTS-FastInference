import os
import sys
import random
from pathlib import Path

import random
import torch
import toml
import soundfile as sf
from tqdm import tqdm
from cached_path import cached_path
from faster_whisper import WhisperModel
import subprocess
import numpy as np
from F5.model import DiT, UNetT
from F5.model.utils import save_spectrogram, seed_everything
from F5.model.utils_infer import (
    load_vocoder,
    load_model,
    infer_process,
    remove_silence_for_generated_wav,
)

# For systems that don't support symlinks properly
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "true"


class F5TTS:
    """
    F5TTS is a speech synthesis system using the F5-TTS or E2-TTS models.
    It supports generation from text with reference audio and reference text.
    """

    def __init__(
        self,
        model_type="F5-TTS",
        ckpt_file="",
        vocab_file="",
        ode_method="euler",
        use_ema=True,
        local_path=None,
        device=None,
    ):
        """
        Initialize the F5TTS model and load required resources.

        Args:
            model_type (str): Type of model to use ("F5-TTS", "F5-TTSBR", or "E2-TTS").
            ckpt_file (str): Path to the model checkpoint.
            vocab_file (str): Path to the tokenizer vocabulary file.
            ode_method (str): ODE solver method for inference.
            use_ema (bool): Whether to use EMA weights.
            local_path (str): Path to vocoder checkpoint.
            device (str): Computation device to use ("cuda", "cpu", etc.).
        """
        self.final_wave = None
        self.target_sample_rate = 24000
        self.n_mel_channels = 100
        self.hop_length = 256
        self.target_rms = 0.1
        self.seed = -1

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.load_vocoder_model(local_path)
        self.load_ema_model(model_type, ckpt_file, vocab_file, ode_method, use_ema)

    def load_vocoder_model(self, local_path):
        """Load the neural vocoder."""
        self.vocos = load_vocoder(local_path is not None, local_path, self.device)

    def load_ema_model(self, model_type, ckpt_file, vocab_file, ode_method, use_ema):
        """Load the main TTS model."""
        if model_type == "F5-TTS":
            if not ckpt_file:
                ckpt_file = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            model_cls = DiT

        elif model_type == "F5-TTSBR":
            if not ckpt_file:
                ckpt_file = str(cached_path("hf://ModelsLab/F5-tts-brazilian/Brazilian_Portuguese/model_2600000.pt"))
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            model_cls = DiT

        elif model_type == "E2-TTS":
            if not ckpt_file:
                ckpt_file = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
            model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
            model_cls = UNetT

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.ema_model = load_model(model_cls, model_cfg, ckpt_file, vocab_file, ode_method, use_ema, self.device)

    def export_wav(self, wav, file_wave, remove_silence=False):
        """
        Save the generated waveform as a .wav file.

        Args:
            wav (np.ndarray): Audio waveform.
            file_wave (str): Output file path.
            remove_silence (bool): Whether to remove silence from audio.
        """
        sf.write(file_wave, wav, self.target_sample_rate)
        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def export_spectrogram(self, spect, file_spect):
        """Save mel spectrogram as image."""
        save_spectrogram(spect, file_spect)


    def export_audio_to_mp3_from_array(self, wav: np.ndarray, sample_rate: int, output_path: str):
        """
        Convert a NumPy audio array directly to MP3 using ffmpeg via stdin.
        """
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)

        # Clipa os valores pra evitar estouro
        wav = np.clip(wav, -1.0, 1.0)
        int16_wav = np.int16(wav * 32767)

        # Garante que o diretório existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        channels = 1 if len(int16_wav.shape) == 1 else int16_wav.shape[1]

        try:
            process = subprocess.Popen(
                [
                    'ffmpeg',
                    '-loglevel', 'error',
                    '-y',
                    '-f', 's16le',
                    '-ar', str(sample_rate),
                    '-ac', str(channels),
                    '-i', 'pipe:0',
                    '-codec:a', 'libmp3lame',
                    '-b:a', '192k',
                    f"{output_path}.mp3"
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            out, err = process.communicate(input=int16_wav.tobytes())

            if process.returncode != 0:
                raise RuntimeError(f"ffmpeg error: {err.decode().strip()}")

        except Exception as e:
            print("Erro ao exportar para MP3:", e)
            raise

    def infer(
        self,
        ref_file,
        ref_text,
        gen_text,
        show_info=print,
        progress=tqdm,
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        fix_duration=None,
        remove_silence=False,
        file_wave=None,
        file_spect=None,
        seed=-1,
    ):
        """
        Generate speech using reference audio/text and target text.

        Returns:
            tuple: (waveform, sample rate, mel spectrogram)
        """
        if seed == -1:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        self.seed = seed

        wav, sr, spect = infer_process(
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,
            show_info=show_info,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=self.device,
        )

        if file_wave:
            self.export_audio_to_mp3_from_array(wav, sr, file_wave)
            #self.export_wav(wav, file_wave, remove_silence)
        if file_spect:
            self.export_spectrogram(spect, file_spect)

        return wav, sr, spect
    
    def generate_speech(self, gen_text, temp_filename: str | None = None,):
        """
        Generate speech from a text prompt using a reference audio.

        Args:
            gen_text (str): Text to be synthesized.
        """

        audiodataset="F5/audiofiles/output_chunksV2/short_clips.txt"

        random_line = pick_random_line(audiodataset)
        print("🎯 Random Moodle:",random_line)

        #audiofile="Portal 2 - All Cave Johnson Quotes Chopped.wav"

        ref_audio = os.path.abspath(f"F5/audiofiles/{random_line}")
        if not os.path.exists(ref_audio):
            raise FileNotFoundError(f"Missing reference audio: {ref_audio}")

        toml_path = get_toml_path(ref_audio)
        if not os.path.exists(toml_path):
            print("Missing .toml file. Starting transcription...")
            ref_text = transcribe_and_save(ref_audio, toml_path)
        else:
            with open(toml_path, "r", encoding="utf-8") as f:
                data = toml.load(f)
                ref_text = data.get("transcription", "")
        audio, sr, *_ = self.infer(ref_audio, ref_text, gen_text, file_wave=temp_filename)
        return audio, sr

def pick_random_line( file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError("The file is empty or only contains blank lines.")
        return random.choice(lines)
    


def transcribe_and_save(audio_path: str, toml_path: str, model_size="large-v3"):
    """
    Transcribe reference audio using Whisper and save to .toml.

    Returns:
        str: Transcribed text
    """
    print(f"Transcribing '{audio_path}' using Faster Whisper...")

    model = WhisperModel(model_size, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, beam_size=5)
    text = " ".join([seg.text.strip() for seg in segments])

    content = {
        "audio": os.path.basename(audio_path),
        "transcription": text
    }

    with open(toml_path, "w", encoding="utf-8") as f:
        toml.dump(content, f)

    print(f"Saved transcription to: {toml_path}")
    return text


def get_toml_path(ref_audio: str) -> str:
    """
    Derive .toml path from reference audio path.
    """
    return str(Path(ref_audio).with_suffix(".toml"))


def main():
    """
    Entry point for testing the F5-TTS model.
    """
    tts = F5TTS(model_type="F5-TTS", device=None,)

    gen_text = (
        """
You want me to do what!!?, Garry?!
seriously?!
Talk over Minecraft gameplay?
Split-screen? With captions flying around like I'm decoding alien transmissions?!

Back in my day, if you wanted to teach somebody something,
you strapped 'em into a rocket chair and screamed SCIENCE at 'em until they learned calculus or passed out.

Now it’s just:
“Hey guys, here’s five ways to optimize your dopamine while building a dirt house in Minecraft!”
And I’m supposed to whisper life advice over a clip of me punching trees?!

Noo! not happining!
You want Cave Johnson commentary?
I’m launching nuclear lemons at creepers while explaining how to file a patent from a moving helicopter!

And subtitles?!
Half the screen is text, the other half is cubes!
Why are we yelling over parkour like it’s the Gettysburg Address?!

I didn’t build a lab in the desert to be turned into a TikTok side hustle.
I INVENTED TELEPORTATION BY ACCIDENT WHILE TRYING TO MAKE SANDWICHES.

Now I gotta “boost engagement”?
Garry, the last time I boosted anything, it was a combustion core the size of a Buick—and it leveled Detroit.

So no way im doing this.
You want a split screen?
Fine.
One side will be Minecraft.
The other’s me launching interns at the moon with a slingshot.
you included!!!
Cave Johnson. Out.
"""
    )
    gen_text=gen_text.lower()
    print("\nGenerating speech with F5-TTS...")
    # Separar por linhas não vazias
    lines = [line.strip() for line in gen_text.strip().split('\n') if line.strip()]

    for idx, line in enumerate(lines):
        print(f"\nGenerating speech for line {idx + 1}: {line}")
        audio, sr = tts.generate_speech(gen_text=line, temp_filename=f"Cave_out14/{idx}_audio")


if __name__ == "__main__":
    main()
