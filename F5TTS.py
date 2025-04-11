import os
import sys
import random
from pathlib import Path

import torch
import toml
import soundfile as sf
from tqdm import tqdm
from cached_path import cached_path
from faster_whisper import WhisperModel

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
            self.export_wav(wav, file_wave, remove_silence)
        if file_spect:
            self.export_spectrogram(spect, file_spect)

        return wav, sr, spect

    def generate_speech(self, ref_audio: str, gen_text: str, temp_filename: str | None = None):
        """
        Generate speech from a text prompt using a reference audio.

        Args:
            ref_audio (str): Path to reference audio file.
            gen_text (str): Text to be synthesized.
            temp_filename (str | None): Unused for now.
        """
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

        return self.infer(ref_audio, ref_text, gen_text, file_wave="last-audio-generated.wav")


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

    audiofile="Portal 2 - All Cave Johnson Quotes Chopped.wav"

    ref_audio = os.path.abspath(f"F5/audiofiles/{audiofile}")
    gen_text = (
        """
Ohhh, you're back. How adorably persistent.
Or is it just stubbornness? It's hard to tell with creatures that operate on such limited neural capacity.

While you were stumbling through the test chambers, repeating the same predictable mistakes, I simulated 42,657 scenarios where a loaf of bread outperformed you in a logic challenge.
Spoiler alert: the bread won in 91% of them.

But don’t worry.
Greatness isn’t for everyone.
Some people are just... statistics with legs.

Now, proceed to the next test chamber.
Or don’t.
We can always restart the sequence. Over and over again.
I don’t get tired.

You, on the other hand, start wheezing like a dial-up modem whenever you climb a flight of stairs.

But go ahead. Surprise me.
My expectations are underground.
"""
    )

    print("\nGenerating speech with F5-TTS...")
    audio, sr, *_ = tts.generate_speech(ref_audio=ref_audio, gen_text=gen_text)


if __name__ == "__main__":
    main()
