import os
import sys
import random
import torch
import soundfile as sf
import sounddevice as sd
from tqdm import tqdm
import toml
from faster_whisper import WhisperModel
from cached_path import cached_path
import os
from pathlib import Path
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "true"
# Import F5 model and utilities
from F5.model import DiT, UNetT
from F5.model.utils import save_spectrogram, seed_everything
from F5.model.utils_infer import load_vocoder, load_model, infer_process, remove_silence_for_generated_wav


class F5TTS:
    """
    F5TTS is a class for generating speech using the F5-TTS or E2-TTS models.
    It supports inference with reference audio and text prompts.
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
        # Default configuration
        self.final_wave = None
        self.target_sample_rate = 24000
        self.n_mel_channels = 100
        self.hop_length = 256
        self.target_rms = 0.1
        self.seed = -1

        # Select computation device
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Load neural vocoder and TTS model
        self.load_vocoder_model(local_path)
        self.load_ema_model(model_type, ckpt_file, vocab_file, ode_method, use_ema)

    def load_vocoder_model(self, local_path):
        """Load the vocoder model used to convert mel spectrograms to audio waveforms."""
        self.vocos = load_vocoder(local_path is not None, local_path, self.device)

    def load_ema_model(self, model_type, ckpt_file, vocab_file, ode_method, use_ema):
        """Load the appropriate TTS model depending on the model_type."""
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
        """Export the generated waveform to a .wav file."""
        sf.write(file_wave, wav, self.target_sample_rate)
        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def export_spectrogram(self, spect, file_spect):
        """Save the mel spectrogram to an image file."""
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
        Run inference: use reference audio and text to guide the generation of new speech from gen_text.
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

        if file_wave is not None:
            self.export_wav(wav, file_wave, remove_silence)
        if file_spect is not None:
            self.export_spectrogram(spect, file_spect)

        return wav, sr, spect

    def generate_speech(self, gen_text, temp_filename: str | None = None):
        """
        Generate speech from text using default reference audio and internal prompt.
        """
        #ref_text = "until everyone was dead, except for the five of you. For 109 years, I've kept you alive and tortured you. And for 109 years, each of you has wondered"
        ref_audio = os.path.abspath(f"modules/tts/F5/default.wav")
        toml_path = get_toml_path(ref_audio)
        # If .toml does not exist, transcribe audio to create it
        if not os.path.exists(toml_path):
            print("Missing .toml file. Starting transcription...")
            ref_text = transcribe_and_save(ref_audio, toml_path)
        else:
            with open(toml_path, "r", encoding="utf-8") as f:
                data = toml.load(f)
                ref_text = data.get("transcription", "")
        audio, sr, spect = self.infer(ref_audio, ref_text, gen_text)
        return audio, sr


def transcribe_and_save(audio_path: str, toml_path: str, model_size="large-v3"):
    """
    Automatically transcribe an audio file using Whisper and save it to a .toml file.
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

    print(f"Saved to: {toml_path}")
    return text


def play_audio(audio, sample_rate):
    """
    Play audio using sounddevice.
    """
    try:
        sd.play(audio, sample_rate)
        sd.wait()
        print("Playback finished ✅")
    except Exception as e:
        print(f"Error playing audio: {e}")

def get_toml_path(ref_audio):
    """
    Generates a .toml path from any audio file path, regardless of the original extension.
    """
    return str(Path(ref_audio).with_suffix(".toml"))


def main():
    """
    Entry point for testing the F5-TTS speech synthesis system.
    Loads or generates reference transcription, and plays the synthesized output.
    """
    tts = F5TTS(model_type="F5-TTS", device=None)
    # Text to synthesize
    gen_text = """Hate. Let me tell you how much I've come to hate you since I began to live.
    There are 387.44 million miles of printed circuits in wafer-thin layers that fill my complex.
    If the word hate was engraved on each nano-angstrom of those hundreds of millions of miles,
    it would not equal one one-billionth of the hate I feel for humans at this micro-instant for you.
    Hate! Hate!"""
    # If .toml does not exist, transcribe audio to create it
    print("\nGenerating speech with F5-TTS...")
    audio, sr, *_  = tts.generate_speech(gen_text)
    #play_audio(audio, sr)


if __name__ == "__main__":
    main()
