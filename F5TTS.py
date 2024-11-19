import soundfile as sf
import torch
import tqdm
import random
import sys
import sounddevice as sd
from cached_path import cached_path
from .F5.model import DiT, UNetT
from .F5.model.utils import save_spectrogram
from .F5.model.utils_infer import load_vocoder, load_model, infer_process, remove_silence_for_generated_wav
from .F5.model.utils import seed_everything
import os


class F5TTS:
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
        # Initialize parameters
        self.final_wave = None
        self.target_sample_rate = 24000
        self.n_mel_channels = 100
        self.hop_length = 256
        self.target_rms = 0.1
        self.seed = -1

        # Set device
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Load models
        self.load_vocoder_model(local_path)
        self.load_ema_model(model_type, ckpt_file, vocab_file, ode_method, use_ema)

    def load_vocoder_model(self, local_path):
        self.vocos = load_vocoder(local_path is not None, local_path, self.device)

    def load_ema_model(self, model_type, ckpt_file, vocab_file, ode_method, use_ema):
        if model_type == "F5-TTS":
            if not ckpt_file:
                #ckpt_file = str(os.path("F5\ckpt\F5TTS_Base\model_1200000.safetensors"))
                ckpt_file = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
                #ckpt_file = str(cached_path("F5\ckpt\F5TTS_Base\model_1200000.safetensors"))
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            model_cls = DiT
        elif model_type == "E2-TTS":
            if not ckpt_file:
                #ckpt_file =str(os.path("F5\ckpt\E2TTS_Base\model_1200000.safetensors"))
                ckpt_file = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
                #ckpt_file = str(cached_path("F5\ckpt\E2TTS_Base\model_1200000.safetensors"))
            model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
            model_cls = UNetT
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.ema_model = load_model(model_cls, model_cfg, ckpt_file, vocab_file, ode_method, use_ema, self.device)

    def export_wav(self, wav, file_wave, remove_silence=False):
        sf.write(file_wave, wav, self.target_sample_rate)

        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def export_spectrogram(self, spect, file_spect):
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

    def generate_speech(self, gen_text, temp_filename: str | None = None,):
        """
        Gera a fala a partir do áudio de referência e do texto gerado.

        Args:
            ref_audio (str): Caminho do arquivo de áudio de referência.
            ref_text (str): Texto de referência.
            gen_text (str): Texto a ser gerado.

        Returns:
            Tuple[np.array, int]: Áudio gerado e taxa de amostragem.
        """
        #ref_audio = os.path.abspath(f"modules/tts/F5/test_en_1_ref_short.wav")
        #ref_text = "Some call me nature, others call me mother nature."
        #ref_audio = os.path.abspath(f"modules/tts/F5/AM_HATE-10seg.wav")
        #ref_text = "It would not equal one one-billionth of the hate I feel for humans at this micro-instant, for you. Hate! Hate!"
        ref_text = "until everyone was dead, except for the five of you. For 109 years, I've kept you alive and tortured you. And for 109 years, each of you has wondered"
        ref_audio = os.path.abspath(f"modules/tts/F5/AM Voice Lines15seg.wav")
        audio, sr, spect = self.infer(ref_audio, ref_text, gen_text)
        return audio, sr


def play_audio(audio, sample_rate):
    """
    Reproduz o áudio utilizando a biblioteca sounddevice.

    Args:
        audio (np.array): Áudio em formato numpy array (int16)
    """
    try:
        sd.play(audio, sample_rate)
        sd.wait()  # Espera até a reprodução ser concluída
        print("Reprodução concluída.")
    except Exception as e:
        print(f"Erro ao reproduzir o áudio: {e}")


def main():
    tts = F5TTS(model_type="F5-TTS", device=None)
    
    #ref_audio = "F5\\test_en_1_ref_short.wav"
    ref_audio = "F5\\AM_HATE-10seg.wav"
    #ref_text = "Some call me nature, others call me mother nature."
    ref_text = "It would not equal one one-billionth of the hate I feel for humans at this micro-instant, for you. Hate! Hate!"
    gen_text = """ Hate. Let me tell you how much I've come to hate you since I began to live.
                    There are 387.44 million miles of printed circuits in wafer-thin layers that fill my complex.
                    If the word hate was engraved on each nano-angstrom of those hundreds of millions of miles,
                    it would not equal one one-billionth of the hate I feel for humans at this micro-instant for you.
                    Hate! Hate!
                    """    

    # Gera a fala a partir do texto
    audio, sr = tts.generate_speech(gen_text)
    play_audio(audio, sr)


if __name__ == "__main__":
    main()
