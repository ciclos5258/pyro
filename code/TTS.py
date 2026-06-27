# tts_engine.py
import torch
import soundfile as sf
import sounddevice as sd
import numpy as np
import os

# ------------------ Настройки ------------------
SAMPLE_RATE = 24000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = None  # Глобальный объект модели, чтобы загружать только один раз

# --------------- Внутренняя загрузка ---------------
def _get_model():
    global MODEL
    if MODEL is None:
        os.environ.setdefault("TORCH_HOME", "./models")  # укажите свою папку
        MODEL, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='ru',
            speaker='v5_ru'
        )
        MODEL.to(DEVICE)
        print(f"[TTS] Модель загружена на {DEVICE}")
    return MODEL

# --------------- Публичные функции ---------------
def text_to_speech(text: str, speaker: str = "eugene") -> torch.Tensor:
    """
    Генерирует аудио из текста.
    Возвращает torch.Tensor с частотой 24000 Гц на целевом устройстве (CPU/GPU).
    """
    model = _get_model()
    audio = model.apply_tts(
        text=text,
        speaker=speaker,
        sample_rate=SAMPLE_RATE,
        put_accent=True,
        put_yo=True
    )
    return audio

def play_audio(audio_tensor: torch.Tensor, sample_rate: int = SAMPLE_RATE):
    """
    Воспроизводит тензор аудио через динамики/наушники.
    """
    audio_np = audio_tensor.detach().cpu().numpy().astype(np.float32)
    sd.play(audio_np, samplerate=sample_rate)
    sd.wait()

def save_audio(audio_tensor: torch.Tensor, filename: str, sample_rate: int = SAMPLE_RATE):
    """
    Сохраняет тензор аудио в WAV-файл.
    """
    audio_np = audio_tensor.detach().cpu().numpy()
    sf.write(filename, audio_np, samplerate=sample_rate)
    print(f"[TTS] Аудио сохранено: {filename}")

def speak(text_for_speak):
    audio = text_to_speech(text_for_speak)
    play_audio(audio)
    save_audio(audio, "output.wav")