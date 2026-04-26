import torch
import soundfile as sf
import sounddevice as sd
from silero import silero_tts

import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется: {device}")

# Path to the model directory
os.environ["TORCH_HOME"] = "./models"
sample_rate = 24000

# Load the model

model, example_text = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='ru',
    speaker='v5_ru'
)
model.to(device)
print("Model loaded successfully.")

text = "Привет, как дела? Это тестовая фраза для генерации речи."

# speech generate

audio = model.apply_tts(

    text=text, 
    speaker="eugene",
    sample_rate=sample_rate,
    put_accent=True,
    put_yo=True

    )

def play_audio(audio_tensor, sample_rate):
    audio_numpy = audio_tensor.detach().cpu().numpy().astype('float32')
    sd.play(audio_numpy, sample_rate)
    sd.wait()

play_audio(audio, sample_rate)

# save to wav
sf.write("output.wav", audio.cpu().numpy(), sample_rate)
print("Audio saved as output.wav")