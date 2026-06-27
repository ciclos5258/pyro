import requests
import json
import time
import subprocess
import queue
import sys
import os
import asyncio
import webbrowser
import re
import threading
from collections import deque

import ollama
import vosk
import pyaudio

from russian_prompt import russian_prompt
from TTS import text_to_speech, play_audio, save_audio, speak
from functionReading import CommandParser
from research import web_search

# Глобальная блокировка для потокобезопасной печати
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """Потокобезопасный вывод в консоль."""
    with print_lock:
        print(*args, **kwargs)

main = True

speak("Подключаю все системы")

class pyroQwen:
    def start(self):
        try:
            requests.get('http://localhost:11434', timeout=2)
            safe_print("Server already started")
        except:
            safe_print("Try to start server")
            self.ollama_process = subprocess.Popen(
                ['ollama', 'serve'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            time.sleep(3)
            safe_print("Server was started")

    def restart(self, voiceRecorder):
        safe_print("Restarting PYRO system...")

        if hasattr(self, 'ollama_process') and self.ollama_process is not None:
            self.ollama_process.terminate()
            time.sleep(3)
        try:
            requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt":"",
                    "keep_alive": 0
                }
            )
        except:
            pass

        self.start()
        self.conversation_history = []

        try:
            self.process("Привет", voiceRecorder)
            safe_print("Model was restart")
        except:
            safe_print("Warning of restart model")

    def __init__(self, voiceRecorder):
        self.model = "llama3.1:8b"
        self.url = "http://localhost:11434/api/chat"
        self.system_promt = russian_prompt
        self.running = True
        self.voiceRecorder = voiceRecorder
        self.conversation_history = []
        self.start()
        self.cmd_parser = CommandParser(examples_file="../commands_dataset.json")

    def check_connection(self):
        try:
            response = requests.get("http://localhost:11434/api/tags")
            return response.status_code == 200
        except:
            return False

    def think(self, user_input, voiceRecorder):
        messages = [
            {"role": "system", "content": self.system_promt}
        ]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_input})

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "repeat_penalty": 1.1,
                    "top_p": 0.9,
                    "presence_penalty": 0.5,
                    "stream": True
                }
            },
            timeout=120
        )

        response_text = response.json()["message"]["content"]

        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response_text})

        max_history = 20
        if len(self.conversation_history) > max_history * 2:
            self.conversation_history = self.conversation_history[-(max_history*2):]

        return response_text

    def process(self, user_input, voiceRecorder):
        intent = self.cmd_parser.parse(user_input)
        cmd = intent.get("command", "none")
        args = intent.get("args", "")

        if cmd == "shutdown":
            self.running = False
            return "Завершаю работу."
        elif cmd == "restart":
            self.restart(voiceRecorder)
            return self.think("Поприветствуй пользователя после перезагрузки", voiceRecorder)
        elif cmd == "record":
            safe_print("🎤 Начинаю запись голоса...")
            try:
                recorded_text = voiceRecorder.record()
                if recorded_text:
                    safe_print(f"✅ Распознано: {recorded_text}")
                    return self.process(recorded_text, voiceRecorder)
                else:
                    return "Не удалось распознать голосовую команду."
            except Exception as e:
                safe_print(f"Ошибка записи голоса: {e}")
                return "Произошла ошибка при записи голоса."
        elif cmd == "web":
            site_domen = args
            safe_print("Открываю " + site_domen)
            web_search(site_domen)
            safe_print(site_domen + " открыт!")
            return self.think(f"отчитайся об успешном выполнении {user_input}")
        else:
            return self.think(user_input, voiceRecorder)


class voiceRecorder:
    def __init__(self, model_path="../vosk-model-small-ru-0.22/vosk-model-small-ru-0.22"):
        safe_print(f"Загрузка модели распознавания речи...", end="", flush=True)
        try:
            stderr_backup = sys.stderr
            sys.stderr = open(os.devnull, 'w')

            self.model = vosk.Model(model_path)

            sys.stderr.close()
            sys.stderr = stderr_backup

            self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
            self.recognizer.SetWords(False)
            self.audio = None
            self.stream = None
            self.is_recording = False
            safe_print(" готово!")
        except Exception as e:
            if 'stderr_backup' in locals():
                sys.stderr.close()
                sys.stderr = stderr_backup
            safe_print(f"\nОшибка загрузки модели: {e}")
            raise

    def start_recording(self):
        if self.is_recording:
            self.stop_recording()

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=4000
        )
        self.is_recording = True
        safe_print("Запись началась...")

    def stop_recording(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        self.is_recording = False
        safe_print("Запись остановлена.")

    def record(self, timeout=5):
        try:
            self.start_recording()

            safe_print("Говорите... (тишина остановит запись)")
            result_text = ""
            silence_counter = 0
            max_silence_chunks = 50

            chunks_per_second = int(16000 / 4000)
            max_chunks = timeout * chunks_per_second

            for chunk_count in range(max_chunks):
                data = self.stream.read(4000, exception_on_overflow=False)

                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        result_text = text
                        safe_print(f"\rРаспознано: {result_text}")
                        silence_counter = 0
                else:
                    partial = json.loads(self.recognizer.PartialResult())
                    partial_text = partial.get("partial", "")
                    if partial_text:
                        safe_print(f"\rРаспознается: {partial_text}", end="", flush=True)
                    if not partial_text:
                        silence_counter += 1
                        if silence_counter > max_silence_chunks and result_text:
                            safe_print("\nТишина - завершаю запись.")
                            break
                    else:
                        silence_counter = 0

            self.stop_recording()

            if not result_text:
                final_result = json.loads(self.recognizer.FinalResult())
                result_text = final_result.get("text", "")

            return result_text

        except Exception as e:
            safe_print(f"Ошибка при записи: {e}")
            self.stop_recording()
            return ""

    def record_continuous(self):
        self.start_recording()
        safe_print("Непрерывная запись... (нажмите Ctrl+C для остановки)")

        try:
            while True:
                data = self.stream.read(4000, exception_on_overflow=False)
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        safe_print(f"\n[ФИНАЛ] {text}")
                else:
                    partial = json.loads(self.recognizer.PartialResult())
                    partial_text = partial.get("partial", "")
                    if partial_text:
                        safe_print(f"\r[ЧАСТЬ] {partial_text}", end="", flush=True)
        except KeyboardInterrupt:
            safe_print("\nЗапись прервана пользователем")
        finally:
            self.stop_recording()

    async def background_voice_listener(self, keyword="слушай", callback=None):
        safe_print(f"🎤 Фоновое прослушивание активно...")
        safe_print(f"Скажите '{keyword}' для активации")
        safe_print("-" * 50)

        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=4000
        )

        recognizer = vosk.KaldiRecognizer(self.model, 16000)
        audio_buffer = []
        buffer_size = 30

        try:
            while True:
                data = stream.read(4000, exception_on_overflow=False)
                audio_buffer.append(data)

                if len(audio_buffer) > buffer_size * 2:
                    audio_buffer = audio_buffer[-buffer_size:]

                if len(audio_buffer) >= buffer_size:
                    combined_audio = b''.join(audio_buffer[-buffer_size:])

                    if recognizer.AcceptWaveform(combined_audio):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "").lower()

                        if keyword in text:
                            safe_print(f"\n🔊 Ключевое слово '{keyword}' распознано!")
                            command = await self._record_command_async(stream, recognizer)
                            if command and callback:
                                await callback(command)
                            recognizer.Reset()
                            audio_buffer.clear()

                await asyncio.sleep(0.01)

        except KeyboardInterrupt:
            safe_print("\n⏹️ Остановка фонового прослушивания")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

    async def _record_command_async(self, stream, recognizer, timeout=10):
        safe_print("🎙️ Слушаю команду... (тишина автоматически остановит)")
        result_text = ""
        silence_counter = 0
        max_silence_chunks = 25

        for _ in range(timeout * 4):
            data = stream.read(4000, exception_on_overflow=False)

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
                    result_text = text
                    safe_print(f"\r✅ Распознано: {result_text}")
                    silence_counter = 0
            else:
                partial = json.loads(recognizer.PartialResult())
                partial_text = partial.get("partial", "")
                if partial_text:
                    safe_print(f"\r🎤 Распознается: {partial_text}", end="", flush=True)
                    silence_counter = 0
                else:
                    silence_counter += 1

                if silence_counter > max_silence_chunks and result_text:
                    safe_print("\n⏸️ Тишина - завершаю запись.")
                    break

            await asyncio.sleep(0.01)

        if not result_text:
            final_result = json.loads(recognizer.FinalResult())
            result_text = final_result.get("text", "")
            if result_text:
                safe_print(f"✅ Распознано: {result_text}")

        return result_text

    def __del__(self):
        if self.is_recording:
            self.stop_recording()


async def voice_command_handler(command):
    global pyro, recorder
    safe_print(f"\n📝 Голосовая команда: {command}")
    response = pyro.process(command, recorder)
    safe_print(f"🤖 {response}")
    speak(response)
    # Печатаем приглашение после обработки голосовой команды
    safe_print("\n> ", end="", flush=True)


async def main():
    global pyro, recorder

    recorder = voiceRecorder("../vosk-model-small-ru-0.22/vosk-model-small-ru-0.22")
    pyro = pyroQwen(recorder)

    safe_print("Welcome back.")
    safe_print("-"*38)

    response = pyro.process("Привет", recorder)
    safe_print(response)
    speak(response)

    safe_print("-"*38)
    speak("Система подгружена.")

    # Первое приглашение для текстового ввода
    safe_print("\n> ", end="", flush=True)

    listener_task = asyncio.create_task(
        recorder.background_voice_listener(
            keyword="слушай",
            callback=voice_command_handler
        )
    )

    loop = asyncio.get_event_loop()

    def text_input_loop():
        while pyro.running:
            # Ожидаем ввод (без печати приглашения – оно печатается после каждой команды)
            data = sys.stdin.readline()
            if data:
                data = data.strip()
                if data.lower() in ['выход', 'exit', 'quit']:
                    pyro.running = False
                    break
                asyncio.run_coroutine_threadsafe(
                    process_text_command(data), loop
                )

    async def process_text_command(text):
        if text:
            safe_print("-"*38)
            response = pyro.process(text, recorder)
            safe_print(response)
            speak(response)
            safe_print("-"*38)
            # Печатаем приглашение для следующей команды
            safe_print("\n> ", end="", flush=True)

    await loop.run_in_executor(None, text_input_loop)
    await listener_task

    safe_print("Ассистент завершил работу.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        safe_print("\nПрограмма прервана пользователем")