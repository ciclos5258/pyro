import requests
import json
import time
import subprocess
import queue
import sys 
import os
import asyncio
from collections import deque  # ДОБАВЛЕНО

import ollama
import vosk
import pyaudio

from russian_prompt import russian_prompt

main = True

class pyroQwen:
    def start(self):
        try:
            requests.get('http://localhost:11434', timeout=2)
            print("Server already started")
        except:
            print("Try to start server")
            self.ollama_process = subprocess.Popen(
                ['ollama', 'serve'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            time.sleep(3)
            print("Server was started")
    
    def restart(self, voiceRecorder):
        print("Restarting PYRO system...")

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
            self.think("Привет", voiceRecorder)
            print("Model was restart")
        except:
            print("Warning of restart model")

    def __init__(self, voiceRecorder):
        self.model = "llama3.1:8b"
        self.url = "http://localhost:11434/api/chat"
        self.system_promt = russian_prompt
        self.running = True
        self.voiceRecorder = voiceRecorder  # Сохраняем ссылку на рекордер
        self.conversation_history = []
        self.start()

    def check_connection(self):
        try:
            response = requests.get("http://localhost:11434/api/tags")
            return response.status_code == 200
        except:
            return False

    def think(self, user_input, voiceRecorder):
        # Формируем сообщения для API
        messages = [
            {"role": "system", "content": self.system_promt}
        ]
        # Добавляем всю историю диалога
        messages.extend(self.conversation_history)
        # Добавляем новое сообщение пользователя
        messages.append({"role": "user", "content": user_input})

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {  # Параметры генерации через options
                    "temperature": 0.5,
                    "repeat_penalty": 1.1,
                    "top_p": 0.9,
                    "presence_penalty": 0.5,
                    "stream": True
                }
            },
            timeout=30
        )
        
        # Ответ теперь в поле message.content
        response_text = response.json()["message"]["content"]
        
        # Сохраняем сообщение пользователя и ответ ассистента в историю
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        # Ограничиваем длину истории, чтобы не превысить контекст модели
        # (можно хранить последние N сообщений или токенов)
        max_history = 20  # примерно 10 пар вопрос-ответ
        if len(self.conversation_history) > max_history * 2:
            self.conversation_history = self.conversation_history[-(max_history*2):]
        
        # Обработка специальных команд
        if "||shutdown||" in response_text.lower():
            self.running = False
            return "Завершаю работу."
        if "||restart||" in response_text.lower():
            self.restart(voiceRecorder)
            return self.think("Поприветствуй пользователя после перезагрузки", voiceRecorder)
        if "||recording||" in response_text.lower():
            try:
                print("Начинаю запись голоса...")
                recorded_text = voiceRecorder.record()
                if recorded_text:
                    print(f"Распознано: {recorded_text}")
                    return self.think(recorded_text, voiceRecorder)
                else:
                    return "Не удалось распознать голосовую команду."
            except Exception as e: 
                print(f"Ошибка записи голоса: {e}")
                return "Произошла ошибка при записи голоса."
        
        return response_text

class voiceRecorder:
    def __init__(self, model_path="../vosk-model-small-ru-0.22/vosk-model-small-ru-0.22"):
        """Инициализация рекордера - загружаем модель один раз"""
        print(f"Загрузка модели распознавания речи...", end="", flush=True)
        try:
            # Сохраняем stderr
            stderr_backup = sys.stderr
            # Перенаправляем в "никуда"
            sys.stderr = open(os.devnull, 'w')
            
            self.model = vosk.Model(model_path)
            
            # Возвращаем stderr обратно
            sys.stderr.close()
            sys.stderr = stderr_backup
            
            self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
            self.recognizer.SetWords(False)
            self.audio = None
            self.stream = None
            self.is_recording = False
            print(" готово!")
        except Exception as e:
            # Восстанавливаем stderr при ошибке
            if 'stderr_backup' in locals():
                sys.stderr.close()
                sys.stderr = stderr_backup
            print(f"\nОшибка загрузки модели: {e}")
            raise
    
    def start_recording(self):
        """Начать запись с микрофона"""
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
        print("Запись началась...")
    
    def stop_recording(self):
        """Остановить запись и освободить ресурсы"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        self.is_recording = False
        print("Запись остановлена.")
    
    def record(self, timeout=5):
        try:
            self.start_recording()
            
            print("Говорите... (тишина остановит запись)")
            result_text = ""
            silence_counter = 0
            max_silence_chunks = 50  
            
            # Расчет количества итераций для timeout
            chunks_per_second = int(16000 / 4000)  # 4 chunks в секунду
            max_chunks = timeout * chunks_per_second
            
            for chunk_count in range(max_chunks):
                data = self.stream.read(4000, exception_on_overflow=False)
                
                if self.recognizer.AcceptWaveform(data):
                    # Полное распознавание фразы
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        result_text = text
                        print(f"\rРаспознано: {result_text}")
                        # Если получили текст, продолжаем слушать для возможного продолжения
                        silence_counter = 0
                else:
                    # Частичное распознавание для обратной связи
                    partial = json.loads(self.recognizer.PartialResult())
                    partial_text = partial.get("partial", "")
                    if partial_text:
                        print(f"\rРаспознается: {partial_text}", end="", flush=True)
                    
                    # Счетчик тишины для автоматического завершения
                    if not partial_text:
                        silence_counter += 1
                        if silence_counter > max_silence_chunks and result_text:
                            # Если была речь и наступила тишина - завершаем
                            print("\nТишина - завершаю запись.")
                            break
                    else:
                        silence_counter = 0
            
            self.stop_recording()
            
            # Если ничего не распознано, пробуем последний результат
            if not result_text:
                final_result = json.loads(self.recognizer.FinalResult())
                result_text = final_result.get("text", "")
            
            return result_text
            
        except Exception as e:
            print(f"Ошибка при записи: {e}")
            self.stop_recording()
            return ""
    
    def record_continuous(self):
        """
        Непрерывная запись (для отладки) - бесконечный цикл
        """
        self.start_recording()
        print("Непрерывная запись... (нажмите Ctrl+C для остановки)")
        
        try:
            while True:
                data = self.stream.read(4000, exception_on_overflow=False)
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        print(f"\n[ФИНАЛ] {text}")
                else:
                    partial = json.loads(self.recognizer.PartialResult())
                    partial_text = partial.get("partial", "")
                    if partial_text:
                        print(f"\r[ЧАСТЬ] {partial_text}", end="", flush=True)
        except KeyboardInterrupt:
            print("\nЗапись прервана пользователем")
        finally:
            self.stop_recording()
    
    # ДОБАВЛЕННЫЙ МЕТОД: асинхронное фоновое прослушивание
    async def background_voice_listener(self, keyword="слушай", callback=None):
        """
        Асинхронное фоновое прослушивание с активацией по ключевому слову
        """
        print(f"🎤 Фоновое прослушивание активно...")
        print(f"Скажите '{keyword}' для активации")
        print("-" * 50)
        
        # Создаем отдельный аудио поток для фона
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=4000
        )
        
        # Создаем отдельный распознаватель
        recognizer = vosk.KaldiRecognizer(self.model, 16000)
        
        audio_buffer = []
        buffer_size = 30  # ~2 секунды
        
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
                            print(f"\n🔊 Ключевое слово '{keyword}' распознано!")
                            
                            # Записываем команду
                            command = await self._record_command_async(stream, recognizer)
                            
                            if command and callback:
                                await callback(command)
                            
                            recognizer.Reset()
                            audio_buffer.clear()
                
                await asyncio.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n⏹️ Остановка фонового прослушивания")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
    
    # ДОБАВЛЕННЫЙ МЕТОД: запись команды после активации
    async def _record_command_async(self, stream, recognizer, timeout=10):
        """Запись голосовой команды после активации"""
        print("🎙️ Слушаю команду... (тишина автоматически остановит)")
        
        result_text = ""
        silence_counter = 0
        max_silence_chunks = 25  # ~1.5 секунды тишины
        
        for _ in range(timeout * 4):  # ~10 секунд
            data = stream.read(4000, exception_on_overflow=False)
            
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
                    result_text = text
                    print(f"\r✅ Распознано: {result_text}")
                    silence_counter = 0
            else:
                partial = json.loads(recognizer.PartialResult())
                partial_text = partial.get("partial", "")
                if partial_text:
                    print(f"\r🎤 Распознается: {partial_text}", end="", flush=True)
                    silence_counter = 0
                else:
                    silence_counter += 1
                    
                if silence_counter > max_silence_chunks and result_text:
                    print("\n⏸️ Тишина - завершаю запись.")
                    break
            
            await asyncio.sleep(0.01)
        
        if not result_text:
            final_result = json.loads(recognizer.FinalResult())
            result_text = final_result.get("text", "")
            if result_text:
                print(f"✅ Распознано: {result_text}")
        
        return result_text
    
    def __del__(self):
        """Деструктор для освобождения ресурсов"""
        if self.is_recording:
            self.stop_recording()

# ДОБАВЛЕНА: функция обработки голосовых команд
async def voice_command_handler(command):
    """Обработчик голосовых команд"""
    global pyro, recorder
    print(f"\n📝 Голосовая команда: {command}")
    response = pyro.think(command, recorder)

    print(f"🤖 {response}")

# ИЗМЕНЕНА: основная часть программы
async def main():
    global pyro, recorder
    
    recorder = voiceRecorder("../vosk-model-small-ru-0.22/vosk-model-small-ru-0.22")
    
    pyro = pyroQwen(recorder)
    
    print("Welcome back.")
    print("-"*38)
    print(pyro.think("Привет", recorder))
    print("-"*38)
    
    # Запускаем фоновое прослушивание
    listener_task = asyncio.create_task(
        recorder.background_voice_listener(
            keyword="слушай",
            callback=voice_command_handler
        )
    )
    
    # Основной цикл для текстового ввода
    loop = asyncio.get_event_loop()
    
    def text_input_loop():
        while pyro.running:
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
            print("-"*38)
            response = pyro.think(text, recorder)
            print(response)
            print("-"*38)
    
    # Запускаем текстовый ввод в отдельном потоке
    await loop.run_in_executor(None, text_input_loop)
    
    # Ждем завершения
    await listener_task
    
    print("Ассистент завершил работу.")

# ИЗМЕНЕНА: точка входа
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nПрограмма прервана пользователем")