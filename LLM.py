import requests
import json
import time
import subprocess
import queue
import sys 
import os

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

        if hasattr(self, 'ollama_process'):
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
        try:
            warmup_prompt = "Привет"
            self.think(warmup_prompt, voiceRecorder)
            print("Model was restart")
        except:
            print("Warning of restart model")

    def __init__(self, voiceRecorder):
        self.model = "llama3.1:8b"
        self.url = "http://localhost:11434/api/generate"
        self.system_promt = russian_prompt
        self.running = True
        self.voiceRecorder = voiceRecorder  # Сохраняем ссылку на рекордер
        self.start()

    def check_connection(self):
        try:
            response = requests.get("http://localhost:11434/api/tags")
            return response.status_code == 200
        except:
            return False

    def think(self, user_input, voiceRecorder):
        prompt = f"{self.system_promt}\n\nПользователь: {user_input}\nPyro:"
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.5,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "presence_penalty": 0.5,
            },
            timeout=30
        )
        response_text = response.json()["response"]
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
                    # Обрабатываем распознанный текст
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
        """
        Записать голосовую команду и вернуть распознанный текст
        
        Args:
            timeout: максимальная длительность записи в секундах
        
        Returns:
            str: распознанный текст или пустая строка
        """
        try:
            self.start_recording()
            
            print("Говорите... (тишина остановит запись)")
            result_text = ""
            silence_counter = 0
            max_silence_chunks = 20  # ~2 секунды тишины (4000/16000 * 20 = 5 сек)
            
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
    
    def __del__(self):
        """Деструктор для освобождения ресурсов"""
        if self.is_recording:
            self.stop_recording()

# Создаем экземпляр рекордера (модель загружается один раз)
recorder = voiceRecorder("../vosk-model-small-ru-0.22/vosk-model-small-ru-0.22")

# Создаем экземпляр pyro с передачей рекордера
pyro = pyroQwen(recorder)

print("Welcome back.")
print("-"*38)
print(pyro.think("Привет", recorder))
print("-"*38)

# Main loop
while pyro.running:
    data = input()
    print("-"*38)
    response = pyro.think(data, recorder)
    print(f"Pyro: {response}")
    print("-"*38)

print("Ассистент завершил работу.")