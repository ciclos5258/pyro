import requests
import json
import time
import subprocess

import ollama

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
    
    def restart(self):
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
            self.think(warmup_prompt)
            print("Model was restart")
        except:
            print("Warning of restart model")


    def __init__(self):
        self.model = "llama3.1:8b"
        self.url = "http://localhost:11434/api/generate"
        self.system_promt = russian_prompt
        self.running = True
        self.start()

    def check_connection(self):
        try:
            response = requests.get("http://localhost:11434/api/tags")
            return response.status_code == 200
        except:
            return False

    def think(self,user_input):
        prompt = f"{self.system_promt}\n\nПользователь: {user_input}\nPyro:"
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": 1,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "presence_penalty": 0.5,
                "presence_penalty": 0.5,
            },
            timeout=30
        )
        response_text = response.json()["response"]
        if "||shutdown||" in response_text.lower():
            self.running = False
            return "Завершаю работу."
        if "||restart||" in response_text.lower():
            self.restart()
            return "Перезагрузка..."
        return response_text
        

pyro = pyroQwen()
print("Welcome back.")
print("-"*38)
#Main loop
while pyro.running:
    data = input()
    print("-"*38)
    print(pyro.think(data))
    print("-"*38)