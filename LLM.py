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


    def __init__(self):
        self.model = "llama3.1:8b"
        self.url = "http://localhost:11434/api/generate"
        self.system_promt = russian_prompt
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
                "temperature": 0.1,
            },
            timeout=30
        )
        response_text = response.json()["response"]
        if "shutdown" in response_text.lower():
            self.running = False
            return "Завершаю работу."
        return response_text
    

pyro = pyroQwen()
print("Welcome back.")
print("-"*38)
#Main loop
while True:
    data = input()
    print("-"*38)
    print(pyro.think(data))
    print("-"*38)