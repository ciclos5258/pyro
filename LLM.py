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

    def query(self, prompt):
        self.start()
        


    def __init__(self):
        self.model = "qwen2.5:7b"
        self.url = "http://localhost:11434/api/generate"
        self.system_promt = russian_prompt

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
                "temperature": 0.5,
            },
            timeout=30
        )
        response_text = response.json()["response"]
        if "shotdown" in response_text.lower():
            global main
            main = False
            return "Завершаю работу."
        return response_text
    

pyro = pyroQwen()
print("Welcome back.")
while main:
    data = input()
    if data == "bye":
        main = False
    else:
        print("--------------------------------------------------------------------")
        print(pyro.think(data))
        print("--------------------------------------------------------------------")