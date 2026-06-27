import json
import requests
from typing import List, Dict, Optional
import os

class CommandParser:
    def __init__(
        self,
        model: str = "qwen2.5:1.5b",
        examples: Optional[List[Dict[str, str]]] = None,
        examples_file: Optional[str] = None
    ):
        """
        :param model: имя модели в Ollama
        :param examples: список словарей с ключами "user" и "assistant"
        :param examples_file: путь к JSON-файлу с массивом таких словарей
        """
        self.model = model
        self.url = "http://localhost:11434/api/chat"

        self.system_prompt = (
            "Ты — классификатор команд голосового ассистента. "
            "Определи, содержит ли сообщение команду из списка: "
            "shutdown, restart, record, none, web. "
            "Верни ТОЛЬКО JSON без пояснений вида:\n"
            '{"command": "<одна из команд>", "args": ""}\n\n'
            "Если сообщение не является командой, верни "
            '{"command": "none", "args": ""}.\n'
            "Никакого дополнительного текста, только JSON."
        )

        # Загружаем примеры
        self.examples = []
        if examples_file:
            self.examples = self._load_examples_from_file(examples_file)
        elif examples:
            self.examples = examples
        else:
            # По умолчанию загружаем из ../comands_datatset.json
            default_path = os.path.join("..", "comands_datatset.json")
            if os.path.exists(default_path):
                self.examples = self._load_examples_from_file(default_path)
            else:
                print(f"[CommandParser] Файл по умолчанию {default_path} не найден, примеры не загружены.")

    def _load_examples_from_file(self, file_path: str) -> List[Dict[str, str]]:
        """Загружает примеры из JSON-файла."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list) and all("user" in ex and "assistant" in ex for ex in data):
                return data
            else:
                raise ValueError("Файл должен содержать список объектов с полями 'user' и 'assistant'")
        except Exception as e:
            print(f"[CommandParser] Ошибка загрузки примеров: {e}")
            return []

    def parse(self, user_text: str) -> dict:
        if not user_text.strip():
            return {"command": "none", "args": ""}

        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        for ex in self.examples:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})

        messages.append({"role": "user", "content": user_text})

        try:
            resp = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 30
                    }
                },
                timeout=10
            )
            raw = resp.json()["message"]["content"].strip()

            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else raw

            return json.loads(raw)

        except Exception as e:
            print(f"[CommandParser] Ошибка: {e}")
            return {"command": "none", "args": ""}