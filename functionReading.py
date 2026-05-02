# function_gemma.py
import json
import requests

class CommandParser:
    def __init__(self, model="qwen2.5:1.5b"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"
        # Строгий промпт для классификации
        self.system_prompt = (
            "Ты — классификатор команд голосового ассистента. "
            "Определи, содержит ли сообщение команду из списка: "
            "shutdown, restart, record, listen, none. "
            "Верни ТОЛЬКО JSON без пояснений вида:\n"
            '{"command": "<одна из команд>", "args": ""}\n\n'
            "Если сообщение не является командой, верни "
            '{"command": "none", "args": ""}.\n'
            "Примеры:\n"
            "Q: выключись\nA: {\"command\": \"shutdown\", \"args\": \"\"}\n"
            "Q: перезагрузи систему\nA: {\"command\": \"restart\", \"args\": \"\"}\n"
            "Q: слушай\nA: {\"command\": \"record\", \"args\": \"\"}\n"
            "Q: привет\nA: {\"command\": \"none\", \"args\": \"\"}"
        )

    def parse(self, user_text: str) -> dict:
        if not user_text.strip():
            return {"command": "none", "args": ""}

        prompt = f"{self.system_prompt}\n\nQ: {user_text}\nA:"
        try:
            resp = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,   # максимальная детерминированность
                        "num_predict": 30
                    }
                },
                timeout=10
            )
            raw = resp.json()["response"].strip()
            # Очистка от возможных ```json ... ```
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else raw
            return json.loads(raw)
        except Exception as e:
            print(f"[CommandParser] Ошибка при разборе команды: {e}")
            return {"command": "none", "args": ""}