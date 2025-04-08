import os
# Отключаем интеграцию FlexAttention
os.environ["TRANSFORMERS_NO_FLEX_ATTENTION"] = "1"

import torch

# Если у torch отсутствует атрибут compiler, создаём dummy-заглушку
if not hasattr(torch, "compiler"):
    class DummyCompiler:
        @staticmethod
        def disable(recursive=False):
            def decorator(func):
                return func
            return decorator
    torch.compiler = DummyCompiler()

# Если отсутствует float8_e4m3fn – задаём его как dummy (заменяем на torch.float32)
if not hasattr(torch, "float8_e4m3fn"):
    torch.float8_e4m3fn = torch.float32

print("Torch version:", torch.__version__)
import transformers
print("Transformers version:", transformers.__version__)

from flask import Flask, request, jsonify
from transformers import pipeline

# Минимальная реализация Conversation для версии transformers 4.51.1
class Conversation:
    def __init__(self, text, conversation_id=None):
        # Сохраняем историю пользовательских сообщений и ответов модели
        self.past_user_inputs = [text]
        self.generated_responses = []
        self.conversation_id = conversation_id

    def add_user_input(self, text):
        self.past_user_inputs.append(text)

app = Flask(__name__)

# Инициализируем pipeline для диалогов с DialoGPT-medium
chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")
conversation = None  # Объект для хранения истории диалога

@app.route("/")
def index():
    return "Chatbot API is running."

@app.route("/chat", methods=["POST"])
def chat():
    global conversation
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data["message"]

    # Если диалог еще не начат — создаём новый, иначе добавляем новый ввод
    if conversation is None:
        conversation = Conversation(user_message)
    else:
        conversation.add_user_input(user_message)
    
    result = chatbot(conversation)
    # Если сгенерированные ответы отсутствуют, возвращаем пустую строку
    answer = result.generated_responses[-1] if result.generated_responses else ""
    return jsonify({"response": answer})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
