import os

# Отключаем интеграцию FlexAttention
os.environ["TRANSFORMERS_NO_FLEX_ATTENTION"] = "1"

import torch

# Если у torch отсутствует атрибут compiler, добавляем заглушку
if not hasattr(torch, "compiler"):
    class DummyCompiler:
        @staticmethod
        def disable(recursive=False):
            def decorator(func):
                return func
            return decorator
    torch.compiler = DummyCompiler()

# Если отсутствует float8_e4m3fn – задаем dummy (используем torch.float32)
if not hasattr(torch, "float8_e4m3fn"):
    torch.float8_e4m3fn = torch.float32

print("Torch version:", torch.__version__)

import transformers
print("Transformers version:", transformers.__version__)

from flask import Flask, request, jsonify
from transformers import pipeline

# Пытаемся импортировать Conversation из transformers,
# если не удалось – определяем минимальную реализацию
try:
    from transformers import Conversation
except ImportError:
    class Conversation:
        def __init__(self, text, conversation_id=None):
            # Список предыдущих пользовательских сообщений
            self.past_user_inputs = [text]
            # Список сгенерированных ответов модели (будет обновляться пайплайном)
            self.generated_responses = []
            self.conversation_id = conversation_id

        def add_user_input(self, text):
            self.past_user_inputs.append(text)

app = Flask(__name__)

# Инициализируем чат-бота с моделью DialoGPT-medium
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

    # Создаем новый диалог или обновляем существующий
    if conversation is None:
        conversation = Conversation(user_message)
    else:
        conversation.add_user_input(user_message)
    
    result = chatbot(conversation)
    answer = result.generated_responses[-1]
    return jsonify({"response": answer})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
