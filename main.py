import os
# Отключаем интеграцию FlexAttention
os.environ["TRANSFORMERS_NO_FLEX_ATTENTION"] = "1"

import torch
# Если у torch отсутствует атрибут compiler, создаём dummy‑заглушку
if not hasattr(torch, "compiler"):
    class DummyCompiler:
        @staticmethod
        def disable(recursive=False):
            def decorator(func):
                return func
            return decorator
    torch.compiler = DummyCompiler()

# Если отсутствует float8_e4m3fn – задаём dummy (используем torch.float32)
if not hasattr(torch, "float8_e4m3fn"):
    torch.float8_e4m3fn = torch.float32

print("Torch version:", torch.__version__)
import transformers
print("Transformers version:", transformers.__version__)

from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

# Загружаем модель и токенизатор для DialoGPT-medium
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Глобальное состояние для хранения истории диалога (chat_history_ids)
chat_history_ids = None

app = Flask(__name__)

@app.route("/")
def index():
    return "Chatbot API is running."

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history_ids
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data["message"]
    # Кодируем новое сообщение, добавляя токен конца последовательности
    new_input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors="pt")

    # Если это первое сообщение, используем его, иначе объединяем с историей диалога
    if chat_history_ids is None:
         bot_input_ids = new_input_ids
    else:
         bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)

    # Генерируем ответ модели с учётом всей истории
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    # Извлекаем сгенерированный ответ (новые токены после пользовательского ввода)
    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
