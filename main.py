import os
# Отключаем интеграцию FlexAttention
os.environ["TRANSFORMERS_NO_FLEX_ATTENTION"] = "1"

import torch

# Если отсутствует атрибут compiler – задаём dummy-заглушку
if not hasattr(torch, "compiler"):
    class DummyCompiler:
        @staticmethod
        def disable(recursive=False):
            def decorator(func):
                return func
            return decorator
    torch.compiler = DummyCompiler()

# Если отсутствует float8_e4m3fn – задаём dummy (заменяем на torch.float32)
if not hasattr(torch, "float8_e4m3fn"):
    torch.float8_e4m3fn = torch.float32

# Монки-патчим метод load_state_dict, чтобы игнорировать параметр assign
old_load_state_dict = torch.nn.Module.load_state_dict
def patched_load_state_dict(self, state_dict, strict=True, *args, **kwargs):
    if "assign" in kwargs:
        kwargs.pop("assign")
    return old_load_state_dict(self, state_dict, strict, *args, **kwargs)
torch.nn.Module.load_state_dict = patched_load_state_dict

print("Torch version:", torch.__version__)

import transformers
print("Transformers version:", transformers.__version__)

from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

# Используем DialoGPT-small для уменьшения расхода памяти
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Глобальное состояние для хранения истории диалога
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
    # Кодируем новое сообщение и добавляем токен конца последовательности
    new_input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors="pt")

    # Если диалог первый или история слишком длинная, сбрасываем историю
    if chat_history_ids is None or chat_history_ids.shape[-1] > 256:
        bot_input_ids = new_input_ids
    else:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)

    # Генерируем ответ модели с ограничением max_length для снижения расхода памяти
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=200,
        pad_token_id=tokenizer.eos_token_id
    )
    # Извлекаем сгенерированный ответ (новые токены после пользовательского ввода)
    response_ids = chat_history_ids[:, bot_input_ids.shape[-1]:]
    bot_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
