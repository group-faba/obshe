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

# Если отсутствует float8_e4m3fn, задаём dummy (заменяем на torch.float32)
if not hasattr(torch, "float8_e4m3fn"):
    torch.float8_e4m3fn = torch.float32

# Монки-патчим метод load_state_dict, чтобы игнорировать аргумент assign
old_load_state_dict = torch.nn.Module.load_state_dict

def patched_load_state_dict(self, state_dict, strict=True, *args, **kwargs):
    if "assign" in kwargs:
        kwargs.pop("assign")
    return old_load_state_dict(self, state_dict, strict, *args, **kwargs)

torch.nn.Module.load_state_dict = patched_load_state_dict

print("Torch version:", torch.__version__)

import transformers
print("Transformers version:", transformers.__version__)

# Импортируем Flask и создаём экземпляр приложения
from flask import Flask
app = Flask(__name__)

# Дальше можно загружать модель, например:
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Запуск веб-сервиса
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
