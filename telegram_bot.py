import numpy as np
print("NumPy version:", np.__version__)
import os
import logging
import torch
import threading
import nest_asyncio

# Применяем nest_asyncio для предотвращения ошибок с уже запущенным циклом событий
nest_asyncio.apply()

# Запускаем dummy HTTP-сервер (Flask), чтобы Render видел открытый порт
from flask import Flask
dummy_app = Flask(__name__)
@dummy_app.route("/")
def index():
    return "OK"
def run_dummy_server():
    port = int(os.environ.get("PORT", 5000))
    dummy_app.run(host="0.0.0.0", port=port)
threading.Thread(target=run_dummy_server, daemon=True).start()

# Отключаем интеграцию FlexAttention
os.environ["TRANSFORMERS_NO_FLEX_ATTENTION"] = "1"

# Если у torch отсутствует атрибут compiler, задаём dummy-заглушку
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

# Монки-патчим метод load_state_dict, чтобы игнорировать аргумент "assign"
old_load_state_dict = torch.nn.Module.load_state_dict
def patched_load_state_dict(self, state_dict, strict=True, *args, **kwargs):
    if "assign" in kwargs:
        kwargs.pop("assign")
    return old_load_state_dict(self, state_dict, strict, *args, **kwargs)
torch.nn.Module.load_state_dict = patched_load_state_dict

# Выводим версии для отладки
print("Torch version:", torch.__version__)
import transformers
print("Transformers version:", transformers.__version__)

# Импорт необходимых модулей из transformers и telegram
from transformers import AutoModelForCausalLM, AutoTokenizer
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Используем DialoGPT-small для экономии памяти.
# Загрузка модели с явным перемещением на CPU и параметром low_cpu_mem_usage=True,
# чтобы избежать состояния meta.
MODEL_NAME = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="cpu", low_cpu_mem_usage=True)

# Словарь для хранения истории диалога для каждого чата.
chat_histories = {}

# Настройка логгирования.
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Команда /start.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Привет! Я Telegram-бот на основе DialoGPT. Напиши сообщение для начала диалога.")

# Обработка входящих сообщений.
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_message = update.message.text

    # Кодируем новое сообщение (добавляем токен конца последовательности).
    new_input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors="pt")

    # Если для данного чата истории ещё нет или она слишком длинная, начинаем новую.
    if chat_id not in chat_histories or chat_histories[chat_id] is None or chat_histories[chat_id].shape[-1] > 256:
        bot_input_ids = new_input_ids
    else:
        bot_input_ids = torch.cat([chat_histories[chat_id], new_input_ids], dim=-1)

    # Генерируем ответ, ограничивая max_length для экономии памяти.
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=200,
        pad_token_id=tokenizer.eos_token_id
    )

    # Сохраняем обновлённую историю для данного чата.
    chat_histories[chat_id] = chat_history_ids

    # Извлекаем сгенерированный ответ (токены после пользовательского ввода).
    response_ids = chat_history_ids[:, bot_input_ids.shape[-1]:]
    bot_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    await update.message.reply_text(bot_response)

# Главная функция для запуска Telegram-бота.
async def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Не задан TELEGRAM_BOT_TOKEN. Задай его в переменной окружения.")
        return

    application = ApplicationBuilder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запускаем бота (polling) без закрытия основного цикла.
    await application.run_polling(close_loop=False)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
