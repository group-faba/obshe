import os
import logging
import torch
import threading
import nest_asyncio
from flask import Flask

# ── Запуск dummy-сервера (для Render)
dummy_app = Flask(__name__)
@dummy_app.route("/")
def index():
    return "OK"

def run_dummy_server():
    port = int(os.environ.get("PORT", 5000))
    dummy_app.run(host="0.0.0.0", port=port)

threading.Thread(target=run_dummy_server, daemon=True).start()

# ── Исправления для PyTorch и Transformers
os.environ["TRANSFORMERS_NO_FLEX_ATTENTION"] = "1"
nest_asyncio.apply()

if not hasattr(torch, "compiler"):
    class DummyCompiler:
        @staticmethod
        def disable(recursive=False):
            def decorator(func): return func
            return decorator
    torch.compiler = DummyCompiler()

if not hasattr(torch, "float8_e4m3fn"):
    torch.float8_e4m3fn = torch.float32

old_load_state_dict = torch.nn.Module.load_state_dict
def patched_load_state_dict(self, state_dict, strict=True, *args, **kwargs):
    kwargs.pop("assign", None)
    return old_load_state_dict(self, state_dict, strict, *args, **kwargs)
torch.nn.Module.load_state_dict = patched_load_state_dict

# ── Логирование
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

# ── Импорты Telegram и Transformers
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Загрузка модели без meta
MODEL_NAME = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="./model", force_download=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir="./model", force_download=True)
# ⚠️ Без .to(), без device_map, без low_cpu_mem_usage

chat_histories = {}

# ── Хендлер команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я DialoGPT-бот. Напиши мне что-нибудь.")

# ── Хендлер сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_message = update.message.text
    logging.info(f"[DEBUG] Сообщение от {chat_id}: {user_message}")

    try:
        new_input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors="pt")

        if chat_id not in chat_histories or chat_histories[chat_id].shape[-1] > 256:
            bot_input_ids = new_input_ids
        else:
            bot_input_ids = torch.cat([chat_histories[chat_id], new_input_ids], dim=-1)

        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=200,
            pad_token_id=tokenizer.eos_token_id
        )

        chat_histories[chat_id] = chat_history_ids
        response_ids = chat_history_ids[:, bot_input_ids.shape[-1]:]
        bot_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

        await update.message.reply_text(bot_response)

    except Exception as e:
        logging.error(f"[ERROR] Ошибка в генерации: {e}")
        await update.message.reply_text("⚠️ Произошла ошибка генерации. Попробуй ещё раз.")

# ── Запуск бота
async def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logging.error("⛔ Не задан TELEGRAM_BOT_TOKEN.")
        return

    application = ApplicationBuilder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logging.info("✅ Бот запущен.")
    await application.run_polling(close_loop=False)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
