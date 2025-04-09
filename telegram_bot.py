import os
import logging
import torch
import threading
import nest_asyncio

# ─── Flask dummy-сервер для Render ──────────────────────────────────────────────
from flask import Flask
dummy_app = Flask(__name__)

@dummy_app.route("/")
def index():
    return "OK"

def run_dummy_server():
    port = int(os.environ.get("PORT", 5000))
    dummy_app.run(host="0.0.0.0", port=port)

threading.Thread(target=run_dummy_server, daemon=True).start()

# ─── Fixes для Torch & Transformers ─────────────────────────────────────────────
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

# patch для assign
old_load_state_dict = torch.nn.Module.load_state_dict
def patched_load_state_dict(self, state_dict, strict=True, *args, **kwargs):
    kwargs.pop("assign", None)
    return old_load_state_dict(self, state_dict, strict, *args, **kwargs)
torch.nn.Module.load_state_dict = patched_load_state_dict

# ─── Версии ─────────────────────────────────────────────────────────────────────
print("Torch version:", torch.__version__)
import transformers
print("Transformers version:", transformers.__version__)

# ─── Импорты Telegram и Transformers ────────────────────────────────────────────
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters
)
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)  # не трогай .to()

chat_histories = {}

# ─── Логгирование ───────────────────────────────────────────────────────────────
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

# ─── Хендлеры ───────────────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я DialoGPT-бот. Напиши мне что-нибудь.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_message = update.message.text
    logging.info(f"[DEBUG] Сообщение от {chat_id}: {user_message}")

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

# ─── Главная функция ────────────────────────────────────────────────────────────
async def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("⛔ Не задан TELEGRAM_BOT_TOKEN. Добавь переменную окружения.")
        return

    application = ApplicationBuilder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logging.info("✅ Бот запущен. Ожидаю сообщения...")
    await application.run_polling(close_loop=False)

# ─── Запуск ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
