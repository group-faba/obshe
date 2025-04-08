import os
# Отключаем FlexAttention
os.environ["TRANSFORMERS_NO_FLEX_ATTENTION"] = "1"

import torch
# Если у torch отсутствует атрибут compiler, задаём заглушку
if not hasattr(torch, "compiler"):
    class DummyCompiler:
        @staticmethod
        def disable(recursive=False):
            def decorator(func):
                return func
            return decorator
    torch.compiler = DummyCompiler()

# Если нет атрибута float8_e4m3fn – задаём dummy, заменяя на torch.float32
if not hasattr(torch, "float8_e4m3fn"):
    torch.float8_e4m3fn = torch.float32

print("Torch version:", torch.__version__)
import transformers
print("Transformers version:", transformers.__version__)

from transformers import AutoModelForCausalLM, AutoTokenizer
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Загружаем модель и токенизатор DialoGPT-small (меньшая модель — экономия памяти)
MODEL_NAME = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Словарь для хранения истории переписки для каждого пользователя (chat_id)
chat_histories = {}

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("Привет! Я Telegram бот на основе DialoGPT. Напиши сообщение, чтобы начать диалог.")

def handle_message(update: Update, context: CallbackContext) -> None:
    chat_id = update.effective_chat.id
    user_message = update.message.text

    # Кодируем новое сообщение с токеном конца последовательности
    new_input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors="pt")

    # Если истории для этого чата ещё нет или она слишком длинная — начинаем новую историю
    if chat_id not in chat_histories or chat_histories[chat_id] is None or chat_histories[chat_id].shape[-1] > 256:
        bot_input_ids = new_input_ids
    else:
        bot_input_ids = torch.cat([chat_histories[chat_id], new_input_ids], dim=-1)

    # Генерируем ответ, ограничивая максимальную длину (max_length=200) для экономии памяти
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=200,
        pad_token_id=tokenizer.eos_token_id
    )

    # Сохраняем историю для этого чата
    chat_histories[chat_id] = chat_history_ids

    # Извлекаем ответ — токены, сгенерированные после пользовательского ввода
    response_ids = chat_history_ids[:, bot_input_ids.shape[-1]:]
    bot_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    update.message.reply_text(bot_response)

def main():
    # Чтение токена бота из переменной окружения TELEGRAM_BOT_TOKEN
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Не задан TELEGRAM_BOT_TOKEN. Укажи его в переменной окружения или в коде.")
        return

    updater = Updater(token, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    # Запуск бота в режиме polling
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
