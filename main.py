from transformers import pipeline, Conversation

# Инициализируем пайплайн для диалогов с моделью DialoGPT-medium.
chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")

print("ИИ запущен. Введи текст для общения, 'new' для нового диалога или 'exit' для выхода.")

conversation = None  # Объект для хранения истории диалога

while True:
    user_input = input("Ты: ").strip()
    if user_input.lower() in ['exit', 'quit']:
        print("ИИ: До встречи!")
        break
    if user_input.lower() == 'new':
        conversation = None
        print("ИИ: Новый диалог начался.")
        continue

    # Если это первая реплика или после команды "new" – создаём новый объект Conversation,
    # иначе добавляем ввод пользователя в уже существующий диалог.
    if conversation is None:
        conversation = Conversation(user_input)
    else:
        conversation.add_user_input(user_input)

    # Получаем ответ модели с учётом контекста диалога.
    result = chatbot(conversation)
    print("ИИ:", result.generated_responses[-1])
