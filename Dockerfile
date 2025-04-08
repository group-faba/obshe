# Используем официальный образ Python (можно взять полноценный, а не slim, чтобы не было проблем с зависимостями)
FROM python:3.11

# Обновляем пакеты и устанавливаем необходимые системные библиотеки
RUN apt-get update && apt-get install -y build-essential

# Создаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код приложения
COPY . .

# Открываем порт (Render использует переменную окружения PORT, по умолчанию 5000)
EXPOSE 5000

# Команда для запуска приложения
CMD ["python", "main.py"]
