FROM python:3.11-slim

WORKDIR /app

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Фиксированный порт 8000
ENV PORT=8000
EXPOSE 8000

# Запуск (без использования $PORT как переменной)
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port 8000"]