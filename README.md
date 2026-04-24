# Sales Prediction API

API для предсказания продаж на основе характеристик магазина. Модель машинного обучения предсказывает значение `avg` (средние продажи) на основе 8 признаков.

## 🚀 Демо

- **API URL**: `'https://sales-prediction-api.up.railway.app'`
- **Документация**: `https://sales-prediction-api.up.railway.app/docs`
- **Health check**: `https://sales-prediction-api.up.railway.app/health`
- **Predict**: `https://sales-prediction-api.up.railway.app/predict`

## 📊 Метрики модели

| Метрика | Значение |
|---------|----------|
| R² Score | 0.718 |
| MAE | 100541.82 |
| RMSE | 151530.17 |

## 🔧 Установка и запуск локально

### Требования
- Python 3.10+
- pip

### Установка

```bash
# Клонирование репозитория
git clone https://github.com/Sergey-Karpov/so_api.git
cd so_api

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # для Linux/Mac
# venv\Scripts\activate   # для Windows

# Установка зависимостей
pip install -r requirements.txt