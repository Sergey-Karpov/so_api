import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from transformers import (
    OutlierHandler,
    FeatureCreator,
    ShareCalculator,
    PopulationTransformer,
    MarketShareTransformer,
    InputFeatureValidator,
    ColumnDropper
)


def predict_from_input(input_data):
    """
    Функция для предсказания на основе входных данных

    Параметры:
    input_data: pandas DataFrame с колонками (8 признаков, БЕЗ city!):
        - chain: str - сеть магазина
        - cereals: float - количество SKU каш
        - milk: float - количество SKU молока
        - population: int - население города
        - market_share: float - доля рынка
        - aushan_count_in_city: int - кол-во магазинов Ашан в городе
        - detmir_count_in_city: int - кол-во магазинов Детский мир в городе
        - lenta_count_in_city: int - кол-во магазинов Лента в городе

    Возвращает:
    predictions: numpy array с предсказанными значениями avg
    """
    # Применяем pipeline предобработки
    input_prepared = loaded_pipeline.transform(input_data)

    # One-hot encoding для chain
    input_encoded = pd.get_dummies(input_prepared, columns=['chain'], drop_first=True)

    # Выравниваем колонки
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Удаляем лишние колонки
    input_encoded = input_encoded[feature_columns]

    # Масштабирование
    input_scaled = loaded_scaler.transform(input_encoded)

    # Предсказание
    predictions = loaded_model.predict(input_scaled)

    return predictions


class SalesPredictionInput(BaseModel):
    chain: str
    cereals: int
    milk: int
    population: int
    market_share: float
    aushan_count_in_city: int
    detmir_count_in_city: int
    lenta_count_in_city: int


class SalesPredictionResult(BaseModel):
    prediction: int
    success: bool = True

class SalesPredictionError(BaseModel):
    error: str
    success: bool = False

try:
    artifacts = joblib.load("model_artifacts.joblib")
    loaded_pipeline = artifacts['prediction_pipeline']
    loaded_scaler = artifacts['scaler']
    loaded_model = artifacts['model']
    feature_columns = artifacts['feature_columns']
    metadata = artifacts['metadata']
    print("✅ Модель успешно загружена")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    raise

app = FastAPI(
    title=metadata.get("name", 'Sales prediction API'),
    version=metadata.get("version", '0.0.1'),
    description=metadata.get("description", ''),
)

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {metadata.get('name', 'Sales prediction API')}!",
        "version": metadata.get("version", '0.0.1'),
        "type": metadata.get("type", 'random_forest_regressor'),
        "status": "active"
    }

@app.get('/health')
async def health_check():
    return {
        "status": "active",
        "model_loaded": loaded_model is not None,
    }

@app.post('/predict',
          response_model=SalesPredictionResult,)
async def predict(input_data: SalesPredictionInput):
    df = pd.DataFrame([input_data.model_dump()])
    predictions = predict_from_input(df)
    return SalesPredictionResult(prediction=int(predictions[0]))

# if __name__ == "__main__":
#     import uvicorn
#
#     print(f"Запуск {metadata.get('name', 'Sales Prediction API')}")
#     print(f"Версия: {metadata.get('version', '1.0.0')}")
#     print(f"Автор: {metadata.get('author', 'Unknown')}")
#     print(f"R2 на тесте: {test_metrics.get('R2', 'N/A')}")
#     print("=" * 60)
#     print("Health check: http://localhost:8000/health")
#     print("Предсказание: POST http://localhost:8000/predict")
#
#     uvicorn.run(app, host="0.0.0.0", port=8000)