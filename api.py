import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np

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
    artifacts = joblib.load("simple_model.joblib")
    model = artifacts['model']
    scaler = artifacts['scaler']
    feature_columns = artifacts['feature_columns']
    metadata = artifacts.get('metadata', {})
    print("✅ Модель успешно загружена")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    raise


def predict_from_input(df: pd.DataFrame) -> np.ndarray:
    """
    Ручная предобработка, копирующая логику pipeline БЕЗ использования кастомных классов
    """
    # Создаем копию, чтобы не изменять оригинал
    data = df.copy()

    # 1. Создаем новые признаки (как FeatureCreator)
    data["cereals_milk_ratio"] = data["cereals"] / (data["milk"] + 1)
    data["cereals_milk_multi"] = data["cereals"] * data["milk"]

    # 2. Рассчитываем доли сетей (как ShareCalculator)
    total_stores = data['aushan_count_in_city'] + data['detmir_count_in_city'] + data['lenta_count_in_city']
    denominator = total_stores.replace(0, 1)
    data['aushan_count_share_in_city'] = data['aushan_count_in_city'] / denominator
    data['detmir_count_share_in_city'] = data['detmir_count_in_city'] / denominator
    data['lenta_count_share_in_city'] = data['lenta_count_in_city'] / denominator

    # 3. Удаляем исходные колонки с количествами (как ColumnDropper)
    data = data.drop(columns=['aushan_count_in_city', 'detmir_count_in_city', 'lenta_count_in_city'])

    # 4. One-hot encoding для chain
    data = pd.get_dummies(data, columns=['chain'], drop_first=True)

    # 5. Выравниваем колонки с обучающими данными
    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0

    data = data[feature_columns]

    # 6. Масштабирование
    scaled = scaler.transform(data)

    # 7. Предсказание
    predictions = model.predict(scaled)

    return predictions


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
        "model_loaded": model is not None,
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