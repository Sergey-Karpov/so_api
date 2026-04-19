import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd


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
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    prediction_pipeline = artifacts["prediction_pipeline"]
    feature_columns = artifacts.get("feature_columns", [])
    input_features = artifacts.get("input_features", [])
    metadata = artifacts.get("metadata", {})
    train_metrics = artifacts.get("train_metrics", {})
    test_metrics = artifacts.get("test_metrics", {})


except Exception as e:
    print(f"Ошибка загрузки модели {e}")
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
        "model_loaded": model is not None,
    }

@app.post('/predict',
          response_model=SalesPredictionResult,)
async def predict(input_data: SalesPredictionInput):
    df = pd.DataFrame([input_data.model_dump()])

    input_prepared = prediction_pipeline.transform(df)

    input_encoded = pd.get_dummies(input_prepared, columns=['chain'], drop_first=True)

    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[feature_columns]
    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)[0]

    return SalesPredictionResult(prediction=int(prediction))


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