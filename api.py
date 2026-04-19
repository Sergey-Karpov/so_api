import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

#
# from transformers import (
#     FeatureCreator,
#     ShareCalculator,
#     PopulationTransformer,
#     MarketShareTransformer,
#     InputFeatureValidator,
#     ColumnDropper
# )

class FeatureCreator(BaseEstimator, TransformerMixin):
    """Создание новых признаков из cereals и milk"""

    def __init__(self, use_predict=False):
        self.use_predict = use_predict
        self.store_counts = None

    def fit(self, X, y=None):
        if not self.use_predict:
            self.store_counts = pd.crosstab(X["city"], X["chain"])
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy["cereals_milk_ratio"] = X_copy["cereals"] / (X_copy["milk"] + 1)
        X_copy["cereals_milk_multi"] = X_copy["cereals"] * X_copy["milk"]

        if not self.use_predict:
            X_copy["aushan_count_in_city"] = X_copy["city"].map(self.store_counts["Ашан"]).fillna(0)
            X_copy["detmir_count_in_city"] = X_copy["city"].map(self.store_counts["Детский мир"]).fillna(0)
            X_copy["lenta_count_in_city"] = X_copy["city"].map(self.store_counts["Лента"]).fillna(0)
        else:
            required_cols = ["aushan_count_in_city", "detmir_count_in_city", "lenta_count_in_city"]
            missing_cols = [col for col in required_cols if col not in X_copy.columns]
            if missing_cols:
                raise ValueError(f"Для предсказания необходимы колонки {missing_cols}")

        return X_copy


class ShareCalculator(BaseEstimator, TransformerMixin):
    """Расчет долей топовых сетей на основе входных данных"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy['top_chains_stores_count'] = (
                X_copy['aushan_count_in_city'] +
                X_copy['detmir_count_in_city'] +
                X_copy['lenta_count_in_city']
        )
        denominator = X_copy['top_chains_stores_count'].replace(0, 1)
        X_copy['aushan_count_share_in_city'] = X_copy['aushan_count_in_city'] / denominator
        X_copy['detmir_count_share_in_city'] = X_copy['detmir_count_in_city'] / denominator
        X_copy['lenta_count_share_in_city'] = X_copy['lenta_count_in_city'] / denominator
        return X_copy


class PopulationTransformer(BaseEstimator, TransformerMixin):
    """Добавление данных о населении"""

    def __init__(self, population_file='data/population.xlsx', use_predict=False):
        self.population_file = population_file
        self.use_predict = use_predict
        self.population_data = None
        self.manual_population = {
            'Москва обл': 8775735,
            'Санкт-Петербург обл': 2059479,
            'Симферополь': 335009,
            'Славгород': 27040,
            'Орел': 289503,
            'Артем': 108274
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        if self.use_predict:
            if 'population' not in X_copy.columns:
                raise ValueError("Для предсказания необходима колонка 'population'")
        else:
            if self.population_data is not None:
                X_copy = X_copy.merge(self.population_data, on='city', how='left')
                X_copy['population'] = X_copy.apply(
                    lambda row: self.manual_population.get(row['city'], row.get('population', 0)),
                    axis=1
                )
            else:
                X_copy['population'] = X_copy['city'].map(self.manual_population).fillna(0)
        return X_copy


class MarketShareTransformer(BaseEstimator, TransformerMixin):
    """Добавление данных о доле рынка"""

    def __init__(self, market_share_file='data/market_share.xlsx', use_predict=False):
        self.market_share_file = market_share_file
        self.use_predict = use_predict
        self.market_share_data = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        if self.use_predict:
            if 'market_share' not in X_copy.columns:
                raise ValueError("Для предсказания необходима колонка 'market_share'")
        else:
            if self.market_share_data is not None:
                X_copy = X_copy.merge(self.market_share_data, how='left', on='city')
                X_copy['market_share'] = X_copy['market_share'].fillna(0)
            else:
                X_copy['market_share'] = 0
        return X_copy


class InputFeatureValidator(BaseEstimator, TransformerMixin):
    """Валидатор входных признаков для предсказания"""

    def __init__(self, required_features):
        self.required_features = required_features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        missing_features = set(self.required_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Отсутствуют обязательные признаки: {missing_features}")
        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Удаление ненужных колонок"""

    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        columns_to_drop = [col for col in self.columns_to_drop if col in X_copy.columns]
        if columns_to_drop:
            X_copy = X_copy.drop(columns=columns_to_drop)
        return X_copy

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


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ручное применение всех трансформеров в правильной последовательности
    """
    # 1. Валидация входных признаков
    validator = InputFeatureValidator(input_features)
    df = validator.transform(df)

    # 2. Создание новых признаков (ratio, multi)
    feature_creator = FeatureCreator(use_predict=True)
    df = feature_creator.transform(df)

    # 3. Расчет долей сетей
    share_calc = ShareCalculator()
    df = share_calc.transform(df)

    # 4. Проверка population (уже есть во входных данных)
    population = PopulationTransformer(use_predict=True)
    df = population.transform(df)

    # 5. Проверка market_share (уже есть во входных данных)
    market_share = MarketShareTransformer(use_predict=True)
    df = market_share.transform(df)

    # 6. Удаление ненужных колонок
    column_dropper = ColumnDropper([
        'aushan_count_in_city', 'detmir_count_in_city',
        'lenta_count_in_city', 'top_chains_stores_count'
    ])
    df = column_dropper.transform(df)

    return df

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
    df_prepared = preprocess_data(df)
    input_encoded = pd.get_dummies(df_prepared, columns=['chain'], drop_first=True)
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