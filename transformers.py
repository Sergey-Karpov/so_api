from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


__all__ = [
    'OutlierHandler',
    'FeatureCreator',
    'ShareCalculator',
    'PopulationTransformer',
    'MarketShareTransformer',
    'InputFeatureValidator',
    'ColumnDropper'
]

class OutlierHandler(BaseEstimator, TransformerMixin):
    """Обработка выбросов в колонке 'avg'"""

    def __init__(self, column='avg'):
        self.column = column
        self.lower_bound = None
        self.upper_bound = None

    def fit(self, X, y=None):
        q1 = X[self.column].quantile(0.25)
        q3 = X[self.column].quantile(0.75)
        iqr = q3 - q1
        self.lower_bound = q1 - 1.5 * iqr
        self.upper_bound = q3 + 1.5 * iqr
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy[self.column] = X_copy[self.column].clip(self.lower_bound, self.upper_bound)
        return X_copy


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

        # Суммируем количество магазинов топовых сетей
        X_copy['top_chains_stores_count'] = (
                X_copy['aushan_count_in_city'] +
                X_copy['detmir_count_in_city'] +
                X_copy['lenta_count_in_city']
        )

        # Избегаем деления на ноль
        denominator = X_copy['top_chains_stores_count'].replace(0, 1)

        # Создаем доли
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
        if not self.use_predict:
            try:
                df_pop = pd.read_excel(self.population_file)
                df_pop.rename(columns={
                    'Русское\nназвание': 'city',
                    'Перепись населения 2021 года[3]': 'population'
                }, inplace=True)

                df_pop['population'] = (df_pop['population']
                                        .astype(str)
                                        .str.replace(',', '', regex=False)
                                        .str.replace('.', '', regex=False)
                                        .str.replace(' ', '', regex=False)
                                        .astype(int))

                self.population_data = df_pop
            except:
                print("Не удалось загрузить population.xlsx, использую manual_population")
                self.population_data = None
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        if self.use_predict:
            # Для предсказаний - population уже есть во входных данных
            if 'population' not in X_copy.columns:
                raise ValueError("Для предсказания необходима колонка 'population'")
        else:
            # Для обучения - добавляем population из справочника
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
        if not self.use_predict:
            try:
                df_share = pd.read_excel(self.market_share_file)
                df_share['city'] = df_share['city'].replace({
                    'Московская обл': 'Москва обл',
                    'Ленинградская': 'Санкт-Петербург обл'
                })
                df_share = df_share.rename(columns={'share': 'market_share'})
                self.market_share_data = df_share
            except:
                print("Не удалось загрузить market_share.xlsx")
                self.market_share_data = None
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        if self.use_predict:
            # Для предсказаний - market_share уже есть во входных данных
            if 'market_share' not in X_copy.columns:
                raise ValueError("Для предсказания необходима колонка 'market_share'")
        else:
            # Для обучения - добавляем market_share из справочника
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
        # Проверяем наличие всех обязательных признаков
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
            X_copy = X_copy.drop(columns=columns_to_drop, axis=1)
        return X_copy