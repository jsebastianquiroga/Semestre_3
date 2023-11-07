from statsforecast.core import StatsForecast
from statsforecast.models import (
    AutoARIMA, AutoETS, AutoCES, AutoTheta,
    SeasonalNaive, ADIDA, CrostonClassic, IMAPA, TSB
)
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from joblib import dump
from datetime import datetime


class Autoregresive:
    def __init__(self, train, test, df, season_length=2, semestres_predecir=5):
        self.train = train
        self.test = test
        self.df = df
        self.season_length = season_length
        self.semestres_predecir = semestres_predecir
        self.Y_hat_df = None
        self.tiempo = None
        self.models = None
        self.merged_df = None
        self.validacion = None
        self.merged_df = pd.DataFrame()

    def format_data(self):
        self.train['ds'] = pd.to_datetime(self.train['ano'].astype(str) + '-' + (self.train['semestre'] * 6).astype(str) + '-01')
        self.test['ds'] = pd.to_datetime(self.test['ano'].astype(str) + '-' + (self.test['semestre'] * 6).astype(str) + '-01')
        self.df['ds'] = pd.to_datetime(self.df['ano'].astype(str) + '-' + (self.df['semestre'] * 6).astype(str) + '-01')

        train_SeriePronosticar = self.train[["ds", "unique_id", "y"]].copy().fillna(0)
        test_SeriePronosticar = self.test[["ds", "unique_id"]].copy().fillna(0)
        SeriePronosticar = self.df[["ds", "unique_id", "y"]].copy().fillna(0)

        return train_SeriePronosticar, test_SeriePronosticar, SeriePronosticar

    def setup_models_and_forecast(self, train_data):
        horizon = self.semestres_predecir
        self.models = [
            AutoARIMA(season_length=self.season_length),
            AutoETS(season_length=self.season_length),
            AutoCES(season_length=self.season_length),
            AutoTheta(season_length=self.season_length),
            SeasonalNaive(season_length=self.season_length),
            ADIDA(),
            CrostonClassic(),
            IMAPA(),
            TSB(alpha_d=0.2, alpha_p=0.2),
        ]

        model = StatsForecast(
            df=train_data,
            models=self.models,
            freq="6M",
            n_jobs=-1,
            fallback_model=SeasonalNaive(season_length=1),
        )

        self.Y_hat_df = model.forecast(horizon)

    
    def post_process(self):
        self.tiempo = self.df[["ds", "time_index"]].drop_duplicates()
        self.Y_hat_df = self.Y_hat_df.reset_index()
        self.Y_hat_df["ds"] = pd.to_datetime(self.Y_hat_df["ds"]) + pd.Timedelta(days=1)
        self.Y_hat_df = self.Y_hat_df.merge(
            self.tiempo[["ds", "time_index"]], left_on=["ds"], right_on=["ds"], how="left"
        )
    
        self.Y_hat_df["key"] = (
            self.Y_hat_df["time_index"].astype(str)
            + "_"
            + self.Y_hat_df["unique_id"].astype(str)
        )
        self.Y_hat_df = self.Y_hat_df.rename(
            columns={
                "AutoARIMA": "y_AutoARIMA",
                "AutoETS": "y_AutoETS",
                "CES": "y_AutoCES",
                "AutoTheta": "y_AutoTheta",
                "SeasonalNaive": "y_SeasonalNaive",
                "ADIDA": "y_ADIDA",
                "CrostonClassic": "y_CrostonClassic",
                "IMAPA": "y_IMAPA",
                "TSB": "y_TSB",
            }
        )
    
        prediction_columns = [
            "y_AutoARIMA",
            "y_AutoETS",
            "y_AutoCES",
            "y_AutoTheta",
            "y_SeasonalNaive",
            "y_ADIDA",
            "y_CrostonClassic",
            "y_IMAPA",
            "y_TSB",
        ]
    
        # Procesar cada columna de predicción
        for col in prediction_columns:
            # Convertir NaNs a 0 y luego redondear y convertir a enteros
            self.Y_hat_df[col] = np.nan_to_num(self.Y_hat_df[col], nan=0)
            self.Y_hat_df[col] = np.round(self.Y_hat_df[col]).astype(int)
            # Asegurarse de que no haya valores negativos
            self.Y_hat_df[col] = np.where(self.Y_hat_df[col] < 0, 0, self.Y_hat_df[col])


    def merge_with_validacion(self):
        self.validacion = self.test.copy()  #
        self.validacion["key"] = (
            self.validacion["time_index"].astype(str)
            + "_"
            + self.validacion["unique_id"].astype(str)
        )

        self.merged_df = self.validacion.merge(
            self.Y_hat_df[
                [
                    "y_AutoARIMA",
                    "y_AutoETS",
                    "y_AutoCES",
                    "y_AutoTheta",
                    "y_SeasonalNaive",
                    "y_ADIDA",
                    "y_CrostonClassic",
                    "y_IMAPA",
                    "y_TSB",
                    "key",
                ]
            ],
            left_on=["key"],
            right_on=["key"],
            how="left",
        )
        self.merged_df = self.merged_df.drop("key", axis=1)


    def run_workflow(self):
        train_data, test_data, SeriePronosticar = self.format_data()
        self.setup_models_and_forecast(train_data)
        self.post_process()
        self.merge_with_validacion()

        prediction_columns = [
            "y_AutoARIMA",
            "y_AutoETS",
            "y_AutoCES",
            "y_AutoTheta",
            "y_SeasonalNaive",
            "y_ADIDA",
            "y_CrostonClassic",
            "y_IMAPA",
            "y_TSB",
        ]

        # Procesar cada columna de predicción para convertir valores no numéricos a 0
        for col in prediction_columns:
            self.merged_df[col] = pd.to_numeric(self.merged_df[col], errors='coerce').fillna(0)

        return self.merged_df 

    def save_model(self):
        today = datetime.now().strftime('%Y-%m-%d')
        filename = f"Modelos/Autoregresive_{today}.joblib"
        dump(self.models, filename)
        print(f"Model saved as {filename}")


import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import column_or_1d
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
import os
import joblib
from datetime import datetime
from catboost import Pool
import shap
import matplotlib.pyplot as plt

class MyLabelEncoder(LabelEncoder):
    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = np.append(pd.Series(y).unique(), ["Unseen"])
        return self

    def transform(self, y):
        classes = pd.Series(y).unique()
        new_classes = set(classes) - set(self.classes_)
        for c in new_classes:
            y[y == c] = "Unseen"
        return super().transform(y)

    def fit_transform(self, y):
        return self.fit(y).transform(y)


class GradientBoostingModels:
    def __init__(self, train, validation, test, validacion, teste):
        # Solo adaptar las fechas si 'ds' no está en los DataFrame
        if 'ds' not in train.columns:
            self.train = self.adapt_dates(train)
        else:
            self.train = train

        if 'ds' not in validation.columns:
            self.validation = self.adapt_dates(validation)
        else:
            self.validation = validation
        
        if 'ds' not in test.columns:
            self.test = self.adapt_dates(test)
        else:
            self.test = test

        if 'ds' not in validacion.columns:
            self.validacion = self.adapt_dates(validacion)
        else:
            self.validacion = validacion
            
        if 'ds' not in teste.columns:
            self.teste = self.adapt_dates(teste)
        else:
            self.teste = teste

        self.models = {
            "lightgbm": lgb.LGBMRegressor(),
            "xgboost": xgb.XGBRegressor(),
            "catboost": CatBoostRegressor(),
        }
        self.encoders = {}

    
    def preprocess_data(self):
        # Comprobar y eliminar la columna 'ds' solo si existe en los DataFrame
        if 'ds' in self.train.columns:
            self.train.drop("ds", axis=1, inplace=True)
        if 'ds' in self.validation.columns:
            self.validation.drop("ds", axis=1, inplace=True)
        if 'ds' in self.test.columns:
            self.test.drop("ds", axis=1, inplace=True)
        
        for col in self.train.columns:
            if self.train[col].dtype == "object":
               encoder = MyLabelEncoder()
               self.train[col] = encoder.fit_transform(self.train[col])
               self.encoders[col] = encoder
               if col in self.validation.columns:
                   self.validation[col] = encoder.transform(self.validation[col])
               if col in self.test.columns:
                   self.test[col] = encoder.transform(self.test[col])


    @staticmethod
    def adapt_dates(df):
        df['ds'] = pd.to_datetime(df['ano'].astype(str) + '-' + (df['semestre'] * 6).astype(str) + '-01')
        return df



    def feature_selection(self):
        X_train, y_train = self.train.drop("y", axis=1), self.train["y"]

        # Using RandomForestRegressor for feature selection
        model = RandomForestRegressor()
        rfecv = RFECV(
            estimator=model, step=1, cv=KFold(10), scoring="neg_mean_squared_error"
        )
        rfecv.fit(X_train, y_train)

        selected_features = X_train.columns[rfecv.support_]

        # Keep only selected features
        self.train = self.train[["y"] + list(selected_features)]
        self.validation = (
            self.validation[["y"] + list(selected_features)]
            if "y" in self.validation.columns
            else self.validation[list(selected_features)]
        )
        self.test = self.test[list(selected_features)]

    def train_models(self):
        self.preprocess_data()
        self.feature_selection()
        X_train, y_train = self.train.drop("y", axis=1), self.train["y"]
        X_val, y_val = self.validation.drop("y", axis=1), self.validation["y"]

        for model in self.models.values():
            model.fit(
                X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10
            )

    def predict(self):
        self.train_models()
        predictions_validation = {}
        predictions_test = {}

        for name, model in self.models.items():
            X_validation = (
                self.validation.drop("y", axis=1)
                if "y" in self.validation.columns
                else self.validation
            )
            predictions_validation[name] = model.predict(X_validation)
            predictions_test[name] = model.predict(self.test)

        return predictions_validation, predictions_test


    def merge_predictions(self, predictions_validation, predictions_test):
        self.validacion = self.validacion.copy()
        self.validacion["key"] = (
            self.validacion["time_index"].astype(str)
            + "_"
            + self.validacion["unique_id"].astype(str)
        )

        for name, preds in predictions_validation.items():
            temp_df_validacion = pd.DataFrame(
                {"key": self.validacion["key"], f"y_{name}": preds}
            )
            self.validacion = self.validacion.merge(
                temp_df_validacion, on="key", how="left"
            )

        self.teste = self.teste.copy()
        self.teste["key"] = (
            self.teste["time_index"].astype(str)
            + "_"
            + self.teste["unique_id"].astype(str)
        )

        for name, preds in predictions_test.items():
            temp_df_teste = pd.DataFrame({"key": self.teste["key"], f"y_{name}": preds})
            self.teste = self.teste.merge(temp_df_teste, on="key", how="left")

        # list of columns to check for negative and NaN values
        columns = [
            "y_lightgbm",
            "y_xgboost",
            "y_catboost",
            # Agrega aquí más nombres de columnas de predicción según sea necesario
        ]

        # Redondear a enteros, luego reemplazar valores negativos y NaNs con 0
        for col in columns:
            if col in self.validacion.columns:
                self.validacion[col] = self.validacion[col].round().astype(int)
                self.validacion[col] = self.validacion[col].apply(lambda x: max(x, 0))
            if col in self.teste.columns:
                self.teste[col] = self.teste[col].round().astype(int)
                self.teste[col] = self.teste[col].apply(lambda x: max(x, 0))

        self.validacion = self.validacion.drop("key", axis=1)
        self.teste = self.teste.drop("key", axis=1)

        return self.validacion, self.teste
    def pipeline(self):
        predictions_validation, predictions_test = self.predict()
        validation_data_final, test_data_final = self.merge_predictions(
            predictions_validation, predictions_test
        )
        
        return validation_data_final, test_data_final

    def save_models(self):
        # Obtener la fecha actual
        current_date = datetime.now().strftime("%Y%m%d")
        # Guardar cada modelo con su nombre y la fecha actual
        for name, model in self.models.items():
            filename = f"Modelos/{name}_{current_date}.pkl"
            data_to_save = {
                'model': model,
                'encoders': self.encoders
            }
            joblib.dump(data_to_save, filename)
    def calculate_shap_values(self, X_train):
        shap_values = {}
        for name, model in self.models.items():
            if name in ['lightgbm', 'xgboost']:
                explainer = shap.TreeExplainer(model)
                shap_values[name] = explainer.shap_values(X_train)
            elif name == 'catboost':
                explainer = shap.TreeExplainer(model)
                shap_values[name] = explainer.shap_values(Pool(X_train))
        return shap_values

    def calculate_and_plot_shap_values(self, X_train):
        shap_values = self.calculate_shap_values(X_train)
        for name, values in shap_values.items():
            plt.figure()
            shap.summary_plot(values, X_train)
            plt.title(f"SHAP Summary Plot for {name.capitalize()} Model")  # Agregar título
            plt.savefig(f"shap_summary_{name}.png")  # Guardar la figura



import pandas as pd
import numpy as np

class TimeSeriesToSupervised:
    def __init__(self, df, datetime_col, entity_col, value_col, n_features, n_targets=0, column_name='Semestre'):
        self.df = df
        self.datetime_col = datetime_col
        self.entity_col = entity_col
        self.value_col = value_col
        self.n_features = n_features
        self.n_targets = n_targets
        self.column_name = column_name
    
    def transform(self):
        # Ordenar el DataFrame según las columnas especificadas
        self.df.sort_values(by=[self.entity_col, self.datetime_col], inplace=True)
        
        # Verificar si el DataFrame tiene suficientes filas para crear los lags
        if len(self.df) <= self.n_features:
            raise ValueError("El DataFrame no tiene suficientes filas para crear los lags requeridos.")

        # Crear columnas de lag/semestre
        for i in range(1, self.n_features + 1):
            self.df[f'{self.column_name}_anterior_{i}'] = self.df.groupby(self.entity_col)[self.value_col].shift(i)

        # Crear columnas para los targets si se especificaron
        if self.n_targets > 0:
            for i in range(1, self.n_targets + 1):
                self.df[f'{self.value_col}_ahead{i}'] = self.df.groupby(self.entity_col)[self.value_col].shift(-i)

        # Eliminar las filas donde alguna de las columnas de lag tenga NaN
        self.df.dropna(subset=[f'{self.column_name}_anterior_{i}' for i in range(1, self.n_features + 1)], inplace=True)

        return self.df
