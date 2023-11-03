import numpy as np
import pandas as pd
from dtw import dtw


class ForecastEvaluator:
    def __init__(self, y_actual, y_forecast):
        self.y_actual = np.array(y_actual)
        self.y_forecast = np.array(y_forecast)

    def mean_absolute_error(self):
        return np.mean(np.abs(self.y_actual - self.y_forecast))

    def mean_absolute_percentage_error(self):
        self.y_actual = np.where(self.y_actual == 0, 1e-10, self.y_actual)
        return np.mean(np.abs((self.y_actual - self.y_forecast) / self.y_actual)) * 100

    def mean_squared_error(self):
        return np.mean(np.square(self.y_actual - self.y_forecast))

    def root_mean_squared_error(self):
        return np.sqrt(self.mean_squared_error())

    def mean_absolute_scaled_error(self):
        if np.isnan(self.y_actual).any():
            raise ValueError(
                "y_actual contains NaN values. Please remove or impute missing values before calculating MASE."
            )

        naive_forecast = self.y_actual[:-1]
        mae_naive = np.mean(np.abs(self.y_actual[1:] - naive_forecast))

        epsilon = 1e-10
        mae_model = self.mean_absolute_error()

        return mae_model / (mae_naive + epsilon)

    def symmetric_mape(self):
        self.y_actual = np.where(self.y_actual == 0, 1e-10, self.y_actual)
        return (
            np.mean(
                2
                * np.abs(self.y_actual - self.y_forecast)
                / (np.abs(self.y_actual) + np.abs(self.y_forecast))
            )
            * 100
        )

    def mean_directional_accuracy(self):
        direction_actual = np.sign(np.diff(self.y_actual))
        direction_forecast = np.sign(np.diff(self.y_forecast))
        return np.mean(direction_actual == direction_forecast) * 100

    def cumulative_forecast_error(self):
        return np.sum(self.y_actual - self.y_forecast)

    def forecast_bias(self):
        return np.mean(self.y_actual - self.y_forecast)

    def tracking_signal(self):
        forecast_errors = self.y_actual - self.y_forecast
        mad = np.mean(np.abs(forecast_errors))
        return np.sum(forecast_errors) / mad

    def forecast_accuracy(self):
        # Reemplazar valores reales de 0 con un número muy pequeño para evitar división por cero
        y_actual_adj = np.where(self.y_actual == 0, 1e-10, self.y_actual)
        accuracy = 1 - np.abs(self.y_actual - self.y_forecast) / y_actual_adj
        return np.mean(accuracy) * 100

    def forecast_accuracy_sum(self):
        # Reemplazar valores reales de 0 con un número muy pequeño para evitar división por cero
        y_actual_adj = np.where(self.y_actual == 0, 1e-10, self.y_actual)
        # Calcular la suma total de y_actual y y_forecast
        total_actual = np.sum(y_actual_adj)
        total_forecast = np.sum(self.y_forecast)
        # Calcular la exactitud del pronóstico en base a la suma total
        accuracy = 1 - np.abs(total_actual - total_forecast) / total_actual
        return accuracy * 100

    def hit_rate(self, lower_bound=0.69, upper_bound=1.31):
        """Calcula el hit rate basado en el rango especificado."""
        correct_predictions = np.where(
            (lower_bound < (self.y_forecast / self.y_actual))
            & ((self.y_forecast / self.y_actual) < upper_bound),
            1,
            0,
        )
        return np.mean(correct_predictions) * 100




class ModelEvaluator:
    def __init__(self, validacion, test):
        self.validacion = validacion
        self.test = test
        self.models = [
            col
            for col in self.validacion.columns
            if col.startswith("y_") and col not in ["y", "y_mean", "y_std"]
        ]
        self.best_model_dict = None

    def add_bagging_variables(self, df):
        df = df.copy()
        # df["y_promedio"] = df[self.models].median(axis=1)
        df["y_promedio"] = df[self.models].mean(axis=1)
        df["y_25"] = df[self.models].quantile(0.25, axis=1)
        df["y_75"] = df[self.models].quantile(0.75, axis=1)
        return df
        
    def evaluate_models(self):
        results = []
        for model in self.models:
            evaluator = ForecastEvaluator(self.validacion["y"], self.validacion[model])
            maen = evaluator.mean_absolute_error()
            mapen = evaluator.mean_absolute_percentage_error()
            msen = evaluator.mean_squared_error()
            rmsen = evaluator.root_mean_squared_error()
            masen = evaluator.mean_absolute_scaled_error()
            smapen = evaluator.symmetric_mape()
            mdan = evaluator.mean_directional_accuracy()
            cfen = evaluator.cumulative_forecast_error()
            biasn = evaluator.forecast_bias()
            tsn = evaluator.tracking_signal()
            hit_rate_n = evaluator.hit_rate()
            for_acc_n = evaluator.forecast_accuracy()
            for_acc_sum = evaluator.forecast_accuracy_sum()

            d = {
                 "Modelo": [model],
                 "mae": [maen],
                 "mape": [mapen],
                 "mse": [msen],
                 "rmse": [rmsen],
                 "mase": [masen],
                 "smape": [smapen],
                 "mda": [mdan],
                 "cfe": [cfen],
                 "bias": [biasn],
                 "ts": [tsn],
                 "for_acc_pro": [for_acc_n],
                 "for_acc_sum": [for_acc_sum],
                 "hit_rate": [hit_rate_n],
             }

             results.append(pd.DataFrame(data=d))

        Resultados = pd.concat(results)
        return Resultados

    def best_model_per_id(self, alfa=0.5, beta=0.5):
        best_model_dict = {}

        for unique_id in self.validacion["unique_id"].unique():
            id_subset = self.validacion[self.validacion["unique_id"] == unique_id]
            best_ponderado = np.inf
            best_model = None

            for model in self.models:
                model_preds = id_subset[model].to_numpy()
                actual_values = id_subset["y"].to_numpy()

                # Calcular RMSE
                evaluator = ForecastEvaluator(actual_values, model_preds)
                rmsen = evaluator.root_mean_squared_error()

                # Calcular DTW distance
                # Usar la función de distancia como np.linalg.norm(x - y, ord=1)
                dist, _, _, _ = dtw(
                    actual_values,
                    model_preds,
                    dist=lambda x, y: np.linalg.norm(x.flatten() - y.flatten(), ord=1),
                )

                # Ponderación combinada de RMSE y DTW distance
                # ponderado = alfa * rmsen + beta * dist
                # Ponderación del RMSE por el dist
                ponderado = rmsen * dist

                if ponderado < best_ponderado:
                    best_ponderado = ponderado
                    best_model = model

            best_model_dict[unique_id] = best_model

        self.best_model_dict = best_model_dict
        return self.best_model_dict

    def add_best_model_prediction(self, teste):
        # Ensure that test is a copy to avoid modifying the original DataFrame
        test = teste.copy()
        # Add a new column 'y_mejor_modelo' to the test data
        test["y_mejor_modelo"] = 0
        # For each unique_id, set the value of 'y_mejor_modelo' to the value of the best model for that id
        for unique_id, model in self.best_model_dict.items():
            test.loc[test["unique_id"] == unique_id, "y_mejor_modelo"] = test.loc[
                test["unique_id"] == unique_id, model
            ]
        return test

    def pipeline(self):
        # Finding the best models
        self.best_model_dict = self.best_model_per_id()
        # Adding the best model prediction to both validation and test data
        self.validacion = self.add_best_model_prediction(self.validacion)
        self.test = self.add_best_model_prediction(self.test)
        best_vars = ["y_mejor_modelo"]
        self.models += best_vars

        # Adding bagging variables to both validation and test data
        self.validacion = self.add_bagging_variables(self.validacion)
        self.test = self.add_bagging_variables(self.test)

        # Update the models to include the new variables
        bagging_vars = ["y_promedio", "y_25", "y_75"]
        self.models += bagging_vars
        # Evaluate models
        model_evaluation = self.evaluate_models()
        # Return the evaluated models and the updated DataFrames
        return model_evaluation, self.validacion, self.test


class RealValueEvaluator:
    def __init__(self, fact, predictions):
        self.df_final = self._prepare_data(fact, predictions)
        self.models = [
            col
            for col in self.df_final.columns
            if col.startswith("y_") and col not in ["y", "y_mean", "y_std"]
        ]

    def _prepare_data(self, fact, predictions):
        # Crear la clave en ambos dataframes
        predictions["key"] = (
            predictions["ds"].astype(str)
            + "-"
            + predictions["desc_country"].astype(str)
            + "-"
            + predictions["id_material"].astype(str)
        )

        fact["key"] = (
            fact["ds"].astype(str)
            + "-"
            + fact["desc_country"].astype(str)
            + "-"
            + fact["id_material"].astype(str)
        )

        pred = predictions[
            ["prediction_date", "y_mejor_modelo", "y_mediana", "y_25", "y_75", "key"]
        ]

        df_final = fact.merge(pred, how="inner", on="key")
        df_final = df_final.drop_duplicates()

        df_final = df_final[['prediction_date','unique_id', 'ds', 'id_material' ,'desc_business_category',
           'desc_category', 'desc_country', 'y', 'y_mejor_modelo', 'y_mediana', 'y_25',
           'y_75']]
        
        return df_final

    def evaluate_models(self, subgroup_col=None):
        results = []
        unique_dates = self.df_final['prediction_date'].unique()
        unique_dates.sort()  # Ordena las fechas de más antigua a más reciente

        for idx, date in enumerate(unique_dates):
            temp_df_date = self.df_final[self.df_final['prediction_date'] == date]
            
            if subgroup_col:
                unique_subgroups = temp_df_date[subgroup_col].unique()
                for subgroup in unique_subgroups:
                    temp_df = temp_df_date[temp_df_date[subgroup_col] == subgroup]
                    self._evaluate_for_model(temp_df, results, idx, subgroup_col, subgroup)
            else:
                self._evaluate_for_model(temp_df_date, results, idx)

        Resultados = pd.concat(results)
        
        # Reordenar las columnas
        columns_order = [
            "Modelo_#", "Meses_predecidos", "Modelo",
            "mae", "mape", "mse", "rmse", "mase", "smape", "mda", "cfe", "bias", "ts", "hit_rate"
        ]
        if subgroup_col and subgroup_col in Resultados.columns:
            columns_order.insert(2, subgroup_col)  # Inserta 'subgroup_col' después de 'unique_ds_count'
        
        Resultados = Resultados[columns_order]
        
        return Resultados

    def _evaluate_for_model(self, temp_df, results, idx, subgroup_col=None, subgroup=None):
        for model in self.models:
            evaluator = ForecastEvaluator(temp_df["y"], temp_df[model])
            maen = evaluator.mean_absolute_error()
            mapen = evaluator.mean_absolute_percentage_error()
            msen = evaluator.mean_squared_error()
            rmsen = evaluator.root_mean_squared_error()
            masen = evaluator.mean_absolute_scaled_error()
            smapen = evaluator.symmetric_mape()
            mdan = evaluator.mean_directional_accuracy()
            cfen = evaluator.cumulative_forecast_error()
            biasn = evaluator.forecast_bias()
            tsn = evaluator.tracking_signal()

            temp_df["model_HT"] = np.where(
                (0.69 < (temp_df[model] / temp_df["y"]))
                & ((temp_df[model] / temp_df["y"]) < 1.31),
                1,
                0,
            )
            hit_rate_n = (
                temp_df["model_HT"].sum() / temp_df["model_HT"].count()
            )
            temp_df = temp_df.drop("model_HT", axis=1)
            d = {
                "Modelo_#": [idx + 1],  # Consecutivo de la fecha
                "Meses_predecidos": [temp_df['ds'].nunique()],  # Cantidad de valores únicos en 'ds'
                "Modelo": [model],
                "mae": [maen],
                "mape": [mapen],
                "mse": [msen],
                "rmse": [rmsen],
                "mase": [masen],
                "smape": [smapen],
                "mda": [mdan],
                "cfe": [cfen],
                "bias": [biasn],
                "ts": [tsn],
                "hit_rate": [hit_rate_n]
            }
            
            if subgroup_col and subgroup:  # Si se proporciona un subgrupo, agrégalo al diccionario
                d[subgroup_col] = [subgroup]

            results.append(pd.DataFrame(data=d))
