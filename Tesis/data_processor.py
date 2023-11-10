import numpy as np
import pandas as pd
from datetime import datetime
from pandas.tseries.offsets import MonthBegin


class ClasePrediccion:
    def __init__(self, df, semestres_a_predecir):
        self.df = df
        self.semestres_a_predecir = semestres_a_predecir

    def crear_dataframe_unico(self):
        self.df_unico = self.df.copy()
        self.df_unico = self.df_unico[
            [
                "Rectoria",
                "Modalidad",
                "objetivo",
                "unique_id",
                "y_mean",
                "y_std",
                "upper_limit",
                "lower_limit",
            ]
        ]
        self.df_unico = self.df_unico.drop_duplicates()

    def obtener_observaciones(self):
        self.df["ds"] = pd.to_datetime(self.df["ds"])
        ultima_fecha = self.df["ds"].max()
        max_time_index = self.df["time_index"].max()
        dataframes_prediccion = []
        unique_ids = self.df["unique_id"].unique()
        for i in range(1, self.semestres_a_predecir + 1):
            # Incrementar en 6 meses por cada semestre
            nueva_fecha = ultima_fecha + pd.DateOffset(months=i * 6)
            nuevo_time_index = max_time_index + i
            # Crear un dataframe por cada semestre
            df_prediccion_semestre = pd.DataFrame({
                "ds": [nueva_fecha] * len(unique_ids),
                "time_index": [nuevo_time_index] * len(unique_ids),
                "unique_id": unique_ids,
            })
            dataframes_prediccion.append(df_prediccion_semestre)
        self.df_prediccion = pd.concat(dataframes_prediccion).reset_index(drop=True)


    def join_dataframes(self):
        # Combina el DataFrame de predicción con el DataFrame único
        self.df_final = pd.merge(
            self.df_prediccion, self.df_unico, on="unique_id", how="left"
        )
        # Calcula 'ano' y 'semestre' basado en 'ds'
        self.df_final["ano"] = self.df_final["ds"].dt.year
        self.df_final["semestre"] = self.df_final["ds"].dt.month.apply(lambda x: 1 if x <= 6 else 2)



    def get_test_dataframe(self):
        self.df_final = self.df_final[
            [
                "ano",
                "semestre",  # Asegúrate de incluir 'semestre' en lugar de 'mes'
                "Rectoria",
                "Modalidad",
                "objetivo",
                "unique_id",
                "y_mean",
                "y_std",
                "upper_limit",
                "lower_limit",
                "time_index",
            ]
        ]
        return self.df_final
    def get_train_dataframe(self):
        return self.df

    def get_combined_dataframe(self):
        return pd.concat([self.df, self.df_final])

    def procesar(self):
        self.crear_dataframe_unico()
        self.obtener_observaciones()
        self.join_dataframes()
        self.get_train_dataframe()
        self.get_test_dataframe()
        self.get_combined_dataframe()
        self.df_final['ds'] = pd.to_datetime(self.df_final['ano'].astype(str) + '-' + (self.df_final['semestre'] * 6).astype(str) + '-01')

        return self.df, self.df_final, pd.concat([self.df, self.df_final])
