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
        # Se modifica el bucle para iterar sobre semestres en lugar de meses
        for semestre in range(1, self.semestres_a_predecir + 1):
            # Añadir 6 meses al final de la fecha para obtener el próximo semestre
            fechas_a_predecir = ultima_fecha + MonthBegin(semestre * 6)
            # Aumentar el índice de tiempo por semestre (6 meses)
            time_index_predecir = max_time_index + semestre * 6
            df_prediccion_semestre = pd.DataFrame(
                {
                    "unique_id": self.df["unique_id"].unique(),
                    "ds": [fechas_a_predecir] * len(self.df["unique_id"].unique()),
                    "time_index": time_index_predecir,
                }
            )
            dataframes_prediccion.append(df_prediccion_semestre)
        self.df_prediccion = pd.concat(dataframes_prediccion)

    
    def join_dataframes(self):
        self.df_final = pd.merge(
            self.df_prediccion, self.df_unico, on="unique_id", how="left"
        )
        self.df_final["ano"] = self.df_final["ds"].dt.year - 2009
        self.df_final["year"] = self.df_final["ds"].dt.year
        # Aquí calculamos el semestre en lugar del mes
        self.df_final['semestre'] = (self.df_final['ds'].dt.month - 1) // 6 + 1

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
        return self.df, self.df_final, pd.concat([self.df, self.df_final])
