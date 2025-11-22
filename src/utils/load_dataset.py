import pandas as pd, os, sys, pathlib
import logging


#===================  FONCTION POUR CHARGMEMENT DES DONNÉES =============
DATA_PATH = pathlib.Path(__file__).parent.parent.parent.parent / "backend/src/data/"
file_1 = "dataset_2050.csv"
file_2 = "df_mortality_2019_2025.csv"
def load_data():
    try:
        data_2050 = pd.read_csv(DATA_PATH/file_1)
        data_2022_mortality = pd.read_csv(DATA_PATH/file_2)
        return data_2022_mortality, data_2050
    except Exception as e:
        logging.error(f"erreur de chargement de fichier {e}")
        

data_2050, data_2022_mortality = load_data()
def preprocess_data_2050(data_2050):
    # Renommage des colonnes pour une meilleure lisibilité
    df = data_2050.rename(columns={
            "Cancer id": "cancer_id",
            "Cancer label": "cancer_label",
            "Population id": "population_id",
            "Population": "population",
            "Sex": "sex",
            "Type": "type",
            "Cases base in 2022": "cases_2022",
            "Year": "year",
            "Prediction": "prediction",
            "Change in number of cases": "change_total",
            "Change in number of cases due to population": "change_population",
            "Change in number of cases due to risk": "change_risk"
        })

        # Casts utiles
    for col in ("cases_2022", "prediction", "change_total", "change_population", "change_risk"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce")

        # Nettoyage basique : drop rows entièrement vides
        df = df.dropna(axis=0, how="all")

        return df

df = preprocess_data_2050(load_data()[1])
print(df.head())
#print(data_2022_mortality.head())