import pandas as pd

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza los nombres de columnas de un DataFrame a snake_case.
    Convierte a minúsculas, reemplaza espacios por guion bajo y elimina espacios extra.
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df