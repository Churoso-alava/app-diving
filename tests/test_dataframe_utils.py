import pandas as pd
from utils.dataframe_utils import normalize_columns

def test_normalize_columns():
    df = pd.DataFrame(
        columns=["VMP Hoy", "  Nombre Atleta ", "ValorTotal"]
    )
    df_norm = normalize_columns(df)
    assert list(df_norm.columns) == ["vmp_hoy", "nombre_atleta", "valortotal"], "Las columnas no fueron normalizadas correctamente"

if __name__ == "__main__":
    test_normalize_columns()
    print("Test passed ✔️")