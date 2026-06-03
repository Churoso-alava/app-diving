import pandas as pd

def test_df_grupal_init():
    atletas_list = ["Atleta1", "Atleta2"]
    df_grupal = pd.DataFrame({
        "Atleta": atletas_list,
        "VMP Hoy": [0.0] * len(atletas_list)
    })
    print(df_grupal)
    assert df_grupal["VMP Hoy"].iloc[0] == 0.0
    assert df_grupal["VMP Hoy"].iloc[1] == 0.0
    assert len(df_grupal) == 2

if __name__ == "__main__":
    test_df_grupal_init()
    print("Test passed!")
