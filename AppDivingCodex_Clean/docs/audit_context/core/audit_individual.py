import pandas as pd
import scipy.stats as stats

df = pd.read_csv('sesiones_rows (1).csv')
grouped = df.groupby('nombre')

print(f"{'Atleta':<10} | {'Count':<5} | {'Mean':<6} | {'Std':<6} | {'P-Value':<10}")
print('-' * 55)

for name, group in grouped:
    data = group['vmp_hoy'].dropna()
    count = len(data)
    if count < 3:
        p_str = 'N/A'
    else:
        stat, p = stats.shapiro(data)
        p_str = f"{p:.4f}"
    
    print(f"{name:<10} | {count:<5} | {data.mean():.3f} | {data.std():.3f} | {p_str:<10}")
