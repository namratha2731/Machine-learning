import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

file_path='Lab Session Data (1).xlsx'
sheet_name='thyroid0387_UCI'

df = pd.read_excel(file_path, sheet_name=sheet_name)

df.replace('?', np.nan, inplace=True)

numeric_c = df.select_dtypes(include = ['float64', 'int64']).columns
categorical_c = df.select_dtypes(include = ['object', 'category']).columns

def has_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    out = ((series < lower_bound) | (series > upper_bound))
    return out.any()

def normalize_data(series):
    outliers = has_outliers(series)
    if outliers:
        scaler = MinMaxScaler()
        normalize_data = scaler.fit_transform(series.values.reshape(-1, 1))
        return normalize_data.flatten()
    else:
        return stats.zscore(series)

for col in numeric_c:
    df[col] = normalize_data(df[col])

print(df[numeric_c])