import pandas as pd
import numpy as np

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
    return ((series < lower_bound) | (series > upper_bound)).any()

for column in numeric_c:
    if has_outliers(df[column].dropna()):
        median = df[column].median()
        df[column] = df[column].fillna(median)
    else:
        mean = df[column].mean()
        df[column] = df[column].fillna(mean)

for column in categorical_c:
    mode = df[column].mode()[0]
    df[column] = df[column].fillna(mode)

output_file_path = 'Lab Session Data Imputed.xlsx'
df.to_excel(output_file_path, index=False)
output_file_path