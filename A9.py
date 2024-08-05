import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

file_path = 'Lab Session Data (1).xlsx'

df1 = pd.read_excel(file_path, sheet_name='Purchase data')
df2 = pd.read_excel(file_path, sheet_name='IRCTC Stock Price')
df3 = pd.read_excel(file_path, sheet_name='thyroid0387_UCI')
df4 = pd.read_excel(file_path, sheet_name='marketing_campaign')

def preprocess_feature(df, feature_column):
    df[feature_column] = pd.to_numeric(df[feature_column], errors='coerce')
    df[feature_column] = df[feature_column].fillna(0)
    return df


def calculate_cosine_similarity(df):
    if len(df) < 2:
        print("Not enough data in the DataFrame.")
        return

    feature_column = df.columns[1]

    df = preprocess_feature(df, feature_column)

    v1 = df.iloc[0][feature_column]
    v2 = df.iloc[1][feature_column]

    print(f"Values being compared: {v1} (Document 1) vs {v2} (Document 2)")

    v1_vector = np.array([[v1]])
    v2_vector = np.array([[v2]])

    similarity = cosine_similarity(v1_vector, v2_vector)[0, 0]

    print(f"Cosine Similarity between the second features: {similarity}")

print("DataFrame 1:")
calculate_cosine_similarity(df1)
print("\nDataFrame 2:")
calculate_cosine_similarity(df2)
print("\nDataFrame 3:")
calculate_cosine_similarity(df3)
print("\nDataFrame 4:")
calculate_cosine_similarity(df4)