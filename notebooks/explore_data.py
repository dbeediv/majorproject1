import pandas as pd

# Load the dataset
df = pd.read_excel("data_utils/raw_data/insurance_claims.xlsx")

print("=== Shape ===")
print(df.shape)  # (rows, columns)

print("\n=== Column names ===")
print(df.columns.tolist())

print("\n=== Data types ===")
print(df.dtypes)

print("\n=== Missing values per column ===")
print(df.isnull().sum())

print("\n=== First 5 rows ===")
print(df.head())

print("\n=== Fraud rate ===")
print(df['fraud_reported'].value_counts())
print(df['fraud_reported'].value_counts(normalize=True))

print("\n=== authorities_contacted unique values ===")
print(df['authorities_contacted'].unique())