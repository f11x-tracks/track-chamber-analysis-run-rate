import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('Book1.csv')

print(f"Original DataFrame shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# Create TRACK_RCP column: text before the '+' in RECIPE column
df['TRACK_RCP'] = df['RECIPE'].str.split('+').str[0]

# Create LAYER column: 7th and 8th character from RETICLE column (0-indexed: positions 6 and 7)
df['LAYER'] = df['RETICLE'].str[6:8]

print("\nAfter adding new columns:")
print(f"DataFrame shape: {df.shape}")
print("\nSample of new columns:")
print(df[['RECIPE', 'TRACK_RCP', 'RETICLE', 'LAYER']].head())

print("\nUnique TRACK_RCP values:")
print(df['TRACK_RCP'].unique())

print("\nUnique LAYER values:")
print(df['LAYER'].unique())

# Save the updated DataFrame
df.to_csv('Book1_processed.csv', index=False)
print("\nProcessed data saved to 'Book1_processed.csv'")