import pandas as pd

# Read Titanic dataset
df = pd.read_csv('train.csv')

# count Survived
survived_count = df['Survived'].sum()
print(f"People Survived: {survived_count}")

# filter (Survived == 1)
survived_df = df[df['Survived'] == 1]

# export CSV
survived_df.to_csv('survived_passengers.csv', index=False)
print("Generated 'survived_passengers.csv', include all survived passangers details")

# Display first few rows of the filtered DataFrame
print(survived_df.head())