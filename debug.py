import pandas as pd
df = pd.read_csv('train.csv')
print(df.dtypes)
print(df['Fare'].apply(type).value_counts())  # 檢查 Fare 类型