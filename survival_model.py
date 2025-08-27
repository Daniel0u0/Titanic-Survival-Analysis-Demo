import pandas as pd
import statsmodels.api as sm
import numpy as np

# read data
df = pd.read_csv('train.csv')

# check columns
required_columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise KeyError(f"Missing: {missing}.")

# force numeric
numeric_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# filter NaN
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna('S')

# code Sex
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Sex'] = df['Sex'].fillna(0)

# one-hot Embarked
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# add dummies to 0
for col in ['Embarked_Q', 'Embarked_S']:
    if col not in df.columns:
        df[col] = 0

# select features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
X = df[features]
y = df['Survived']

# force float to solve object dtype
X = X.astype(float)

# Debug prints
print("X dtypes after astype:\n", X.dtypes)
print("\nX head:\n", X.head())
print("\nnp.asarray(X).dtype:\n", np.asarray(X).dtype)  # should be float64

# Drop NaN
X = X.dropna()
y = y[X.index]

# add constant
X = sm.add_constant(X)

# train model
model = sm.Logit(y, X)
result = model.fit()

# summary
print(result.summary())

# predict
df.loc[X.index, 'Predicted_Survival_Prob'] = result.predict(X)
df['Predicted_Survival'] = np.where(df['Predicted_Survival_Prob'] > 0.5, 1, 0)

# output CSV
df.to_csv('titanic_predictions.csv', index=False)
print("Created 'titanic_predictions.csv' with predictions.")

# explain results
print("The model shows that women have a higher chance of survival (positive Sex coefficient).")