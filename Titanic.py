# Hello!

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score , recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("Titanic-Dataset.csv")
print(data.head())

data.shape
data.info()
data.describe()
data.columns.tolist()

data_dtypes = data.dtypes
numericaldata = data_dtypes

numerical_data_df_cols = data_dtypes[data_dtypes.apply(lambda x: np.issubdtype(x, np.number))].index.tolist()
categorical_data_df_cols = data_dtypes[~data_dtypes.apply(lambda x: np.issubdtype(x, np.number))].index.tolist()

numerical_data = data[numerical_data_df_cols]
categorical_data = data[categorical_data_df_cols]
print(f"Numerical Columns: {len(numerical_data_df_cols)}, Categorical Columns: {len(categorical_data_df_cols)}")

print(categorical_data_df_cols)
print(numerical_data_df_cols)

data = data.drop(columns = ['Name', 'Ticket'])
print(data.columns.tolist())

missingvalues = data.isnull().sum()
missingpercentage = (missingvalues / len(data)) * 100
missing_df = pd.DataFrame({'missingvalues': missingvalues, 'missingpercentage': missingpercentage })
print(missing_df[missing_df['missingvalues'] > 0])

data = data.drop(columns = ['Cabin'])

missing_embarked = data['Embarked'].isnull()

print(data[missing_embarked])

data = data.drop([61,829])
print(len(data))

print(data[missing_embarked])

data['Age'].fillna(data['Age'].median(), inplace=True)
print(data['Age'].isnull().sum())
data_encoded = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first= True)
data_encoded.to_csv('Encoded_data.csv', index= False)

x = data_encoded.drop(columns= ['Survived'])
y = data_encoded['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
weight = model.fit(x_train ,y_train)
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))