
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df = pd.read_csv('creditcard.csv')


print(df["Class"].value_counts())
legit = df[df.Class == 0]
fraud = df[df.Class == 1]
legit.Amount.describe()
fraud.Amount.describe()
df.groupby("Class").mean()
legit_sample = legit.sample(n=492)
new_df = pd.concat([legit_sample , fraud], axis = 0)
new_df["Class"].value_counts()
new_df.groupby("Class").mean()
X = new_df.drop(columns = "Class", axis=1)
Y = new_df["Class"]
# print(X)
# print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)


# print(X.shape, X_train.shape, X_test.shape)

model = LogisticRegression()
model.fit(X_train, Y_train)

X_train_predict = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predict, Y_train)

print("ACCURACY ON TRAINING DATA:", training_data_accuracy)

X_test_predict = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_predict, Y_test)

print("ACCURACY ON TESTING DATA:", testing_data_accuracy)