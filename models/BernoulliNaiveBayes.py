import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, recall_score

# Step 1: Import dataset
data = pd.read_csv('cancerDataset.csv')

# Step 2: Pre-processing

# converting all 2's, which are yes, and 1's, which are no, 
# into binary values with 1 representing yes, and 0 representing no

data = data.replace([2, 1], [1, 0])

# converting the values of gender from male and female into 0 and 1
data['GENDER'] = data['GENDER'].replace(['M', 'F'], [0, 1])

# converting the class values into 0 (no) and 1 (yes)
data['LUNG_CANCER'] = data['LUNG_CANCER'].replace(['YES', 'NO'], [1,0])


# Step 3: Creating the model
myBernoulli = BernoulliNB()

X = data.drop(columns=['LUNG_CANCER'], axis=1)
y = data.LUNG_CANCER

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42, stratify=y)

myBernoulli.fit(X_train, y_train)

y_pred = myBernoulli.predict(X_test)

y_training_predict = myBernoulli.predict(X_train)

# Step 4: Evaluating Metrics

accuracyTrain = accuracy_score(y_train, y_training_predict)

accuracy = accuracy_score(y_test, y_pred)

print("Training accuracy: ", accuracy)
print("Test accuracy: ", accuracy)