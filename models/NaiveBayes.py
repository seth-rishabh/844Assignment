import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, recall_score

# Step 1: Import dataset
data = pd.read_csv('cancerDataset.csv')

# Step 2: Pre-processing

# converting all 2's, which are yes, and 1's, which are no, 
# into binary values with 1 representing yes, and 0 representing no

data = data.replace([2, 1], [1, 0])

# converting the values of gender from male and female into 0 and 1
data['GENDER'] = data['GENDER'].replace(['M', 'F'], [0, 1])

data['LUNG_CANCER'] = data['LUNG_CANCER'].replace(['YES', 'NO'], [1,0])

# Step 3: Creating the model
myGauss = GaussianNB()

X = data.drop(columns=['LUNG_CANCER'])
y = data.LUNG_CANCER

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42, stratify=y)

myGauss.fit(X_train, y_train)

y_pred = myGauss.predict(X_test)

y_training_predict = myGauss.predict(X_train)

# Step 4: Evaluating Metrics

accuracyTrain = accuracy_score(y_train, y_training_predict)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy on training set: ", round(accuracyTrain, 4)*100, "\n")

accuracy = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average = 'weighted')
precision = precision_score(y_test, y_pred, average = 'weighted')
recall = recall_score(y_test, y_pred, average = 'weighted')
confusionMatrix = confusion_matrix(y_test, y_pred)


print("Accuracy on test set:", round(accuracy,3)*100,"%")
print("F1 on test set:", round(f1,3)*100,"%")
print("Precision on test set:", round(precision,3)*100,"%")
print("Recall on test set:", round(recall, 3)*100,"%")
print("\nConfusion Matrix on test set:\n", confusionMatrix)


# K-Fold Cross-Validation

accuracy_cv = cross_val_score(myGauss, X, y, cv=10, scoring='accuracy')
f1_cv = cross_val_score(myGauss, X, y, cv=10, scoring='f1_weighted')
precision_cv = cross_val_score(myGauss, X, y, cv=10, scoring='precision_weighted')
recall_cv = cross_val_score(myGauss, X, y, cv=10, scoring='recall_weighted')

print(f"\nCross-Validation Accuracy: {np.mean(accuracy_cv*100):.4f}%")
print(f"Cross-Validation F1 Score: {np.mean(f1_cv*100):.4f}%")
print(f"Cross-Validation Precision: {np.mean(precision_cv*100):.4f}%")
print(f"Cross-Validation Recall: {np.mean(recall_cv*100):.4f}%")