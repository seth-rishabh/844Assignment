# -*- coding: utf-8 -*-
"""
@author: Samuel Luigi Reyes, Rishabh Seth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('dataset.csv', header=None, skiprows=lambda x: x in [0, 1, 2, ...])
data.columns = [
    'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
    'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 
    'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH',
    'SWALLOWING_DIFFICULTY', 'CHEST_PAIN', 'LUNG_CANCER'
]

# Strings to numeric
data['GENDER'] = data['GENDER'].map({'M':0, 'F':1})
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'NO':0, 'YES':1})

# Convert remaining columns from object to numeric
for i in data.columns[1:-1]:
    data[i] = pd.to_numeric(data[i])
    
data['AGE'] = pd.to_numeric(data['AGE'], errors='coerce')
    
X = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    test_size=0.3,
                                                    random_state=0)

scaler = StandardScaler()
X_trainS = scaler.fit_transform(X_train)
X_testS = scaler.transform(X_test)

nbrs = KNeighborsClassifier(n_neighbors=5)
nbrs.fit(X_trainS, y_train)

y_pred = nbrs.predict(X_testS)
XS = scaler.fit_transform(X)

aScore = cross_val_score(nbrs, XS, y, cv=10, scoring='accuracy')
fScore = cross_val_score(nbrs, XS, y, cv=10, scoring='f1')
pScore = cross_val_score(nbrs, XS, y, cv=10, scoring='precision')
rScore = cross_val_score(nbrs, XS, y, cv=10, scoring='recall')

trainAScore = cross_val_score(nbrs, X_trainS, y_train, cv=4, scoring='accuracy')
trainFScore = cross_val_score(nbrs, X_trainS, y_train, cv=4, scoring='f1')
trainPScore = cross_val_score(nbrs, X_trainS, y_train, cv=10, scoring='precision')
trainRScore = cross_val_score(nbrs, X_trainS, y_train, cv=10, scoring='recall')

print(f'Average accuracy: {np.mean(aScore):.4f}')
print(f'Average f1-score: {np.mean(fScore):.4f}')
print(f'Average precision: {np.mean(pScore):.4f}')
print(f'Average recall: {np.mean(rScore):.4f}\n')

print(f'Average accuracy (training set): {np.mean(trainAScore):.4f}')
print(f'Average f1-score (training set): {np.mean(trainFScore):.4f}')
print(f'Average precision (training set): {np.mean(trainPScore):.4f}')
print(f'Average recall (training set): {np.mean(trainRScore):.4f}\n')


cm = confusion_matrix(y_test, y_pred)
cmDisp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nbrs.classes_)
cmDisp.plot(cmap=plt.cm.Blues)
plt.show()

print('True Positives = ', cm[0,0])
print('True Negatives = ', cm[1,1])
print('False Positives = ', cm[0,1])
print('False Negatives = ', cm[1,0])
