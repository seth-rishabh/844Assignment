# -*- coding: utf-8 -*-
"""
@author: Samuel Luigi Reyes, Rishabh Seth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('dataset.csv', header=None)
data.columns = [
    'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
    'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 
    'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH',
    'SWALLOWING_DIFFICULTY', 'CHEST_PAIN', 'LUNG_CANCER'
]

# Strings to numeric, drop missing values
data['GENDER'] = data['GENDER'].map({'M':0, 'F':1})
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'NO':0, 'YES':1})
data = data.dropna()

features = data.columns[:-1]
for i in features:
    data[i] = pd.to_numeric(data[i], errors='coerce')

# Anomaly detection
scaler = StandardScaler()
X = scaler.fit_transform(data[features])

lof = LocalOutlierFactor(n_neighbors=5, contamination=0.05, novelty=False)
outliers = lof.fit_predict(X)

data['anomalyScore'] = lof.negative_outlier_factor_
data['isOutlier'] = np.where(outliers == -1, 1, 0)

print(f"Number of outliers: {data['isOutlier'].sum()}\n")
plt.figure(figsize=(10,6))
plt.scatter(data['AGE'], data['anomalyScore'], c=data['isOutlier'], cmap='coolwarm')
plt.xlabel('Age')
plt.ylabel('Anomaly Score')
plt.show()