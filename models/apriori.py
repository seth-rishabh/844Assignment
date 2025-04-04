# -*- coding: utf-8 -*-
"""
@author: Samuel Luigi Reyes, Rishabh Seth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori

data = pd.read_csv('dataset.csv', header=None, skiprows=lambda x: x in [0, 1, 2, ...])
data.columns = [
    'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
    'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 
    'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH',
    'SWALLOWING_DIFFICULTY', 'CHEST_PAIN', 'LUNG_CANCER'
]

bins = [30, 40, 50, 60, 70, 80]
labels = ['30-39', '40-49', '50-59', '60-69', '70-80']
data['AGE_GROUP'] = pd.cut(data['AGE'], bins=bins, labels=labels)
boolCol = data.columns.difference(['AGE', 'GENDER', 'LUNG_CANCER'])
data[boolCol] = data[boolCol].replace({1: "Yes", 2: "No"})
data = data.astype(str)

# dataList = data.values.tolist()
dataList = []
for j, row in data.iterrows():
    rename = [f'{col}={val}' for col, val in row.items()]
    dataList.append(rename)

apr = apriori(dataList, min_support=0.28, min_confidence=0.5)

output = []
for i in apr:
    for ordered_stat in i.ordered_statistics:
        output.append((
            set(ordered_stat.items_base),
            set(ordered_stat.items_add),
            i.support,
            ordered_stat.confidence))
        
for x, y, a, b in output:
    xStr = ', '.join(x) if x else '(Empty Set)'
    yStr = ', '.join(y)
    print(f'Rule: {xStr} -> {yStr}')
    print(f'Support: {a:.2f}, Confidence: {b:.2f}\n')