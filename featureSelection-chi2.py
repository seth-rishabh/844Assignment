import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2

# Step 1: Load data
data = pd.read_csv('cancerDataset.csv')

# Step 2: Pre-processing the data to use chi squared for feature selection, specifically using filter approach

# converting all 2's, which are yes, and 1's, which are no, 
# into binary values with 1 representing yes, and 0 representing no
data = data.replace([2, 1], [1, 0])

# converting the class values from yes and no into binary values with yes being 1 and no being 0
data['LUNG_CANCER'] = data['LUNG_CANCER'].replace(['YES', 'NO'], [1,0])

# converting the values of gender from male and female into 0 and 1
data['GENDER'] = data['GENDER'].replace(['M', 'F'], [0, 1])

# age needs to be put into bins as it is continious data
bins = [30, 40, 50, 60, 70, 80]  # Adjusted for given age range
labels = [1, 2, 3, 4, 5]  # Assign numerical labels

data['AGE'] = pd.cut(data['AGE'], bins=bins, labels=labels, include_lowest=True)

# Step 3: Calculating the chi squared statistics and p-value for each feature

X = data.drop(columns='LUNG_CANCER')
y = data.LUNG_CANCER

chi2_stats, p_values = chi2(X, y) 

for i in range(len(X.columns)):
    feature = X.columns[i]
    chi2_stat = chi2_stats[i]
    p_val = p_values[i]
    print(feature,": \nchi square stat: ", chi2_stat," p-value: ",p_val,"\n")


# Step 4: Graphing the chi squared statistic and p-values

graphed_chi_Stats = pd.Series(chi2_stats,index = X.columns)
graphed_chi_Stats.sort_values(ascending = False , inplace = True)
graphed_chi_Stats.plot.bar()
plt.show()

graphed_p_values = pd.Series(p_values,index = X.columns)
graphed_p_values.sort_values(ascending = False , inplace = True)
graphed_p_values.plot.bar()
plt.show()







