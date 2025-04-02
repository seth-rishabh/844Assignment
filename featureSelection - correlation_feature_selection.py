import pandas as pd
import matplotlib.pyplot as plt

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
bins = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]  # Adjusted for given age range
labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Assign numerical labels

data['AGE'] = pd.cut(data['AGE'], bins=bins, labels=labels, include_lowest=True)

# Step 3: Calculating Correlation-based Feature Selection

corr_matrix = data.corr()

corr_with_classification = corr_matrix['LUNG_CANCER']

# k is how many features we want, choose 15, and then looking at all features remomve some
k = 16
top_k = corr_with_classification.abs().sort_values(ascending=False)[:k].index
selected_features = data[top_k]

selected_corr_matrix = selected_features.corr()
print(selected_corr_matrix)

# Step 4: Plotting the matrix

plt.figure(figsize=(10, 8))
plt.title("Correlation matrix for selected features")
plt.imshow(selected_corr_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(selected_corr_matrix.columns)), selected_corr_matrix.columns, rotation=90)
plt.yticks(range(len(selected_corr_matrix.columns)), selected_corr_matrix.columns)
plt.show()
