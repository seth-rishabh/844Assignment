import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, recall_score
from sklearn.datasets import make_classification


# Step 1: Import dataset
data = pd.read_csv('cancerDataset.csv')

# Step 2: Pre-processing

# converting all 2's, which are yes, and 1's, which are no, 
# into binary values with 1 representing yes, and 0 representing no

data = data.replace([2, 1], [1, 0])

# converting the values of gender from male and female into 0 and 1
data['GENDER'] = data['GENDER'].replace(['M', 'F'], [0, 1])

data['LUNG_CANCER'] = data['LUNG_CANCER'].replace(['YES', 'NO'], [1,0])

# Step 3: Creating the Decision Tree

X = data.drop(columns=['LUNG_CANCER'])
y = data.LUNG_CANCER

#X, y = make_classification(n_samples=500, n_features=5, random_state=42)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

myTree = DecisionTreeClassifier()

myTree = myTree.fit(X_train, y_train)

y_pred = myTree.predict(X_test)

# Step 4: Evaluating Metrics
accuracy = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average = 'weighted')
precision = precision_score(y_test, y_pred, average = 'weighted')
recall = recall_score(y_test, y_pred, average = 'weighted')
confusionMatrix = confusion_matrix(y_test, y_pred)

print("Accuracy on test set:", accuracy)
print("F1 on test set:", f1)
print("Precision on test set:", precision)
print("Recall on test set:", recall)
print("Confusion Matrix on test set:\n", confusionMatrix)


"""
# K-Fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform Cross-Validation
accuracy_cv = cross_val_score(myTree, X, y, cv=kf, scoring='accuracy')
f1_cv = cross_val_score(myTree, X, y, cv=kf, scoring='f1_weighted')
precision_cv = cross_val_score(myTree, X, y, cv=kf, scoring='precision_weighted')
recall_cv = cross_val_score(myTree, X, y, cv=kf, scoring='recall_weighted')

# Display Mean Cross-Validation Scores
print(f"Cross-Validation Accuracy: {np.mean(accuracy_cv):.4f}")
print(f"Cross-Validation F1 Score: {np.mean(f1_cv):.4f}")
print(f"Cross-Validation Precision: {np.mean(precision_cv):.4f}")
print(f"Cross-Validation Recall: {np.mean(recall_cv):.4f}")

"""
 