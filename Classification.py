# !pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn

import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

dataset = pd.read_csv("adult.data")

dataset.columns = ['age','workclass','fnlwgt','education',
                   'education-num','marital-status','occupation','relationship',
                   'race','sex','capital-gain','capital-loss','hours-per-week','native-country','earning']

dataset.shape

dataset.head(2)

dataset.info()

dataset.describe(include='all')

dataset.isnull().sum()

dataset.duplicated().sum()

dataset = dataset.drop_duplicates(keep='first')

dataset.duplicated().sum()

# Strip leading/trailing whitespace from the column
dataset['earning'] = dataset['earning'].str.strip()

# Replace the values
dataset['earning'] = dataset['earning'].replace({'<=50K': 0, '>50K': 1})

# Ensure the column is of integer type
dataset['earning'] = dataset['earning'].astype(int)

# Calculate the correlation matrix
correlation_matrix = dataset[['age', 'fnlwgt', 'education-num', 'capital-gain','capital-loss', 'hours-per-week', 'earning']].corr()

# Set up the figure size
plt.figure(figsize=(10, 8))

# Create the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=0.5)

# Add title
plt.title('Heat-Map showing Feature-to-Feature and Feature-to-Label’s Pearson Correlation Coefficients', fontsize=16, fontweight='bold')

# Show the plot
plt.show()

plt.figure(figsize=(16, 6))

# Create a subplot for Age
plt.subplot(1, 2, 1)
sns.boxplot(x='age', data=dataset, color='skyblue')
plt.title('Box Plot of Age', fontsize=14, fontweight='bold')
plt.xlabel('Age', fontsize=12)
plt.xticks(rotation=45)

# Create a subplot for fnlwgt
plt.subplot(1, 2, 2)
sns.boxplot(x='fnlwgt', data=dataset, color='salmon')
plt.title('Box Plot of fnlwgt', fontsize=14, fontweight='bold')
plt.xlabel('fnlwgt', fontsize=12)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 6))

# Create a subplot for education num
plt.subplot(1, 2, 1)
sns.boxplot(x='education-num', data=dataset, color='skyblue')
plt.title('Box Plot of education-num', fontsize=14, fontweight='bold')
plt.xlabel('education-num', fontsize=12)
plt.xticks(rotation=45)

# Create a subplot for hours-per-week
plt.subplot(1, 2, 2)
sns.boxplot(x='hours-per-week', data=dataset, color='salmon')
plt.title('Box Plot of hours-per-week', fontsize=14, fontweight='bold')
plt.xlabel('hours-per-week', fontsize=12)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 6))

# Create a subplot for capital-gain
plt.subplot(1, 2, 1)
sns.boxplot(x='capital-gain', data=dataset, color='skyblue')
plt.title('Box Plot of capital-gain', fontsize=14, fontweight='bold')
plt.xlabel('capital-gain', fontsize=12)
plt.xticks(rotation=45)

# Create a subplot for capital-loss
plt.subplot(1, 2, 2)
sns.boxplot(x='capital-loss', data=dataset, color='salmon')
plt.title('Box Plot of capital-loss', fontsize=14, fontweight='bold')
plt.xlabel('capital-loss', fontsize=12)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

categorical_columns = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
encoder = LabelEncoder()

for col in categorical_columns:
    dataset[col] = encoder.fit_transform(dataset[col])

dataset['earning'].value_counts()

sns.countplot(dataset, x='earning')
plt.show()

X = dataset.iloc[:, range(1,14)]
y = dataset.iloc[:,14]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

resampler = RandomOverSampler(random_state=0)
X_train_oversampled, y_train_oversampled = resampler.fit_resample(X_train, y_train)

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train_oversampled)
X_test_s = sc.transform(X_test)

sns.countplot(x=y_train_oversampled)
plt.show()

knnclassifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knnclassifier.fit(X_train_s, y_train_oversampled)

y_pred = knnclassifier.predict(X_test_s)

# Computing the accuracy and Making the Confusion Matrix
acc=metrics.accuracy_score(y_test,y_pred)
print('accuracy:%.2f\n\n'%(acc))
cm = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm,'\n\n')
print('--------------------------------------------------------')
result = metrics.classification_report(y_test, y_pred)
print("Classification Report:\n",)
print (result)

ax = sns.heatmap(cm, cmap='flare',annot=True, fmt='d')

plt.xlabel("Predicted Class",fontsize=12)
plt.ylabel("True Class",fontsize=12)
plt.title("Confusion Matrix",fontsize=12)

plt.show()

DTclassifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
DTclassifier.fit(X_train_s, y_train_oversampled)

# Predicting the Test set results
y_pred = DTclassifier.predict(X_test_s)

# Computing the accuracy and Making the Confusion Matrix
acc=metrics.accuracy_score(y_test,y_pred)
print('accuracy:%.2f\n\n'%(acc))
cm = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm,'\n\n')
print('--------------------------------------------------------')
result = metrics.classification_report(y_test, y_pred)
print("Classification Report:\n",)
print (result)

ax = sns.heatmap(cm, cmap='flare',annot=True, fmt='d')

plt.xlabel("Predicted Class",fontsize=12)
plt.ylabel("True Class",fontsize=12)
plt.title("Confusion Matrix",fontsize=12)

plt.show()