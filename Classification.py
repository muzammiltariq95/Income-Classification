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