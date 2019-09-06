import numpy as np 
import pandas as pd 
import sklearn.datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class MPNeuron:

    def __init__(self):
        self.b = None
    
    def model(self, x):
        return (sum(x) >= self.b)

    def predict(self, X):
        
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)

    def fit(self, X, Y):
        accuracy = {}
        for b in range(X.shape[1]+1):
            self.b = b
            Y_pred = self.predict(X)
            accuracy[b] = accuracy_score(Y_pred, Y)

        best_b = max(accuracy, key = accuracy.get)
        self.b = best_b

        print('Optimal value of b is', best_b)
        print('Highest accuracy is', accuracy[best_b])

breast_cancer = sklearn.datasets.load_breast_cancer()
X = breast_cancer.data
Y = breast_cancer.target
# print(X.shape, Y.shape) # (569,30) (569,)
data=pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
data['class']=breast_cancer.target
# print(data.head())
print(data.describe())
print(data['class'].value_counts())
print(breast_cancer.feature_names)
print(breast_cancer.target_names)
print(data.groupby('class').mean())

# Train-Test split
X=data.drop('class', axis=1)
Y=data['class']
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Binarization of input
X_binarised_train=X_train.apply(pd.cut, bins=2, labels=[1,0])
X_binarised_test=X_test.apply(pd.cut, bins=2, labels=[1,0])
# plt.plot(X_binarised_train.T, '*')
# plt.xticks(rotation='vertical')
# plt.show()

# numpy.ndarray
X_binarised_test=X_binarised_test.values
X_binarised_train=X_binarised_train.values
# print('X-bin --------->>>>>',type(X_binarised_train))

# creating an instance of class MPNeuron
mp_neuron = MPNeuron()
mp_neuron.fit(X_binarised_train, Y_train)