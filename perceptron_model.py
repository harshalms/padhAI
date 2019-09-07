import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.datasets
from sklearn.metrics import accuracy_score

class Perceptron:
    def __init__(self):
        self.b = None
        self.w = None

    def model(self, w, x):
        return 1 if np.dot(self.w,x)>=self.b else 0

    def predict(self, X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)

    def fit(self, X, Y):
        accuracy = {}
        self.w = np.ones(X.shape[1])
        self.b = 0
        for x,y in zip(X, Y):
            y_pred = self.model(x)
            if y==1 and y_pred==0:
                self.w = self.w + x
                self.b = self.b - 1
            elif y==0 and y_pred==1:
                self.w = self.w - x
                self.b = self.b + 1


breast_cancer = sklearn.datasets.load_breast_cancer()
# print(breast_cancer)
data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
data['class'] = pd.DataFrame(breast_cancer.target)
# print(data)
X = data.drop(['class'], axis=1)
Y = data['class']
X_train,X_test,Y_train,Y_test = train_test_split(X, Y)
# converting into numpy array
X_train = X_train.values
X_test = X_test.values
# creating an instance
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred_train = perceptron.predict(X_train)
print('accuracy score :', accuracy_score(Y_pred_train, Y_train))