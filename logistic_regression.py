import numpy as np
import math

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

class LogisticRegression():
    def __init__(self, learning_rate=.1, gradient_descent=True):
        self.param = None
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent

    def _initialize_parameters(self, X):
        n_features = np.shape(X)[1]
        # 初始化参数范围 [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        y = self.sigmoid(x)
        return y * (1 - y)

    def fit(self, X, y, n_iterations=4000):
        self._initialize_parameters(X)
        for i in range(n_iterations):
            # 预测y
            y_pred = self.sigmoid(X.dot(self.param))
            # 误差
            err = y - y_pred
            if self.gradient_descent:
                # 使用梯度下降
                # f(w)=wTx，对w计算出的梯度就是x:
                # ▽wf(w) = df(w)/dw = dwT.x/dw = x
                self.param += self.learning_rate * err.dot(X)
            else:
                gradient = self.gradient(X.dot(self.param))
                diag_gradient = np.diag(gradient)
                # batch优化:??
                self.param = np.linalg.pinv(X.T.dot(diag_gradient).dot(X)) \
                    .dot(X.T) \
                    .dot(diag_gradient.dot(X).dot(self.param) + err)

    def predict(self, X):
        y_pred = np.round(self.sigmoid(X.dot(self.param)))
        return y_pred.astype(int)

if __name__ == "__main__":
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = 0
    y[y == 2] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)