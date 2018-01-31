import numpy as np
import math

'''
感知机模型
f(x) = sign(wx+b) = {+1， if w'*x+b>=0 || -1， else}
L(w,b) = - Σ(yi(w*xi+b))
▽w L(w,b) = -Σyixi
▽b L(w,b) = -Σyi
'''
class Perceptron():
    def __init__(self, learning_rate=.1):
        self.w = None
        self.b = None
        self.learning_rate = learning_rate

    def _initialize_parameters(self, X):
        n_features = np.shape(X)[1]
        # 初始化参数范围 [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features,))
        self.b = np.random.uniform(-limit, limit, (1,))

    def fit(self, X, y, n_iterations=4000):
        self._initialize_parameters(X)
        n_samples = np.shape(X)[0]
        for t in range(n_iterations):
            # 随机梯度下降
            for i in range(n_samples):
                if self.predict(X[i])[0] != y[i]:
                    self.w += y[i] * X[i] * self.learning_rate
                    self.b += y[i] * self.learning_rate

    '''
    对偶形式
    '''
    def fit_duality(self, X, y, n_iterations=4000):
        n_samples, n_features = X.shape
        alpha = np.zeros(n_samples, dtype=np.float64)
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = np.zeros(1, dtype=np.float64)

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = np.dot(X[i], X[j])

        for t in range(n_iterations):
            for i in range(n_samples):
                if np.sign(np.sum(K[:,i] * alpha * y)) != y[i]:
                    alpha[i] += self.learning_rate
                    self.b += y[i] * self.learning_rate

        for i in range(n_samples):
            self.w += alpha[i]*y[i]*X[i]


    def predict(self, X):
        y_pred = np.round(np.sign(X.dot(self.w)+self.b))
        return y_pred.astype(int)


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = -1
    y[y == 2] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    clf = Perceptron()
    clf.fit(X_train, y_train)
    # clf.fit_duality(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)