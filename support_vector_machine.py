import numpy as np
import cvxopt
cvxopt.solvers.options['show_progress'] = False

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVM(object):
    def __init__(self, kernel=polynomial_kernel, C=None):
        self.kernel = kernel
        self.C = C # 惩罚项
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # 核矩阵
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i,j] = self.kernel(X[i], X[j])

        # 定义二次优化问题
        P = cvxopt.matrix(np.outer(y,y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G_max = np.diag(np.ones(n_samples) * -1)
            G_min = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = np.zeros(n_samples)
            h_min = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((h_max, h_min)))

        # 用cvxopt解决二次优化问题
        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)
        # 拉格朗日乘数
        lagr_mult = np.ravel(minimization['x'])

        # 筛选非0值的拉格朗日乘数的mask以及对应True的索引
        mask = lagr_mult > 1e-7
        indices = np.arange(len(lagr_mult))[mask]
        # 拉格朗日乘数
        self.lagr_multipliers = lagr_mult[mask]
        # 支持向量
        self.support_vectors = X[mask]
        # 支持向量对应的labels
        self.support_vector_labels = y[mask]

        # 计算截距
        self.intercept = 0
        for i in range(len(self.lagr_multipliers)):
            self.intercept += self.support_vector_labels[i]
            self.intercept -= np.sum(self.lagr_multipliers[i] * self.support_vector_labels[i] * kernel_matrix[indices[i],mask])
        self.intercept /= len(self.lagr_multipliers)

        # 为线性核计算权重矩阵weight
        if self.kernel == linear_kernel:
            self.weight = np.zeros(n_features)
            for i in range(len(self.lagr_multipliers)):
                self.weight += self.lagr_multipliers[i] * self.support_vector_labels[i] * self.support_vectors[i]
        else:
            self.weight = None

    def project(self, X):
        if self.weight is not None:
            return np.dot(X, self.weight) + self.intercept
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                prediction = 0
                for a, sv_y, sv in zip(self.lagr_multipliers, self.support_vector_labels, self.support_vectors):
                    prediction += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = prediction
            return y_predict + self.intercept

    def predict(self, X):
        return np.sign(self.project(X))

if __name__ == "__main__":
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = -1
    y[y == 2] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf = SVM(kernel=linear_kernel)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
