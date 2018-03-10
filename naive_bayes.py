import numpy as np
import math

class NaiveBayes():
    def fit(self, X, y):
        self.X, self.y = X, y
        # 独立的类别
        self.classes = np.unique(y)
        self.parameters = []
        # 对每个类别的每个特征计算均值和方差
        for i, c in enumerate(self.classes):
            X_where_c = X[np.where(y == c)]
            self.parameters.append([])
            for j in range(X.shape[1]):
                col = X_where_c[:, j]
                param = {"mean": col.mean(), "var": col.var()}
                self.parameters[i].append(param)

    # 求P(Xj=xj | Y=ck)，这里不作统计，而是使用正态分布的概率密度函数
    def _calculate_likelihood(self, mean, var, x):
        """ 正态分布的概率密度函数：f(x) = 1/σ√2π exp{-(x-μ)^2/2σ^2} """
        eps = 1e-4 # 防止分母为0
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
        return coeff * exponent

    def _calculate_prior(self, c):
        # 计算类别c的先验 (c类sample个数/总sample个数)
        X_where_c = self.X[np.where(self.y == c)]
        n_class_instances = X_where_c.shape[0]
        n_total_instances = self.X.shape[0]
        return n_class_instances / n_total_instances

    def _classify(self, sample):
        """ 返回最大后验概率P(Y|X)的分类
        P(Y|X) = P(X|Y) * P(Y) / P(X)
        Posterior = Likelihood * Prior / Scaling Factor
        其中 Scaling Factor P(X) 不影响类别分布，这里忽略不计算
        """
        posteriors = []
        # 计算所有类别的后验概率
        for i, c in enumerate(self.classes):
            # 先初始化posterior为prior
            posterior = self._calculate_prior(c)
            # 朴素贝叶斯推断: P(X|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y)*...
            for j, params in enumerate(self.parameters[i]):
                sample_feature = sample[j]
                likelihood = self._calculate_likelihood(params["mean"], params["var"], sample_feature)
                posterior *= likelihood
            posteriors.append(posterior)
        # 返回后验概率最大的类
        index_of_max = np.argmax(posteriors)
        return self.classes[index_of_max]

    def predict(self, X):
        y_pred = []
        for sample in X:
            y = self._classify(sample)
            y_pred.append(y)
        return y_pred



from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)