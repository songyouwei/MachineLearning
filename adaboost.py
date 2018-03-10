import math
import numpy as np

# 这里使用决策桩作为弱分类器，是一级决策树
class DecisionStump():
    def __init__(self):
        # 给定阈值后确定样本是否应归类为-1或1
        self.polarity = 1
        # 用来做分类的特征的index
        self.feature_index = None
        # 特征所对比的阀值
        self.threshold = None
        # 分类器准确度
        self.alpha = None


class Adaboost():
    def __init__(self, n_clf=5):
        # n_clf是弱分类器的个数
        self.n_clf = n_clf

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)
        # 初始化权重都为 1/n_samples
        w = np.full(n_samples, (1 / n_samples))
        # 弱分类器们
        self.clfs = []
        # 对弱分类器们迭代
        for _ in range(self.n_clf):
            # 当前决策桩
            clf = DecisionStump()
            # 使用某个特征值阈值预测样本标签的最小误差
            min_error = float('inf')
            # 遍历每个特征，看哪个特征的哪个值是可以预测y的最佳阈值
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                # 尝试此feature每一个唯一的值作为阈值
                for threshold in unique_values:
                    # 分类器的归类标签polarity=1
                    p = 1
                    # 预测值，全初始化为'1'
                    prediction = np.ones(np.shape(y))
                    # 将值低于阈值的样本标记为'-1'
                    prediction[X[:, feature_i] < threshold] = -1
                    # error为误分类样本权重的和
                    error = sum(w[y != prediction])

                    # 如果error超过50%，error和polarity都反转过来
                    # E.g error = 0.8 => (1 - error) = 0.2
                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # 如果此阈值导致最小的错误，就保存此配置
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        min_error = error

            # 计算用于更新样本权重的alpha（代表此分类器的能力）
            clf.alpha = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
            # 预测值，全初始化为'1'
            predictions = np.ones(np.shape(y))
            # 低于阈值的样本的indexes
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            # 对负样本标记为-1
            predictions[negative_idx] = -1
            # 计算新的权重，增大误分类的样本权重，减小正确分类的样本的权重
            w *= np.exp(-clf.alpha * y * predictions)
            # 归一化
            w /= np.sum(w)

            # 保存此分类器
            self.clfs.append(clf)

    def predict(self, X):
        n_samples = np.shape(X)[0]
        y_pred = np.zeros((n_samples, 1))
        # 用每个分类器来标记样本
        for clf in self.clfs:
            # 预测值，全初始化为'1'
            predictions = np.ones(np.shape(y_pred))
            # 低于阈值的样本的indexes
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            # 对负样本标记为-1
            predictions[negative_idx] = -1
            # 使用alpha加权计算预测值
            y_pred += clf.alpha * predictions

        # 返回sign(y_pred)
        y_pred = np.sign(y_pred).flatten()
        return y_pred


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    data = datasets.load_digits()
    X = data.data
    y = data.target
    digit1 = 1
    digit2 = 8
    idx = np.append(np.where(y == digit1)[0], np.where(y == digit2)[0])
    y = data.target[idx]
    y[y == digit1] = -1
    y[y == digit2] = 1
    X = data.data[idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    clf = Adaboost(n_clf=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
