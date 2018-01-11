import random
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from sklearn import datasets
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

class KMeans():
    def __init__(self, k=2, max_iterations=500):
        self.k = k
        self.max_iterations = max_iterations

    def _init_random_centroids(self, X):
        """ 随机使用k个样本作为初始化的中心点 """
        n_samples = np.shape(X)[0]
        centroids = X[random.sample(range(n_samples), self.k)]
        return centroids

    def _create_clusters(self, centroids, X):
        """ 确定每个样本对应的中心点，创建cluster """
        distances = euclidean_distances(X, centroids)
        # 计算每个样本对应的中心点index
        centroid_labels = distances.argmin(axis=1)
        clusters = [X[centroid_labels==i] for i in range(self.k)]
        return centroid_labels, clusters

    def _calculate_centroids(self, clusters):
        """ 用每个cluster的样本的均值重新计算新的中心点 """
        n_features = np.shape(clusters[0])[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroids[i] = np.mean(clusters[i], axis=0)
        return centroids

    def predict(self, X):
        centroids = self._init_random_centroids(X)
        centroid_labels = np.zeros(X.shape[0])
        for _ in range(self.max_iterations):
            # 确定每个样本对应的中心点，创建cluster
            centroid_labels, clusters = self._create_clusters(centroids, X)
            # 保存当前中心点用于收敛性检验
            prev_centroids = centroids
            # 根据cluster重新计算中心点
            centroids = self._calculate_centroids(clusters)
            # 收敛
            diff = centroids - prev_centroids
            if not diff.any():
                break
        return centroid_labels

if __name__ == "__main__":
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = 0
    y[y == 2] = 1
    clf = KMeans(k=2)
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)
    accuracy = max(accuracy, 1-accuracy)
    print("Accuracy:", accuracy)