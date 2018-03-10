import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class KNN():
    def __init__(self, k=5):
        self.k = k

    # 这里是线性搜索的暴力方法，未实现较复杂的kd树的算法
    def predict(self, X_test, X_train, y_train):
        y_pred = np.empty(X_test.shape[0])
        # 计算欧氏距离矩阵
        distances = euclidean_distances(X_test, X_train)
        # 对X_test[line_i]按feature排序的索引值代替原先的数值表示
        distances_argsort = distances.argsort(axis=1)
        # k个最小距离的列索引
        k_min_indices = np.argwhere(distances_argsort<self.k)[:,1].reshape((-1,self.k))
        # k个列索引对应的labels
        labels = y_train[k_min_indices]
        for i in range(len(X_test)):
            # 投票label
            y_pred[i] = np.bincount(labels[i]).argmax()
        return y_pred



from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    data = datasets.load_iris()
    X = normalize(data.data)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf = KNN(k=5)
    y_pred = clf.predict(X_test, X_train, y_train)
    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)
