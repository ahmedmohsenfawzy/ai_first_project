from pre import pre
import pickle
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

p = pre
r = p.test()
x_train = r[0]
x_test = r[1]
y_train = r[2]
y_test = r[3]


class knn:

    @staticmethod
    def kn():
        kn = KNeighborsClassifier(n_neighbors=9)
        kn.fit(x_train, y_train)
        prd4 = kn.predict(x_test)
        print('accuracy', metrics.accuracy_score(y_test, prd4) * 100, '%')
        with open('g4', 'wb') as s:
            pickle.dump(kn, s)
