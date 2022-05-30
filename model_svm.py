import pickle
from sklearn import svm
from sklearn import metrics
from pre import pre

p = pre
r = p.test()
x_train = r[0]
x_test = r[1]
y_train = r[2]
y_test = r[3]


class model_svm:

    @staticmethod
    def svm():
        s = svm.SVC()
        s.fit(x_train, y_train)
        prd3 = s.predict(x_test)
        print('accuracy', metrics.accuracy_score(y_test, prd3) * 100, '%')
        with open('g3', 'wb') as d:
            pickle.dump(s, d)
