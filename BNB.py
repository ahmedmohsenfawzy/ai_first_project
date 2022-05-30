from sklearn import metrics
import pickle
from sklearn.naive_bayes import BernoulliNB
from pre import pre

p = pre
r = p.test()
x_train = r[0]
x_test = r[1]
y_train = r[2]
y_test = r[3]


class BNB:

    @staticmethod
    def km():

        km = BernoulliNB()
        km.fit(x_train, y_train)
        prd5 = km.predict(x_test)
        print('accuracy', metrics.accuracy_score(y_test, prd5) * 100, '%')
        with open('g5', 'wb') as a:
            pickle.dump(km, a)


