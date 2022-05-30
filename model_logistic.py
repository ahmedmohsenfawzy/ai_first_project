import pickle
from pre import pre
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

p = pre
r = p.test()
x_train = r[0]
x_test = r[1]
y_train = r[2]
y_test = r[3]


class model_logistic:

    @staticmethod
    def lg(self):
        lg = LogisticRegression()
        lg = lg.fit(x_train, y_train)
        prd2 = lg.predict(x_test)
        print('accuracy', metrics.accuracy_score(y_test, prd2) * 100, '%')
        with open('g2', 'wb') as g:
            pickle.dump(lg, g)
