from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import pickle
from pre import pre

p = pre
r = p.test()
x_train = r[0]
x_test = r[1]
y_train = r[2]
y_test = r[3]


class DecisionTree:

    @staticmethod
    def dtc(self):
        dtc = DecisionTreeClassifier(criterion='entropy', random_state=0)
        dtc = dtc.fit(x_train, y_train)
        prd1 = dtc.predict(x_test)

        print('accuracy', metrics.accuracy_score(y_test, prd1) * 100, '%')
        with open('g1', 'wb') as f:
            pickle.dump(dtc, f)
