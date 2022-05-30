from sklearn.ensemble import RandomForestClassifier
from pre import pre
import pickle
from sklearn import metrics

p = pre
r = p.test()
x_train = r[0]
x_test = r[1]
y_train = r[2]
y_test = r[3]


class model_forest:
    @staticmethod
    def forst():
        forest = RandomForestClassifier(random_state=0, criterion='entropy', n_estimators=10)
        forest.fit(x_train, y_train)
        prd6 = forest.predict(x_test)
        print('accuracy', metrics.accuracy_score(y_test, prd6) * 100, '%')
        with open('g6', 'wb') as q:
            pickle.dump(forest, q)
