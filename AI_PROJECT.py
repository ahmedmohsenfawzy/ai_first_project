import numpy as np
import pandas as pd
import pickle
from model_logistic import model_logistic
from model_svm import model_svm
from knn import knn
from BNB import BNB
from model_forest import model_forest
import DecisionTree
import warnings

# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn import svm
# from sklearn import metrics
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import BernoulliNB
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from pre import pre

# p = pre
# r = p.test()
# x_train = r[0]
# x_test = r[1]
# y_train = r[2]
# y_test = r[3]

d = DecisionTree.DecisionTree
d.dtc([])

model_logistic.lg([])

model_svm.svm()

knn.kn()

BNB.km()

model_forest.forst()

with open('g1', 'rb') as f:
    mp1 = pickle.load(f)

with open('g2', 'rb') as g:
    mp2 = pickle.load(g)

with open('g3', 'rb') as d:
    mp3 = pickle.load(d)

with open('g4', 'rb') as s:
    mp4 = pickle.load(s)

with open('g5', 'rb') as a:
    mp5 = pickle.load(a)

with open('g6', 'rb') as q:
    mp6 = pickle.load(q)

m = input('enter the path that you want to predict: ')
dt = pd.read_csv(str(m))
dt = np.array(dt)
# dt = (dt - dt.mean()) / dt.std()
warnings.filterwarnings("ignore")
with open('v1', 'rb') as o:
    sc = pickle.load(o)
dt = sc.transform(dt)

# with open('v2', 'wb') as u:
#     pca = pickle.load(u)
# dt = pca.transform(dt)

m1 = mp1.predict(dt)
m2 = mp2.predict(dt)
m3 = mp3.predict(dt)
m4 = mp4.predict(dt)
m5 = mp5.predict(dt)
m6 = mp6.predict(dt)

k = m1 + m2 + m3 + m4 + m5
print(k)

final_pred = []
for x in k:
    if x >= 3:
        final_pred.append('M')
    else:
        final_pred.append('B')
print(final_pred)
