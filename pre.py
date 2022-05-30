from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class pre:
    def __init__(self):
        pass

    @staticmethod
    def test():
        df = pd.read_csv('Tumor Cancer Prediction_Data.csv')

        df.drop(columns="Index", inplace=True)
        df.drop_duplicates(inplace=True)

        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
        x = df.iloc[:, :30]
        y = df["diagnosis"]

        pca = PCA(n_components=1)
        pca.fit(df)
        with open('v2', 'wb') as u:
            pickle.dump(pca, u)
        # x = (x - x.mean()) / x.std()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True, random_state=0)
        scaling = StandardScaler()
        scaling.fit(x_train)
        assert isinstance(scaling.fit_transform, object)
        x_train = scaling.fit_transform(x_train)
        x_test = scaling.transform(x_test)
        with open('v1', 'wb') as w:
            pickle.dump(scaling, w)

        var = [x_train, x_test, y_train, y_test]
        return var
