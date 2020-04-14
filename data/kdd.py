import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class KDD:

    anomaly_classes = ["normal."]
    multi_categorical_vars = [1, 2, 3]
    train_perc = 0.6
    val_perc = 0.2
    test_perc = 0.2

    def __init__(self, data_path):

        self.data_dict = pickle.load(open(data_path, "rb"))
        self.X = self.data_dict["data"]
        self.y = self.data_dict["target"].astype(str)
        self.non_multi_cat_vars = [i for i in range(
            self.X.shape[1]) if i not in self.multi_categorical_vars]

        self.encoders = [OneHotEncoder(sparse=False, drop='first')
                         for _ in range(len(self.multi_categorical_vars))]

        self._replace_categorical_one_hot()
        self._encode_anomalies()
        self._split_train_val_test()

    def _encode_categorical(self):

        categorical_data = []
        for i in range(len(self.multi_categorical_vars)):
            var = self.multi_categorical_vars[i]
            one_hot = self.encoders[i].fit_transform(
                self.X[:, var].reshape(-1, 1))
            categorical_data.append(one_hot)

        return np.concatenate(categorical_data, axis=1)

    def _replace_categorical_one_hot(self):

        categorical = self._encode_categorical()
        X = self.X[:, self.non_multi_cat_vars]
        self.X = np.concatenate([X, categorical], axis=1)

    def _encode_anomalies(self):

        self.y = np.isin(self.y, self.anomaly_classes).astype(int)

    def _split_train_val_test(self):

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_perc, random_state=42)

        if self.val_perc > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=self.val_perc/self.train_perc)

            self._save_data(X_val, y_val, "kdd_val.pickle")

        X_train, y_train = self._remove_anomalies_from_train(X_train, y_train)

        self._save_data(X_train, y_train, "kdd_train.pickle")
        self._save_data(X_test, y_test, "kdd_test.pickle")

    def _remove_anomalies_from_train(self, X, y):
        return (X[y == 0, :], y[y == 0])

    def _save_data(self, X, y, path):
        pickle.dump((X, y), open(path, "wb"))


if __name__ == "__main__":
    kdd = KDD('kdd99.pickle')
