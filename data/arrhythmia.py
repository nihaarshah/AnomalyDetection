import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Arrhythmia:

    """Arrhythmia dataset from http://odds.cs.stonybrook.edu/arrhythmia-dataset/"""

    def __init__(self, data_path, train_perc=0.75, test_perc=0.25, val_perc=0,  scaling=True):

        self.data_dict = pickle.load(open(data_path, "rb"))
        self.X = self.data_dict["X"]
        self.y = self.data_dict["y"].reshape(-1)
        self.scaler = StandardScaler()
        self.train_perc = train_perc
        self.test_perc = test_perc
        self.val_perc = val_perc
        self.scaling = scaling
        self._split_train_val_test()

    def _split_train_val_test(self):

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_perc, random_state=42)

        if self.val_perc > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=self.val_perc/self.train_perc,  random_state=42)

            if self.scaling:
                X_train = self.scaler.fit_transform(X_train)
                X_val = self.scaler.transform(X_val)

            X_val, y_val = self._remove_anomalies(X_val, y_val)
            self._save_data(X_val, y_val, "arr_val.pickle")

        elif self.scaling:
            X_train = self.scaler.fit_transform(X_train)

        if self.scaling:
            X_test = self.scaler.transform(X_test)

        X_train, y_train = self._remove_anomalies(X_train, y_train)

        self._save_data(X_train, y_train, "arr_train.pickle")
        self._save_data(X_test, y_test, "arr_test.pickle")

    def _remove_anomalies(self, X, y):
        return (X[y == 0, :], y[y == 0])

    def _save_data(self, X, y, path):
        pickle.dump((X, y.reshape(-1, 1)), open(path, "wb"))


if __name__ == "__main__":
    arr = Arrhythmia('arrhythmia.pickle', train_perc=0.6,
                     val_perc=0.2, test_perc=0.2)
