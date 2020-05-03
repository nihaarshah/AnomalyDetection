import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


class BaseData:

    anomaly_classes = []
    multi_categorical_vars = []
    dataset_name = ""
    data_path = ""

    def __init__(self, train_perc=0.6, test_perc=0.2, val_perc=0.2, scaling="binary"):

        self.data_dict = pickle.load(open(self.data_path, "rb"))
        self._get_data()
        self.non_multi_cat_vars = [
            i for i in range(self.X.shape[1]) if i not in self.multi_categorical_vars
        ]

        self.encoders = [
            OneHotEncoder(sparse=False, drop="first")
            for _ in range(len(self.multi_categorical_vars))
        ]
        self.scaler = MinMaxScaler() if scaling == "binary" else StandardScaler()
        self.scaling = scaling
        self.train_perc = train_perc
        self.test_perc = test_perc
        self.val_perc = val_perc
        if len(self.multi_categorical_vars) > 0:
            self._replace_categorical_one_hot()
        if len(self.anomaly_classes) > 0:
            self._encode_anomalies()
        self._split_train_val_test()

    def _get_data(self):
        self.X = self.data_dict["X"]
        self.y = self.data_dict["y"].reshape(-1)

    def _encode_categorical(self):

        categorical_data = []
        for i in range(len(self.multi_categorical_vars)):
            var = self.multi_categorical_vars[i]
            one_hot = self.encoders[i].fit_transform(self.X[:, var].reshape(-1, 1))
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
            self.X, self.y, test_size=self.test_perc, random_state=42
        )

        if self.val_perc > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=self.val_perc / self.train_perc, random_state=42
            )

        X_train = self.scaler.fit_transform(X_train)
        if self.val_perc > 0:
            X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        X_train, y_train = self._remove_anomalies(X_train, y_train)
        if self.val_perc > 0:
            X_val, y_val = self._remove_anomalies(X_val, y_val)

        self._save_data(
            X_train, y_train, "{}_train_{}.pickle".format(self.dataset_name, self.scaling)
        )
        if self.val_perc > 0:
            self._save_data(
                X_val, y_val, "{}_val_{}.pickle".format(self.dataset_name, self.scaling)
            )
        self._save_data(X_test, y_test, "{}_test_{}.pickle".format(self.dataset_name, self.scaling))

    def _remove_anomalies(self, X, y):
        return (X[y == 0, :], y[y == 0])

    def _save_data(self, X, y, path):
        pickle.dump((X, y.reshape(-1, 1)), open(path, "wb"))


class KDD(BaseData):
    anomaly_classes = ["normal."]
    multi_categorical_vars = [1, 2, 3]
    dataset_name = "kdd"
    data_path = "kdd99.pickle"

    def __init__(self, *args, **kwargs):
        super(KDD, self).__init__(*args, **kwargs)

    def _get_data(self):
        self.X = self.data_dict["data"]
        self.y = self.data_dict["target"].astype(str)


class Arrhythmia(BaseData):
    data_path = "arrhythmia.pickle"
    dataset_name = "arr"

    def __init__(self, *args, **kwargs):
        super(Arrhythmia, self).__init__(*args, **kwargs)


class Thyroid(BaseData):
    data_path = "thyroid.pickle"
    dataset_name = "thyroid"

    def __init__(self, *args, **kwargs):
        super(Thyroid, self).__init__(*args, **kwargs)


class Musk(BaseData):
    data_path = "musk.pickle"
    dataset_name = "musk"

    def __init__(self, *args, **kwargs):
        super(Musk, self).__init__(*args, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="kdd, arr, or thyroid", type=str)
    parser.add_argument("--train", help="train percent", type=float, default=0.8)
    parser.add_argument("--test", help="test percent", type=float, default=0.2)
    parser.add_argument(
        "--scaling", help="Data scaling type (normal, binary)", type=str, default="binary"
    )
    args = parser.parse_args()
    if args.dataset == "kdd":
        data_class = KDD
    elif args.dataset == "arr":
        data_class = Arrhythmia
    elif args.dataset == "thyroid":
        data_class = Thyroid
    elif args.dataset == "musk":
        data_class = Musk

    data_class(
        train_perc=args.train,
        val_perc=1 - args.train - args.test,
        test_perc=args.test,
        scaling=args.scaling,
    )
