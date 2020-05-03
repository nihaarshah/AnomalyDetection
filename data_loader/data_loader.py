import pickle
from os.path import join

import torch
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset


class AnomDataLoader(DataLoader):
    """
    Data Loader for anomoly data
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        num_workers=1,
        dataset="kdd",
        data_type="train",
        scaling="binary",
    ):

        assert data_type in ["train", "test", "val"], "Data type must be one of train, test, val"

        X, y = pickle.load(
            open(join(data_dir, "{}_{}_{}.pickle".format(dataset, data_type, scaling),), "rb",)
        )

        y = X if data_type in ["train", "val"] else y
        shuffle = shuffle if data_type != "test" else False
        self.dataset = TensorDataset(tensor(X).float(), tensor(y).float())
        super().__init__(
            dataset=self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
