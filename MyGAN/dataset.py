"""
Main dataset class
"""

from typing import Optional, Iterable, TypeVar, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
import tensorflow as tf

from .nns import noise_layer

REPR_NROWS = 5

T = TypeVar('T', bound='Dataset')
class Dataset:
    """
    Class representing a dataset.
    """

    def __init__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 W: Optional[np.ndarray] = None,
                 x_labels: Optional[Iterable[str]] = None,
                 y_labels: Optional[Iterable[str]] = None) -> None:
        """
        Constructor.
        """

        assert len(X.shape) == 2 and len(Y.shape) == 2, "Only 1D data supported"

        assert X.shape[0] == Y.shape[0], \
                "X and Y shapes must agree along 0 axis, got {} vs {}".format(
                    X.shape, Y.shape
                )

        if W is not None:
            assert len(W.shape) == 1, "Only scalar weights supported, got shape {}".format(
                W.shape
            )
            assert W.shape[0] == X.shape[0], \
                "W, X and Y must agree along 0 axis, got {}, {}, {}".format(
                    W.shape, X.shape, Y.shape
                )

        if x_labels is not None:
            if not isinstance(x_labels, list): x_labels = list(x_labels)
            assert len(x_labels) == X.shape[1]
        if y_labels is not None:
            if not isinstance(y_labels, list): y_labels = list(y_labels)
            assert len(y_labels) == Y.shape[1]

        self.data = np.concatenate([X, Y] + ([] if W is None else [W.reshape((-1, 1))]), axis=1)

        self.X  = self.data[:,:X.shape[1]]
        self.Y  = self.data[:,X.shape[1]:X.shape[1]+Y.shape[1]]
        self.XY = self.data[:,:X.shape[1]+Y.shape[1]]
        self.W  = self.data[:,-1] if W is not None else None

        self.x_labels = x_labels
        self.y_labels = y_labels
        self.nx = X.shape[1]
        self.ny = Y.shape[1]

    def __repr__(self) -> str:
        """
        String representation.
        """

        ldots = ['...'] if len(self.data) > REPR_NROWS else []
        return "\n".join([
            "Dataset ({} lines)".format(len(self.data)),
            "x columns : {}".format(self.x_labels),
            self.X[:REPR_NROWS].__repr__()
        ] + ldots + [
            "y columns : {}".format(self.y_labels),
            self.Y[:REPR_NROWS].__repr__()
        ] + ldots + [
            "weights",
            self.W[:REPR_NROWS].__repr__() if self.W is not None else None.__repr__()
        ] + (ldots if self.W is not None else [])
        )

    def __len__(self) -> int:
        return len(self.data)

    def check_similar(self: T, ds: T) -> bool:
        """
        Check if two datasets have same columns (i.e. they can be concatenated).
        """

        if self.X.shape[1] != ds.X.shape[1]: return False
        if self.Y.shape[1] != ds.Y.shape[1]: return False
        if self.data.shape[1] != ds.data.shape[1]: return False
        
        if self.x_labels is not None and ds.x_labels is not None:
            if self.x_labels != ds.x_labels: return False
        
        if self.y_labels is not None and ds.y_labels is not None:
            if self.y_labels != ds.y_labels: return False

        return True

    def split(self, test_size: Optional[float] = 0.5, random_state: Optional[int] = 42) -> Tuple['Dataset', 'Dataset']:
        """
        Split the dataset into two Dataset objects with train and test parts.
        """
        if self.W is None:
            X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y,
                                                                test_size=test_size, random_state=random_state)
            W_train, W_test = None, None
        else:
            X_train, X_test, Y_train, Y_test, W_train, W_test = train_test_split(self.X, self.Y, self.W,
                                                                                 test_size=test_size, random_state=random_state)
        
        ds_train = Dataset(X_train, Y_train, W_train, self.x_labels, self.y_labels)
        ds_test  = Dataset(X_test , Y_test , W_test , self.x_labels, self.y_labels)

        return ds_train, ds_test

    def fit_transformer(self, transformer: TransformerMixin) -> None:
        """
        Fit a transformer on this dataset.
        """
        transformer.fit(self.XY)
    
    def transform(self, transformer: TransformerMixin) -> None:
        """
        Transform this dataset.
        """
        self.XY[:] = transformer.transform(self.XY)

    def inverse_transform(self, transformer: TransformerMixin) -> None:
        """
        Inverse transform this dataset.
        """
        self.XY[:] = transformer.inverse_transform(self.XY)
    
    def get_tf(
            self,
            batch_size: int,
            cols: Iterable[str] = ['all'],
            seed: Optional[int] = None,
            make_tf_ds: bool = True,
            noise_std: Optional[float] = None
        ) -> List[tf.Tensor]:
        """
        Get tensorflow dataset
        """

        if make_tf_ds:
            shuffler = tf.contrib.data.shuffle_and_repeat(self.data.shape[0], seed=seed)
            shuffled_ds = shuffler(tf.data.Dataset.from_tensor_slices(self.data))
            full_tensor = shuffled_ds.batch(batch_size).make_one_shot_iterator().get_next()
        else:
            assert seed is None, "seed is not used when make_tf_ds=False"
            full_tensor = tf.placeholder_with_default(self.data[:batch_size], [None, self.data.shape[1]])

        if 'W' in ''.join(cols):
            assert self.W is not None

        if noise_std is not None:
            if self.W is None:
                full_tensor = noise_layer(full_tensor, noise_std, tf.constant("train"))
            else:
                full_tensor = tf.concat(
                                    [
                                        noise_layer(full_tensor[:,:-1], noise_std, tf.constant("train")),
                                        full_tensor[:,-1]
                                    ],
                                    axis=1
                                )

        slice_map = {'all' : slice(None),
                     'X'   : slice(self.X.shape[1]),
                     'Y'   : slice(self.X.shape[1], self.X.shape[1] + self.Y.shape[1]),
                     'W'   : -1,
                     'XY'  : slice(self.X.shape[1] + self.Y.shape[1]),
                     'YW'  : slice(self.X.shape[1], None)}
        
        return [full_tensor[:,slice_map[c]] for c in cols]

        


    @staticmethod
    def concatenate(datasets: Iterable['Dataset']) -> 'Dataset':
        """
        Concatenate datasets.
        It is required that all datasets should satisfy the check_similar condition.
        """

        if not isinstance(datasets, list):
            datasets = list(datasets)

        assert all(datasets[0].check_similar(ds) for ds in datasets[1:])

        try:
            x_labels = next(ds.x_labels for ds in datasets if ds.x_labels is not None)
        except(StopIteration):
            x_labels = None

        try:
            y_labels = next(ds.y_labels for ds in datasets if ds.y_labels is not None)
        except(StopIteration):
            y_labels = None

        X = np.concatenate([ds.X for ds in datasets], axis=0)
        Y = np.concatenate([ds.Y for ds in datasets], axis=0)
        W = np.concatenate([ds.W for ds in datasets], axis=0) if datasets[0].W is not None else None

        return Dataset(X, Y, W, x_labels, y_labels)
    
    @staticmethod
    def from_csv(filename: str,
                 x_labels: Iterable[str],
                 y_labels: Iterable[str],
                 w_label: Optional[str] = None,
                 delimiter: str = ',') -> 'Dataset':
        """
        Construct a Dataset object from a CSV file.
        """

        if not isinstance(x_labels, list):
            x_labels = list(x_labels)
        if not isinstance(y_labels, list):
            y_labels = list(y_labels)

        columns = x_labels + y_labels + ([w_label] if w_label else [])
        df = pd.read_csv(filename, delimiter=delimiter, usecols=columns)

        return Dataset(X=df[x_labels].values,
                       Y=df[y_labels].values,
                       W=(df[w_label].values if w_label else None),
                       x_labels=x_labels,
                       y_labels=y_labels)