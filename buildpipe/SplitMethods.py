import math

import numpy as np
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


class SlidingWindowSplit:
    """A class to provide a SlidingWindowSplitter for the sklearn Pipeline

    Attributes:
    ----------
    n_splits (int):         coding is wrecked up, splits equal folds atm
    n_folds (int):          not used anymore !!!
    window_step (int):      size of training window
    sliding_step (int):     size of sliding step for the next training window
                            to begin
    forecast_step (int):    size of test window

    Methods:
    --------
    split(X, y, groups)
        creates the splits and give them back one slice
        after another (yield return)


    get_n_splits(X, y, groups):
        sklearn pipe needs the exact amount of splits, before they
        are calculated in the split funtion. Thus, this function
        catches it, over the initialization of a pipe object like
        GridSearchCV. If return of get_n_splits not matches splits
        in function split, This throws an error.
    """

    def __init__(
        self,
        n_splits: int = None,
        n_folds: int = None,
        window_step: int = 1,
        sliding_step: int = 1,
        forecast_step: int = 1,
    ):
        self.n_splits = n_splits
        self.n_folds = n_folds
        self.window_step = window_step
        self.sliding_step = sliding_step
        self.forecast_step = forecast_step

        if 0 in {
            self.window_step,
            self.forecast_step,
            self.sliding_step,
            self.n_splits,
            self.n_folds,
        }:
            raise ValueError(
                f"No zero values allowed for: "
                f"train window {self.window_step}, "
                f"test window {self.forecast_step}, "
                f"sliding window {self.sliding_step}, "
                f"split length {self.n_splits}."
            )

        if self.n_folds == 1:
            raise ValueError(f"No fold length of only 1 allowed {self.n_folds}.")

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Args:
        -----
        X (array-like of shape (n_samples, n_features)):    Training data, n_samples is the number of samples
                                                            and n_features is the number of features.
        y: (array-like of shape (n_samples, )):             Always ignored, exists for compatibility.
        groups (array-like of shape (n_samples, )):         Always ignored, exists for compatibility.

        Yields:
        -------
        train (ndarray):                                    The training set indices for that split.
        test (ndarray):                                     The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)

        # Make sure we have enough samples for the given split parameters
        if self.n_splits > n_samples:
            raise ValueError(
                (
                    f"Cannot have number of splits={self.n_splits} greater"
                    f" than the number of samples={n_samples}."
                )
            )

        indices = np.arange(n_samples)

        # in case of uneven rows, 1st window gets the uneven fold length
        if n_samples % self.n_splits != 0:
            even_len = math.floor(len(X) / self.n_splits)
            full_even_len = even_len * self.n_splits
            init_window_len = (len(X) - full_even_len) + even_len
            windows_starts = range(init_window_len, n_samples, even_len)
        else:
            init_window_len = math.floor(len(X) / self.n_splits)
            windows_starts = range(init_window_len, n_samples, init_window_len)

        first_window = True
        splits_total = 0

        for window in windows_starts:
            # tracking of splits needed because the window is created on folds not splits, enumerate?
            if splits_total == self.n_splits:
                break

            flexible_test_size = (windows_starts.step * self.window_step) + (
                windows_starts.step * self.forecast_step
            )

            if first_window:
                first_window = False
                flexible_train_size = windows_starts.step * self.window_step + (
                    init_window_len - windows_starts.step
                )
                # varying train sizes for 1st window/split
                # if window size would be uneven (equals the sum of n sliding windows rows
                # to the sum of the n expanding windows rows
                splits_total += 1
                print("First Split")
                yield (
                    indices[:flexible_train_size],
                    indices[
                        flexible_train_size : flexible_test_size  # noqa
                        + (init_window_len - windows_starts.step)
                    ],
                )
            else:
                print("Next Split")
                flexible_train_size = windows_starts.step * self.window_step
                splits_total += 1
                yield (
                    indices[
                        ((window - windows_starts.step) * self.sliding_step) : (  # noqa
                            flexible_train_size
                            + ((window - windows_starts.step) * self.sliding_step)
                        )
                    ],
                    indices[
                        (
                            flexible_train_size
                            + ((window - windows_starts.step) * self.sliding_step)
                        ) : flexible_test_size  # noqa
                        + ((window - windows_starts.step) * self.sliding_step)
                    ],
                )

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Args:
        -----
        X (object):         Always ignored, exists for compatibility.
        y (object):         Always ignored, exists for compatibility.
        groups (object):    Always ignored, exists for compatibility.

        Returns:
        --------
        n_splits (int):     Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits - 1
