import unittest

import numpy as np
import pandas as pd
from ddt import data, ddt, unpack

from buildpipe.SplitMethods import SlidingWindowSplit


@ddt
class TestSlidingWindowSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._data = pd.DataFrame({"X": np.arange(0, 1999, 2)})
        cls._splits = 5  # splits are atm folds renaming is advised
        cls._folds = 2
        cls._window_step = 1
        cls._sliding_step = 1
        cls._forecast_step = 1

    def test_fold_length(self):
        # folds are splits + 1
        folds = list(
            SlidingWindowSplit(
                self._splits,
                self._folds,
                self._window_step,
                self._sliding_step,
                self._forecast_step,
            ).split(self._data)
        )

        self.assertEqual(
            len(folds) + 1,
            self._splits,
        )

    def test_fold_values_equals_one(self):
        n_folds = 1
        self.assertRaises(
            ValueError,
            SlidingWindowSplit,
            self._splits,
            n_folds,
            self._window_step,
            self._sliding_step,
            self._forecast_step,
        )

    @data(
        [0, 3, 1, 1, 1],
        [2, 0, 1, 1, 1],
        [2, 3, 0, 1, 1],
        [2, 3, 1, 0, 1],
        [2, 3, 1, 1, 0],
    )
    @unpack
    def test_zero_value_all_cases(
        self, n_splits, n_folds, window_step, sliding_step, forecast_step
    ):
        self.assertRaises(
            ValueError,
            SlidingWindowSplit,
            n_splits,
            n_folds,
            window_step,
            sliding_step,
            forecast_step,
        )
