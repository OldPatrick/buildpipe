import unittest
import numpy as np
import pandas as pd
from buildpipe.SplitMethods import SlidingWindowSplit
from ddt import ddt, data, unpack

@ddt
class TestSlidingWindowSplit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._data = pd.DataFrame({"X": np.arange(0, 1095, 2)})
        cls._splits = 2
        cls._folds = 3
        cls._window_step = 1
        cls._sliding_step = 1
        cls._forecast_step = 1

    def test_split_length(self):
        #3 years split
        res = list(SlidingWindowSplit(
            self._splits,
            self._folds,
            self._window_step,
            self._sliding_step,
            self._forecast_step
        ).split(self._data))

        self.assertEqual(len(res), self._splits,)

    def test_fold_values_equals_one(self):
        n_folds = 0 #gets +1 in init class
        self.assertRaises(ValueError,
                          SlidingWindowSplit,
                          self._splits,
                          n_folds,
                          self._window_step,
                          self._sliding_step,
                          self._forecast_step)

    @data([0, 3, 1, 1, 1], [2, 0, 1, 1, 1], [2, 3, 0, 1, 1], [2, 3, 1, 0, 1], [2, 3, 1, 1, 0])
    @unpack
    def test_zero_value_all_cases(self, n_splits, n_folds, window_step, sliding_step, forecast_step):
        self.assertRaises(ValueError,
                          SlidingWindowSplit,
                          n_splits,
                          n_folds,
                          window_step,
                          sliding_step,
                          forecast_step)
