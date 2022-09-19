import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from dataclasses import dataclass
from typing import Optional, Any, Callable


#The ExampleTransformer is for explaining purposes only. It not does very much
@dataclass
class ExampleTransformer(BaseEstimator, TransformerMixin):
    """ (:class:) ExampleTransformer which learns nothing but transforms a column of your choice with np.sqrt

        Attributes:
        ----------
        feature_name:   name of a column from a data set


        Methods:
        --------
        fit(X, y):
            Learns something from the training set and uses it in the transform section on the training set

        transform(X, y)
            - during training uses the insight of fit to adjust the training data,
            - during predict uses the learned aspects of the train data without adjusting
    """
    feature_name:   str


    def fit(self, X, y=None):
        # if nothing is learned during training, then the method just returns self, and saves no insight.
        # Typehint return should be omitted as it is possible that before a column transformer may be another step that
        # transforms X into an ndarray, thus you can not have then a return type dataframe in a FunctionalTransformer
        return self


    def transform(self, X, y=None):
        # transform method will be called twice by the pipe, for training and predicting
        X_ = X.copy()
        X_[self.feature_name] = np.sqrt(X_[self.feature_name])
        return X_


