from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
import numpy as np


class MeanRegressor(BaseEstimator, RegressorMixin):
    """ A class to create a mean predictor for y_test

        Attributes:
        ----------
        None


        Methods:
        --------
        fit(X: arrays of values, y: array of values):
           clones the estimator settings, so it cant be changed afterwards
           fits, the trend/season model and then fits on the remains the
           residual model

        predict(X):
            returns the sum of the predicted outcome by the trend/season model and
            the residual model
    """
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> object:
        """ A function to fit the mean of a y_train window

            Args:
            -----
                X (arrays of values):   train data / features
                y (array of values):    target value /target feature


            Returns:
            -------
                self (object): A model fitted on residuals
        """
        self.mean_ = y.mean()
        return self

    def predict(self, X: pd.DataFrame) -> object:
        """ A function to insert the mean of a y_train_window to y_test as a prediction

            Args:
            -----
                X (arrays of values): test data / features


            Returns:
            -------
                self (object): array of predicted values
        """
        return np.array(X.shape[0]*[self.mean_])


class LastObsRegressor(BaseEstimator, RegressorMixin):
    """ A class to create a lastobs predictor for y_test

        Attributes:
        ----------
        None


        Methods:
        --------
        fit(X: arrays of values, y: array of values):
            see above

        predict(X):
            see above
    """
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> object:
        """ A function to fit the last_obs of the y_train window

            Args:
            -----
                see above


            Returns:
            -------
                see above
        """
        self.ylastvalue_ = y[-1:, ].max()
        return self

    def predict(self, X: pd.DataFrame) -> object:
        """ A function to insert the last observation of y_train to y_test as a prediction

            Args:
            -----
                X (arrays of values): test data / features


            Returns:
            -------
                self (object): array of predicted values
        """
        return np.array(X.shape[0]*[self.ylastvalue_])


class RollingRegressor(BaseEstimator, RegressorMixin):
    """ A class to create a lastobs predictor for y_test

        Attributes:
        ----------
        None


        Methods:
        --------
        fit(X: arrays of values, y: array of values):
            see above

        predict(X):
            see above
    """
    def __init__(self, window_size=1):
        self.window_size = window_size

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> object:
        """ A function to fit the last_obs of the y_train window

            Args:
            -----
                see above


            Returns:
            -------
                see above
        """

        self.rolling_ = y.rolling(window=self.window_size, min_periods=1).mean()

        return self

    def predict(self, X: pd.DataFrame) -> object:
        """ A function to insert the last observation of y_train to y_test as a prediction

            Args:
            -----
                X (arrays of values): test data / features


            Returns:
            -------
                self (object): array of predicted values
        """
        return np.array(self.rolling_.tail(X.shape[0]))