from sklearn.base import BaseEstimator, RegressorMixin, clone
import pandas as pd
import numpy as np


class Estimator(BaseEstimator, RegressorMixin):
    """ A class to combine a trend/seasonality model and residual model

        Attributes:
        ----------
        trend_season_model (object):    estimator for the trend/season model
        residual_model (object):        estimator for the remaining residuals after fitting the trend/season model
        time_columns_start (int):       column to tell the fit/predict process, where the time_features start
                                        in the data set (all time columns should be after another)
        reuse_time_feats (bool):        flag to tell the fit/predict process if time features should be
                                        used a second time in the residual model

        Methods:
        --------
        fit(X: arrays of values, y: array of values):
           clones the estimator settings, so it cant be changed afterwards,
           fits the trend/season model to the target, then fits on the residuals a
           second model, most of the time this will be a tree/boosting method

        predict(X):
            returns the sum of the predicted outcome by the trend/season model and
            the residual model
    """


    def __init__(
        self,
        trend_season_model: object, residual_model: object,
        time_columns_start: int = None,
        reuse_time_feats: bool = False
    ) -> None:
        """ A constructor to forward the estimators to the class """
        self.time_columns_start = time_columns_start
        self.trend_season_model = trend_season_model
        self.residual_model = residual_model
        self.reuse_time_feats = reuse_time_feats

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> object:
        """ A function to fit the data with two estimators one after another

            Args:
            -----
                X (ndarray):   train data / features
                y (array):     target value /target feature


            Returns:
            -------
                self (object): A model fitted on residuals
        """
        #unittest left: amount of columns
        self.trend_season_model_ = clone(self.trend_season_model)
        self.residual_model_ = clone(self.residual_model)

        print("length of window y:", len(y))
        print("length of window X:", len(X))

        if self.time_columns_start is not None:
            X_time = np.asarray(X)[:, self.time_columns_start:]
            self.trend_season_model_.fit(X_time, y)
            y_detrended_deseasoned = y - self.trend_season_model_.predict(X_time)

        else:
            self.trend_season_model_.fit(X, y)
            y_detrended_deseasoned = y - self.trend_season_model_.predict(X)

        if not self.reuse_time_feats:
            X_notime = np.asarray(X)[:, 0:self.time_columns_start]
            self.residual_model_.fit(X_notime, y_detrended_deseasoned)
        else:
            self.residual_model_.fit(X, y_detrended_deseasoned)
        return self

    def predict(self, X: pd.DataFrame) -> object:
        """ Predicts outcomes from the combined model, based on usage of time features in the residual model or not

            Args:
            -----
                X (arrays of values): test data / features


            Returns:
            -------
                self (object): array of predicted values
        """
        if self.time_columns_start is not None:
            X_time = np.asarray(X)[:, self.time_columns_start: ]
            if not self.reuse_time_feats:
                X_notime = np.asarray(X)[:, 0:self.time_columns_start]
                return self.trend_season_model_.predict(X_time) + self.residual_model_.predict(X_notime)
            else:
                return self.trend_season_model_.predict(X_time) + self.residual_model_.predict(X)
        else:
            return self.trend_season_model_.predict(X) + self.residual_model_.predict(X)
