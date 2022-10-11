import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

# class needs a lot of refactoring;
# but I have ignored this class for a long time.


class ShowResults:
    """A class to show base, local and global results and process data for deeper insights

    Attributes:
    ----------
    model (object):
        fitted model object of the pipeline from the Pipe class
    X_train (arrays of values):
        self explicable
    X_test (arrays of values):
        self explicable
    y_train (array of values):
        self explicable
    y_test (array of values):
        self explicable
    time_columns_start (int):
        column to tell the fit/predict process, where the time_features start
        in the data set (all time columns should be after another)
    model_is_trend_season (str):
        Flag if the model is a trend/season model or not, only relevant for single trend/season model
        of the Baseline class (attribute will be probably in the future a bool)
    time_feats_in_res (str):
        flag to tell the fit/predict process if time features should be
        used a second time in the residual model (could be probably in the future a bool too, and renamed)
    model_not_combined (bool):
        flag to tell the ShowResults class if the data has to be splitted, because of a
        X_time/X_notime split in the MetaEstimator class


    Methods:
    --------
    show_model_results(y_pred: array of values, **optional params for permutation)
        - shows best hyperparams, R2, MAPE, MdAPE, MSE, MASE,
        - shows permutation importances
        - shows ranks of hyperparams combinations

    <Not Implemented>
    show local_importance()
        Shows the local importance
    < /Not Implemented>

    <Not Implemented>
    show global importance()
        Shows global importance
    < /Not Implemented>

    Notes:
    ------
        class is not yet completed.
    """

    def __init__(
        self,
        model: object,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        add_to_experiment: bool = False,
        time_columns_start: int = None,
        model_is_trend_season: str = "no",
        time_feats_in_res: str = "no",
        model_not_combined: bool = False,
    ) -> None:

        """Constructor to forward the fitted model, data, splits and subestimators"""
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.add_to_experiment = add_to_experiment
        self.time_columns_start = time_columns_start
        self.model_is_trend_season = model_is_trend_season
        self.time_feats_in_res = time_feats_in_res
        self.model_not_combined = model_not_combined

    def generate_plots(self):
        pass

    def show_model_results(self, y_pred, **perm_parameters):
        """A function to show comparable outputs for all models
        Args:
        -----
            y_pred (arrays of values): predicted values of a predict function
            **perm parameters: possible vars for the permutation imp. function


        Returns:
        --------
            print (str): Outputs of the model fit, and model predict
            permutation importance plot (object): plot for train data
            permutation importance plot (object): plot for test data


        Notes:
        ------
            Link to MASE: Performance Measurement for models for time series
            that can reach 0 as the true value:
            https://robjhyndman.com/papers/mase.pdf
            MASE: Mean Absolute Scaled Error, MASE is always 1 for naive time series forecast
            THE MASE has to be beaten and for that smaller than 1, yet calc is broken as it seems
            Thus, I would ignore it, as I had no chance to look into it yet.
        """
        X_train_cut = self.X_train.copy(deep=True)
        X_test_cut = self.X_test.copy(deep=True)

        if self.model_is_trend_season == "yes" and self.model_not_combined:
            X_train_cut = X_train_cut.iloc[:, self.time_columns_start :]  # noqa
            X_test_cut = X_test_cut.iloc[:, self.time_columns_start :]  # noqa

        if self.time_feats_in_res == "no" and self.model_not_combined:
            X_train_cut = X_train_cut.iloc[:, : self.time_columns_start]  # noqa
            X_test_cut = X_test_cut.iloc[:, : self.time_columns_start]  # noqa

        try:
            print(
                f"Best trained parameters from the pipeline (fit):",  # noqa
                self.model.best_params_,
            )
            print(
                f"Best trained mean score (R2, RMSE, ..) from the n-folds of the pipeline (fit):",  # noqa
                self.model.best_score_,
            )
            # "best_score_ = Mean cross-validated score of the best_estimator"
            try:
                rank_params = pd.DataFrame(self.model.cv_results_)  # noqa:
                display(rank_params.sort_values("rank_test_score", ascending=True))
            except AttributeError:
                print(
                    "No cv results function for this SearchMethod available, will continue without"
                )
            finally:
                print("preparing permutation importance plots ...")
                perm_train = permutation_importance(
                    self.model,
                    X_train_cut,
                    self.y_train,
                    n_repeats=30,
                    random_state=42,
                    **perm_parameters,
                )

                sorted_idx_train = perm_train.importances_mean.argsort()

                fig, ax = plt.subplots()
                ax.boxplot(
                    perm_train.importances[sorted_idx_train].T,
                    vert=False,
                    labels=X_train_cut.columns[sorted_idx_train],
                )
                ax.set_title("Permutation Importances (train set)")
                plt.show()

                perm_test = permutation_importance(
                    self.model, X_test_cut, self.y_test, n_repeats=30, random_state=42
                )

                sorted_idx_test = perm_test.importances_mean.argsort()

                fig2, ax2 = plt.subplots()

                ax2.boxplot(
                    perm_test.importances[sorted_idx_test].T,
                    vert=False,
                    labels=X_test_cut.columns[sorted_idx_test],
                )
                ax2.set_title("Permutation Importances (test set)")
                plt.show()

        except BrokenPipeError:
            raise BrokenPipeError(
                "No Pipeline fitted with baselines, continuing with remaining Fit values"
            )

        finally:
            absolute_errors = np.abs(self.y_test - y_pred)
            scaled_error = np.mean(
                absolute_errors / mean_absolute_error(self.y_test, y_pred)
            )

            print(
                f"Best predicted R2 score (predict):",  # noqa
                r2_score(self.y_test, y_pred),
            )
            print(
                f"Best predicted MAPE score (predict):",  # noqa
                mean_absolute_percentage_error(self.y_test, y_pred),
            )
            print(
                f"Best predicted MdAPE score (predict):",  # noqa
                median_absolute_error(self.y_test, y_pred),
            )
            print(
                f"Best predicted MSE score (predict):",  # noqa
                mean_squared_error(self.y_test, y_pred),
            )

            if type(scaled_error) is pd.Series:
                print("No MASE available for NaN Series")
            else:
                print(f"Best predicted MASE score (predict):", scaled_error)  # noqa
