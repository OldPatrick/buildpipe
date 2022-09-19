import re
import pandas as pd
import numpy as np
import math
from typing import Any
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
#from .ModelPipe import Pipe
from .BaselineEstimators import MeanRegressor, LastObsRegressor, RollingRegressor
from .SplitMethods import SlidingWindowSplit
from .MlflowDecorators import conditional_mlflow_start, conditional_mlflow_config, load_mlflow_setup

# sets global mlflow activate variable
try:
    pipe_setup = load_mlflow_setup()
    mlflow_enable = pipe_setup['mlflow_start']['usage']

except FileExistsError:
    print("There exists no config file or the name of the file was wrong")


@dataclass
class Baseline:
    """ A class to build a full comparison pipeline of baseline models

        Attributes:
        ----------
        data (pd.DataFrame):                Original data source
        estimator_base (object, object):    trend_season and residual estimator classes
        base_params (dict):                 params of both estimators
        X_train (pd.DataFrame):             features of train set
        y_train (pd.DataFrame):             target of train set
        X_test (pd.DataFrame):              features of test set
        y_test (pd.DataFrame):              target of test_set
        window_type (str):                  if window type is Expanding or Sliding, Sliding can use more parameters
        window_size (int):                  length of sliding window train size
        sliding_size (int):                 steps taken bei training and test windows
        forecast_size(int):                 length of sliding window test size


        Methods:
        --------
        create_subestimator_baseline(random_grid, time_columns_start, reuse_time_feats, **pipeline_parameters):
            build baselines from the partial/sub estimators of the meta estimator

        create_mean_baseline(**pipeline_parameters):
            build a mean prediction baseline

        create_last_obs_baseline(**pipeline_parameters):
            build a last day is future days prediction baseline

        create_rolling_mean_baseline(window_size, **pipeline_parameters):
            build a rolling mean baseline (no extrapolation)
    """

    data:               pd.DataFrame = None
    estimator_split:    (Any, object) = None
    base_params:        dict = None
    pipe_steps:         list = None
    X_train:            pd.DataFrame = None
    X_test:             pd.DataFrame = None
    y_train:            pd.DataFrame = None
    y_test:             pd.DataFrame = None
    window_type:        str = "Expanding"
    window_size:        int = 1
    sliding_size:       int = 1
    forecast_size:      int = 1
    search_technique:   str = "GridSearchCV"


    def __post_init__(self):
        self.estimator_base = (self.estimator_split.trend_season_model, self.estimator_split.residual_model)


    def fit_with_search_technique(
            self,
            search_technique: str,
            pipesteps: Pipeline,
            params: dict,
            **pipeline_parameters
    ) -> object:
        """ fit the model based on search technique (random. Grid) and Window technique (Expanding, Sliding)

         Args:
         -----
            search_technique (str):         contains a str for eval of randomized search CV or gridsearchCV
            pipesteps (Pipeline):           contains a Pipeline with the name of the estimator and the estimator object
            params (dict):                  contains parameters for the estimator
            **pipeline_parameters:          further pipe parameters like: verbose or scoring


        Returns:
        --------
            model object:                   fitted model with technique Random or Gridsearch
        """
        # GridSearchCV and RandomizedSearchCV use different namings for parameter grids, not using keywords
        # BaseFold Class will automatically throw an error in case of zero splits which is explainable by itself
        # this boilerplate should be reduced in the future because it is now identical to ModelPipe.py
        if self.window_type == "Expanding":
            splits = math.floor(len(self.X_train) / len(self.X_test)) - 1
            splitting_technique = TimeSeriesSplit(n_splits=splits)
        else:
            splits = math.floor(((len(self.data) / len(self.X_test))
                                 - (self.window_size + self.forecast_size)) / self.sliding_size) + 1
            splitting_technique = SlidingWindowSplit(
                n_splits=splits,
                n_folds=splits - 1,
                window_step=self.window_size,
                sliding_step=self.sliding_size,
                forecast_step=self.forecast_size
            )

        return eval(search_technique)(pipesteps, params, cv=splitting_technique, **pipeline_parameters)


    @conditional_mlflow_start(mlflow_enable)
    @conditional_mlflow_config(mlflow_enable)
    def fit_and_predict_partial_estimators(
            self,
            time_columns_start: int = None,
            reuse_time_feats: bool = False,
            **pipeline_parameters
    ) -> (list, list):
        """ A function to create a mean baseline

            Args:
            -----
                random_grid (bool):         flag if you want to use a randomized or gridsearchCV
                time_columns_start (int):   column to tell the fit/predict process, where the time_features start
                                            in the data set (all time columns should be after another)
                reuse_time_feats (bool):    flag to tell the fit/predict process if time features should be
                                            used a second time in the residual model
                **pipeline_parameters:      further pipe parameters like: verbose

            Returns:
            --------
                models (lis[objects]):      fitted models/pipes of trend_season and residual models
                preds (list[np.arrays]):    predicted outcome (values) of trend_season and residual models
        """
        models = []
        preds = []
        res = {}
        index = 0
        length = 0

        length_1st_param_dict = len(self.base_params[list(self.base_params.keys())[0]])

        # X splits for the trend/season and residual model (if res model has no reused time feats)
        X_time_train = self.X_train.copy(deep=True).iloc[:, time_columns_start:]
        X_time_test = self.X_test.copy(deep=True).iloc[:, time_columns_start:]

        X_notime_train = self.X_train.copy(deep=True).iloc[:, :time_columns_start]
        X_notime_test = self.X_test.copy(deep=True).iloc[:, :time_columns_start]

        # rename dict_keys of meta estimator to subestimators
        for outer_key, outer_val in self.base_params.items():
            res[outer_key] = {}
            for inner_key, inner_val in outer_val.items():
                if length == length_1st_param_dict:
                    index += 1
                stringfind = re.split(r'__', f"{inner_key}__")[:-2]
                res[outer_key][
                    inner_key.replace(f"MetaEstimator__{stringfind[1]}",
                                      str(self.estimator_base[index]).strip("()"))] = inner_val
                length += 1

        for _, params in enumerate(res.values()):

            steps_models = [*self.pipe_steps, (str(self.estimator_base[_]).strip("()"), self.estimator_base[_])] \
                if self.pipe_steps is not None else [(str(self.estimator_base[_]).strip("()"), self.estimator_base[_])]

            #steps_models = [(str(self.estimator_base[_]).strip("()"), self.estimator_base[_])]
            pipeline_models_partial = Pipeline(steps_models)

            print("")
            print("******************")
            print("")

            model_base = self.fit_with_search_technique(
                self.search_technique,
                pipeline_models_partial,
                params,
                **pipeline_parameters
            )

            # using fit with different X splits
            if time_columns_start is not None and _ == 0:
                model_base.fit(X_time_train, self.y_train)
                models.append(model_base)
                preds.append(models[_].predict(X_time_test))

            elif time_columns_start is not None and _ == 1:
                if reuse_time_feats:
                    model_base.fit(self.X_train, self.y_train)
                    models.append(model_base)
                    preds.append(models[_].predict(self.X_test))

                elif not reuse_time_feats:
                    model_base.fit(X_notime_train, self.y_train)
                    models.append(model_base)
                    preds.append(models[_].predict(X_notime_test))

        return models, preds


    @conditional_mlflow_start(mlflow_enable)
    @conditional_mlflow_config(mlflow_enable, "Mean_Regressor")
    def fit_an_predict_mean_baseline(self, **pipeline_parameters) -> (object, np.array):
        """ A function to create a mean baseline, fitting the mean of a train fold window as a prediction

            Args:
            -----
                **pipeline_parameters:  further pipe parameters like: verbose

            Returns:
            --------
                tuple of:
                - train_outcome_mean (object, fitted models with folds for the mean technique)
                - predict_outcome_mean (np.array)
        """
        steps_models_mean = [("MeanRegressor", MeanRegressor())]
        pipeline_models_mean = Pipeline(steps_models_mean)
        params = {}

        print("")
        print("******************")
        print("")

        model_mean = self.fit_with_search_technique(
            "GridSearchCV",
            pipeline_models_mean,
            params,
            **pipeline_parameters
        )

        train_outcome_mean = model_mean.fit(self.X_train, self.y_train)
        predict_outcome_mean = model_mean.predict(self.X_test)

        return train_outcome_mean, predict_outcome_mean


    @conditional_mlflow_start(mlflow_enable)
    @conditional_mlflow_config(mlflow_enable, "Last_Obs_Regressor")
    def fit_and_predict_last_obs_baseline(self, **pipeline_parameters) -> (object, np.array):
        """ A function to create a last_obs baseline, fitting the last observation of a train fold as a prediction

            Args:
            -----
                **pipeline_parameters:  further pipe parameters like: verbose

            Returns:
            --------
            tuple of:
                - train_outcome_last (object, fitted models with folds for the last observation technique)
                - predict_outcome_mlast (np.array)
        """
        steps_models_last_obs = [("LastObs_Regressor", LastObsRegressor())]
        pipeline_models_last_obs = Pipeline(steps_models_last_obs)
        params = {}

        print("")
        print("******************")
        print("")

        model_last_obs = self.fit_with_search_technique(
            "GridSearchCV",
            pipeline_models_last_obs,
            params,
            **pipeline_parameters
        )

        train_outcome_last = model_last_obs.fit(self.X_train, self.y_train)
        predict_outcome_last = model_last_obs.predict(self.X_test)

        return train_outcome_last, predict_outcome_last


    @conditional_mlflow_start(mlflow_enable)
    @conditional_mlflow_config(mlflow_enable, "Rolling_Regressor")
    def fit_and_predict_rolling_mean_baseline(self, window_size: list, **pipeline_parameters) -> (object, np.array):
        """ A function to create a rolling mean baseline, fitting a rolling mean window of a train fold to a prediction

            Args:
            -----
            **pipeline_parameters:  further pipe parameters like: verbose


            Returns:
            --------
                tuple of:
                - train_outcome_rolling (object, fitted models with folds for the rolling mean technique)
                - predict_outcome_rolling (np.array)
        """
        steps_models_rolling = [("Rolling_Regressor", RollingRegressor())]
        pipeline_models_rolling = Pipeline(steps_models_rolling)
        params = {
            "Rolling_Regressor__window_size": window_size
        }

        print("")
        print("******************")
        print("")

        model_rolling = self.fit_with_search_technique(
            "GridSearchCV",
            pipeline_models_rolling,
            params,
            **pipeline_parameters
        )

        train_outcome_rolling = model_rolling.fit(self.X_train, self.y_train)
        predict_outcome_rolling = model_rolling.predict(self.X_test)

        return train_outcome_rolling, predict_outcome_rolling
