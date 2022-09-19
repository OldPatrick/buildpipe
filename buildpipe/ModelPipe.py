import pandas as pd
import math
import optuna
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from .SplitMethods import SlidingWindowSplit
from .MlflowDecorators import conditional_mlflow_start, conditional_mlflow_config, load_mlflow_setup

# sets global mlflow activate variable
try:
    pipe_setup = load_mlflow_setup()
    mlflow_enable = pipe_setup['mlflow_start']['usage']

except FileExistsError:
    print("There exists no config file or the name of the file was wrong")


@dataclass
class Pipe(object):
    """ A class to build a full comparison pipeline of time-series and regression models

        Attributes:
        ----------
        data (pd.DataFrame):    Full data set
        target (str):           target outcome to be predicted
        estimator (object):     MetaEstimator Class
        params (dict):          params of both estimators in MetaEstimator Class
        window_type (str):      if window type is Expanding or Sliding, Sliding can use more parameters
        test_level (str):       Timedelta string, e.g. 24 hours should for example also possible.
        date_column (str):      name of the date column in the full data set
        window_size (int):      length of sliding window train size
        sliding_size (int):     steps taken between training and test windows
        forecast_size(int):     length of sliding window test size

        Methods:
        --------
        fit_pipeline(**pipeline_parameters):
        |   creates a pipeline for the estimators chosen and fits it.
        |
        |---train_test_split(internal):
                splits data into equal sizes of folds based on the timedelta, uneven splits will be handled equally
                in Expanding and Sliding window, e.g. a split of 3002 in 3 splits would result in 1002/1000/1000.
                Sliding and Expanding Windows handle these uneven splits equally for alignment of comparison

        predict_pipeline(self, test_data):
            predicts the outcome of a series/array with the fitted model of the pipeline
            from the fit_pipeline function
    """

    data:               pd.DataFrame = None
    target:             str = None
    estimator:          object = None
    params:             dict = None
    pipe_steps:         list = None
    window_type:        str = "Expanding"
    test_level:         str = None
    date_column:        str = None
    window_size:        int = 1
    sliding_size:       int = 1
    forecast_size:      int = 1
    search_technique:   str = "GridSearchCV"


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

        if search_technique == "OptunaSearchCV":
            return optuna.integration.OptunaSearchCV(pipesteps, params, cv=splitting_technique, **pipeline_parameters)
        else:
            return eval(search_technique)(pipesteps, params, cv=splitting_technique, **pipeline_parameters)


    @conditional_mlflow_start(mlflow_enable)
    @conditional_mlflow_config(mlflow_enable)
    def fit_pipeline(self, **pipeline_parameters):
        """ A function to build a modeling pipeline to compare target outcomes

            Args:
            -----
            **pipeline_parameters:      further pipeline parameters


            Returns:
            --------
            model (object):             completely fitted model with the chosen estimator


            Notes:
            ------
            - pipeline creates the train, test split data as class attributes.
              and returns them for further calculation, if needed
        """

        params = pd.json_normalize(self.params).to_dict(orient="records")[0]
        dict_names = [str.split(key, ".")[1] for key in params.keys()]
        params = dict(zip(dict_names, list(params.values())))

        def train_test_split(column, test_level, y):
            """Create train test split for GridSearchCV"""
            self.data.sort_values(by=[column], ascending=True, inplace=True)
            data_calc_timedelta = self.data.copy()

            splitpoint = (data_calc_timedelta[column] - pd.Timedelta(test_level)).tail(1).values
            X_test = data_calc_timedelta[data_calc_timedelta[column].values >= (splitpoint.max() + 1)]
            X_train = data_calc_timedelta[:-len(X_test)]

            y_train = data_calc_timedelta[:-len(X_test)][y]
            y_test = data_calc_timedelta[-len(X_test):][y]

            X_train.drop([y, column], axis=1, inplace=True)
            X_test.drop([y, column], axis=1, inplace=True)

            return X_train, X_test, y_train, y_test

        steps = [*self.pipe_steps, ("MetaEstimator", self.estimator)] \
            if self.pipe_steps is not None else [("MetaEstimator", self.estimator)]

        pipeline = Pipeline(steps)

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.date_column, self.test_level, self.target)

        model = self.fit_with_search_technique(
            self.search_technique,
            pipeline,
            params,
            **pipeline_parameters
        )

        return(
            model.fit(self.X_train, self.y_train),
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        )


    def predict_pipeline(self, model) -> object:
        """ A function to predict target outcomes with the fitted model/pipeline

            Args:
            -----
            model (object):     a fitted model/pipeline object


            Returns:
            --------
            ndarray:            predicted values of self.y_test
        """
        #needs predict_proba in case of classification
        return model.predict(self.X_test)
