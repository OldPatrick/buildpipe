import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from itertools import cycle
from sklearn.preprocessing import LabelEncoder
from sklego.preprocessing import RepeatingBasisFunction


class Processing:
    """ A class to preprocess the raw data before fitting the baseline and the residuals

            Attributes:
            ----------
            None

            Methods:
            --------
            create_gaussian_seasonality(data: pd.DataFrame, date_column: str, cycles: int, target: str):
            |   creates gaussian distributed seasonal dummies depending on min/max dates of a data frame
            |
            |---is_in_datenames(internal):
            |       forces users to give their date columns reasonable names in the long term
            |
            |---reindex_data(internal):
            |       aggregates the data, if necessary, so time gaps can be filled by reindexing the data
            |
            |---add_seasonality(internal):
                    merges the reindexed data so that every day gets its real seasonal share, after that
                    deletes filled gaps again

            <Not implemented>
            show_gaussian_seasonality():
                shows the created seasonality bell curves
            < /Not implemented>

            preprocess_cat_feats(data: pd.DataFrame, *cat_feats: list):
                uses LabelEncoder to encode labels with categorical numbers

            list_cat_feats(data: pd.DataFrame, *cat_feats: list):
                lists all label encoded columns with their values and encodings
           """

    def __init__(self):
        pass

    def create_gaussian_seasonality(self, data: pd.DataFrame, date_column: str, cycles: int) -> pd.DataFrame:
        """ A function to create a gaussian distributed season features

            Args:
            -----
            data (pd.DataFrame):        data frame to add the season features
            date_column (date n64):     date column of the dataset
            cycle (int):                cycle number, 12 indicates monthly, 52 weekly


            Returns:
            --------
            df_seasons (pandas DataFrame): dataframe with seasonal dummies


            Notes:
            ------
            Should start on the first of a month
        """

        set_datenames = {"sendetag"}

        def is_in_datenames(column: str, datenames: set) -> bool:
            """check if date column for seasonality is known"""
            """notes: Purpose is a pool of all datetime column names that are meaningful"""

            if column in datenames:
                print(f"Found {column}")
                return True

            else:
                raise NameError(f"""Date column not known, rename to the following:{datenames}""")

        def reindex_data(data: pd.DataFrame, column: str) -> pd.DataFrame:
            """groups data to have a non-duplicated time index, whose gaps will then be filled"""
            data_reindexed = data.copy()
            data_reindexed["dummy"] = 0
            data_reindexed = data_reindexed.groupby(column, as_index=True).agg({"dummy": pd.Series.mean})
            data_reindexed = data_reindexed.asfreq('1H').reset_index()
            return data_reindexed

        def add_seasonality(
            data: pd.DataFrame,
            data_idx: pd.DataFrame,
            season: pd.DataFrame,
            column: str
        ) -> pd.DataFrame:
            """merges seasonal dummies with data and deletes useless columns after the merge, including added gaps """

            data_add = data.merge(data_idx,
                                  how='outer',
                                  left_on=data[column].astype('datetime64'),
                                  right_on=data_idx[column])

            data_add.drop([f"{column}_x", f"{column}_y", "dummy"], axis=1, inplace=True)
            data_add.rename(columns={"key_0": column}, inplace=True)

            if len(season) > len(data_idx):
                starting_hour = len(season) - len(data_idx)
                season = season.iloc[starting_hour:]

            season[column] = sorted(data_add[column].unique())
            data_added = data_add.merge(season, left_on=column, right_on=column)

            return data_added

        if is_in_datenames(date_column, set_datenames):

            min_date = data[date_column].astype("datetime64").dt.strftime("%Y-%m-%d").min()
            max_date = data[date_column].astype("datetime64").dt.strftime("%Y-%m-%d").max()
            time_diff = (pd.to_datetime(max_date) - pd.to_datetime(min_date)).total_seconds()

            #can be easily added, example is shown below, fit only on hour only
            time_diff_days = (((time_diff / 3600) / 24) + 1)
            time_diff_hours = time_diff_days * 24

            season_data = pd.DataFrame()

            #days = cycle(np.arange(364))
            hours = cycle(np.arange(24))

            #season_data['day'] = [next(days) for day in range(int(time_diff_days))]
            season_data['hour'] = [next(hours) for hour in range(int(time_diff_hours))]


            try:
                rbf = RepeatingBasisFunction(n_periods=cycles,
                                             remainder='passthrough',
                                             column='hour',
                                             input_range=(0, 24))
            except ValueError:
                print("Values for rbf might be not correct, please check ?RepeatingBasisFunction")

            else:
                rbf.fit(season_data)
                season_share_df = pd.DataFrame(rbf.transform(season_data))
                subdata_reindexed = reindex_data(data, date_column)
                data_added_season = add_seasonality(data,
                                                    subdata_reindexed,
                                                    season_share_df,
                                                    date_column)

                data = data_added_season[:len(data)]

                return data

    def create_trend(self, data: pd.DataFrame):
        """ A function to create all sorts of trends
            Args:
            -----
                data (pd.DataFrame):    original data set
                patsy formula           to be determined


                Returns:
                --------
                data (pandas DataFrame): dataframe with extrapolatable trend
        """
        data["trend"] = np.arange(0, len(data))
        return data

    def show_gaussian_seasonality(self, data):
        """ function not integrated yet """
        #fig, axes = plt.subplots(nrows=season_data_t.shape[1], figsize=(17, 12))
        #for i in range(season_data_t.shape[1]):
        #axes[i].plot(season_data['day_hour'], season_data_t[:, i])
        #plt.show()
        pass

    def preprocess_cat_feats(self, data: pd.DataFrame, *cat_feats: list) -> pd.DataFrame:
        """ A function to convert labels into categorical int values

            Args:
            -----
                data (pd.DataFrame): original data set
                cat_feats (list):        column names of columns to be converted to int


            Returns:
            --------
                data (pandas DataFrame): dataframe with or without transformed label cols
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data is no object of type data frame")

        if not cat_feats:
            raise ValueError("Empty List of categorical features")

        else:
            value = [False for feature in cat_feats if feature not in data.columns]
            if all(value):
                self.listencoders = []
                labelencoder = LabelEncoder()
                for col in cat_feats:
                    data[col] = labelencoder.fit_transform(data[col]).astype('int')
                    self.listencoders.append(labelencoder.classes_)
                return data

            else:
                raise NameError(f"""Categorical Columns not match data set columns""")

    def list_cat_feats(self, data: pd.DataFrame, *cat_feats: list):
        #-> DisplayHandle(tuple) return type?
        """ A function to return the labels and the coded int categories from the LabelEncoder

            Args:
            -----
                data (pd.DataFrame): original data set
                cat_feats (list):        column names of columns to be converted to int


            Returns:
            --------
                print statement (data frame): all label decoded columns from the input data frame
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data is no object of type data frame")

        if not cat_feats:
            raise ValueError("Empty List of categorical features")

        else:
            return display(pd.DataFrame(self.listencoders))




