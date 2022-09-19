from google.cloud import bigquery
import pandas as pd
import sys


class GCPDataset:
    """ A class to load data from a gcp data set container

            Attributes:
            ----------
            project_id (str):
                id/name of datalake e.g. data-lake-lab-255209


            Methods:
            --------            ‚
            list_dataset_names():
                lists all datasets within a gcp project

            list_table_names(dataset: str):
                lists all tables in a gcp dataset

            load_data(table: str):
                loads table data from gcp dataset container
    """
    __default_data = ""

    def __init__(self, project_id: str) -> None:
        """ A constructor to create the client object with the project_id """

        # start mlflow here right away to start logging-process, mlflfow seems to start at every process ?
        # called with function(*l) l = [values for variables])
        self.project_id = project_id
        self.client = bigquery.Client(project=self.project_id)

    def list_dataset_names(self) -> str:
        """ A function to list all datasets within gcp

            Returns:
            -------
            print statement (str): names of dataset containers within gcp
        """

        datasets = self.client.list_datasets()
        for dataset in datasets:
            print(dataset.dataset_id)

    def list_table_names(self, dataset: str) -> str:
        """ A function to list all tables within a gcp dataset

            Returns:
            -------
            print statement (str): table names within the dataset of a project id
        """

        table_ref = ".".join((self.client.project, dataset))

        try:
            tables = self.client.list_tables(table_ref)

        except IndexError:
            print("Did you check if there are tables in the dataset?")

        except NameError:
            print("Table doesn't exist, check your spelling")

        except AttributeError:
            print("Can't list children tables without knowing parent dataset")

        except:
            print("Unexpected error:", sys.exc_info()[0])

        else:
            for table in tables:
                print(table.table_id)

    def load_data(self, dataset: str, table: str) -> pd.DataFrame:
        """ A function to load the dataset from a gcp dataset into a pandas df with bigquery

            Args:
            -----
                table (str): name of chosen table, in a created dataset container in gcp


            Returns:
            --------
                bigquery_to_df (pandas DataFrame): dataframe of bigquery job


            Notes:
            ------
                - future versions should incorporate a flexible query, so that not all of the data has to be loaded, not only a
                  default load all
                - function atm not more served as the working code will be developed with test datasets (boston, sklearn), (airline, sktime)
        """

        self.dataset = dataset
        dataset_ref = self.client.get_dataset(self.dataset)

        self.table = dataset_ref.table(table)
        full_ref = self.client.get_table(self.table).full_table_id.replace(":", ".")
        job_config = bigquery.QueryJobConfig()

        sqlquery = f"""
                       SELECT *
                       FROM {full_ref}
                   """

        query_job = self.client.query(query=sqlquery, job_config=job_config)
        bigquery_to_df = query_job.to_dataframe()
        GCPDataset.default_data = bigquery_to_df.copy()
        __default_data = bigquery_to_df

        bigquery_to_df.columns = [label.lower() for label in bigquery_to_df.columns]

        return bigquery_to_df