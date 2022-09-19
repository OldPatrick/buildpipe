import logging
import mlflow
import json
import os
import mlflow.exceptions
import mlflow.sklearn
import mlflow.tracking
import datetime as dt
from os import path
from py_dotenv import read_dotenv
from .default_dict_mlflow import new_dict

global pipe_setup

# Loading credentials for mlflow from .env file
try:
    dotenv_path = os.path.join(os.path.dirname(__file__), '../.env')
    read_dotenv(dotenv_path)
except FileNotFoundError:
    print("No File found, package may be not in local pycharm directory, "
          "Please use: %env MLFLOW_TRACKING_USERNAME=MY_VALUE, %env MLFLOW_TRACKING_PASSWORD=MY_VALUE, if you"
          "are using a jupyter notebook, that is in the cloud or over Anaconda.")



# *****Please read before reviewing *****
# non decorator functions will be refactored into a Mlflowcheck class, they do not belong here anymore,
# also we then should have a unified uri connection, as this is a lot of redundancy atm.

def check_exp_name():
    """Check if experiment name already exists, yet it does not work with dead experiments
    funtion should be combined with the list_dead_mlflow_exps function in the future"""
    try:
        exp_name = pipe_setup["mlflow_start"]["experiment_name"]

        mlflow.set_tracking_uri(
            pipe_setup["mlflow_start"]["uri"]
        )
        exp_names = {exp.name for exp in mlflow.tracking.MlflowClient().list_experiments()}
        if exp_name in exp_names:
            print(f"Experiment name {exp_name} found.")
    except NameError:
        print("If No pipe_setup file found and method is part of main(), it will be skipped. "
              "Pipe setup file will be loaded with the pipe object. Create a pipe object first.")


def delete_all_runs_in_exp():
    """deletes all runs of the exp, named in the config file, before running again"""
    mlflow.set_tracking_uri(
        pipe_setup["mlflow_start"]["uri"]
    )
    try:
        exp_id = mlflow.get_experiment_by_name(pipe_setup["mlflow_start"]["experiment_name"]).experiment_id
        for run_obj in mlflow.list_run_infos(exp_id):
            mlflow.delete_run(run_obj.run_id)
    except AttributeError:
        print("No runs in Experiment to delete.")


def list_dead_mlflow_exps(last_id):
    """Lists ALL experiments regardless of active or inactive/deleted"""
    try:
        print("If Connection to mlflow is not up, the function looks locally")
        for _ in range(0, last_id + 1):
            try:
                print(mlflow.get_experiment(str(_)))
            except mlflow.exceptions.MlflowException:
                break
    except:
        print("Can not look for experiments. Check your local mlruns folder or if your Mlflow connection is up")
    finally:
        print("No more experiments in bucket list for this user")


def load_mlflow_setup():
    """Loads the setup configuration for mlflow"""
    global pipe_setup # is this redundancy needed ? pycharm highlights if not double mentioned
    try:
        with open(path.join(path.dirname(__file__), "../pipe_setup.json")) as file:
            pipe_setup = json.load(file)
            return pipe_setup
    #ugly
    except FileNotFoundError:
        current_dir = os.getcwd()

        if not os.path.exists(f'{current_dir}/pipe_setup.json'):

            with open(f"{current_dir}/pipe_setup.json", 'w', encoding='utf-8') as file:
                json.dump(new_dict, file)
                print("Creating json config gile, please wait until visible in directory.")
            with open(path.join(path.dirname(__file__), f"{current_dir}/pipe_setup.json")) as file:
                pipe_setup = json.load(file)
                return pipe_setup
        else:
            with open(path.join(path.dirname(__file__), f"{current_dir}/pipe_setup.json")) as file:
                pipe_setup = json.load(file)
                return pipe_setup


def conditional_mlflow_start(activate):
    """consideration decorator/decorator factory, if activate, then use mlflow, otherwise return function as is"""

    def _dec_not_started_mlflow(method):
        """No Connection to mlflow"""
        def wrapper(*args, **kwargs):
            return method(*args, **kwargs)
        return wrapper

    # mlflow usage
    def _dec_start_mlflow(method):
        """Start Mlflow Connection"""
        def wrapper(*args, **kwargs):

            _logger = logging.Logger("MlFlow")
            load_mlflow_setup()

            # Try Connection
            try:
                mlflow.set_tracking_uri(
                    pipe_setup["mlflow_start"]["uri"]
                )

            except mlflow.exceptions.MlflowException:
                _logger.warning(
                    "MLflow failed to connect to the tracking server. Continuing without!"
            )

            # Check if experiment was previously created or was deleted from dashboard (no deep delete)
            try:
                if mlflow.get_experiment_by_name(pipe_setup["mlflow_start"]["experiment_name"]).name and \
                        mlflow.get_experiment_by_name(
                            pipe_setup["mlflow_start"]["experiment_name"]).lifecycle_stage == "deleted":
                    raise NameError("Experiments may not have the same name as previously deleted experiments. "
                                    "There is currently no deep delete available. Please choose a new experiment name.")

                elif mlflow.get_experiment_by_name(pipe_setup["mlflow_start"]["experiment_name"]).name and \
                        mlflow.get_experiment_by_name(
                                pipe_setup["mlflow_start"]["experiment_name"]).lifecycle_stage == "active":
                    mlflow.set_experiment(pipe_setup["mlflow_start"]["experiment_name"])

            except AttributeError:
                if not pipe_setup["mlflow_start"]["experiment_name"]:
                    # Set Default Experiment Name if forgotten in config file and set it to active (otherwise exp_id =0)

                    default_name = f"DefaultPipeExperiment_{dt.datetime.now()}"

                    mlflow.create_experiment(name=default_name,
                                             artifact_location=f"{pipe_setup['mlflow_start']['artifact_storage']}"
                                                               f"{default_name}")

                    mlflow.set_experiment(default_name)
                    print("No experiment name in setup file, setting default name DefaultPipeExperiment with timestamp")

                else:
                    # Set Experiment Name from config file and set it to active (otherwise exp_id =0)

                    mlflow.create_experiment(name=f"{pipe_setup['mlflow_start']['experiment_name']}",
                                             artifact_location=f"{pipe_setup['mlflow_start']['artifact_storage']}"
                                                               f"{pipe_setup['mlflow_start']['experiment_name']}")

                    mlflow.set_experiment(experiment_name=f"{pipe_setup['mlflow_start']['experiment_name']}")
                    print("Connection to Ml Flow Dashboard worked")

            mlflow.sklearn.autolog()

            return method(*args, **kwargs)
        return wrapper
    return _dec_start_mlflow if activate else _dec_not_started_mlflow


def conditional_mlflow_config(activate, *decoargs):
    """consideration decorator/decorator factory, if activate, then reconfig mlflow with tags and artifacts
       Tracking of mlflow Experiments can be done without this decorator, but then we have no information about the runs.
    """

    def _dec_no_mlflow_config(method):
        """Do NOT config mlflow after run"""
        def wrapper(*args, **kwargs):
            return method(*args, **kwargs)
        return wrapper

    def _dec_mlflow_config(method):
        """DO Config mlflow after run"""
        def wrapper(*args, **kwargs):

            return_value = method(*args, **kwargs)

            def set_auto_tags_parent(parent_run, decoargs, *args):
                """Sets fixed and auto tags for mlflow"""
                # one condition will vanish after refactoring the baseline class
                if decoargs:
                    subestimator = ''.join(decoargs)
                else:
                    if args[0].params:
                        first_entry = list(args[0].params.values())[0]
                        subestimator = list(first_entry)[0].split("__")[0]
                    else:
                        subestimator = str(args[0].estimator_base[1]).strip("()")

                # setting fixed tags
                for k, v in pipe_setup['mlflow_run']["mlflow_tags"]["fixed_parent_run"].items():
                    client.set_tag(parent_run, k, v)

                # setting auto tags
                mlflow_auto_tags = [
                    ("features", list(args[0].X_train.columns)[: 14]),
                    ("subestimator", subestimator),
                    ("window_class", args[0].window_type),
                    ("window_train", args[0].forecast_size),
                    ("window_test", args[0].forecast_size),
                    ("window_sliding", args[0].sliding_size)
                ]

                for tag, entry in mlflow_auto_tags:
                    try:
                        client.set_tag(parent_run, tag, entry)
                    except TypeError:
                        print("One run was None. Delete corrupted run id with None")

            def log_artifacts_parent(parent_run, decoargs, *args):
                """ log artifacts after the finished run

                    Notes:
                    ------
                        function is not yet complete
                """

                os.makedirs("data", exist_ok=True)
                with open("data/dummy.txt", "w+") as txt:
                    txt.writelines("Hello Artifact")

                client.log_artifacts(parent_run,
                                     local_dir="data")

            # get parent run
            last_parent_run = set()
            exp_id = mlflow.get_experiment_by_name(pipe_setup["mlflow_start"]["experiment_name"]).experiment_id

            for item in mlflow.list_run_infos(exp_id):
                if item.__getattribute__("end_time") is None:
                    print("Unfinished run in experiment or actual run, please delete corrupted runs."
                          "And check tagging of your results")
                    continue
                last_parent_run.add(
                    (item.__getattribute__("end_time"),
                     item.__getattribute__("run_id"))
                )

            parent_run = max(last_parent_run)[1]

            client = mlflow.tracking.MlflowClient()
            set_auto_tags_parent(parent_run, decoargs, *args)
            log_artifacts_parent(parent_run, decoargs, *args)

            return return_value
        return wrapper
    return _dec_mlflow_config if activate else _dec_no_mlflow_config


