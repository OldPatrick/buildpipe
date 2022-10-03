Changelog of buildpipe package
==============================

## [ 0.5 ] -  Date 2022/10/03
### Changes
###### Added
- Added .pre-commit-config.yaml
- Added .isort.cfg

###### Updated
- Updated requirements.txt:
   - black==22.8.0
   - click==8.1.3
   - coverage==6.4.4
   - flake8==5.0.4
   - isort[requirements_deprecated_finder]
   - mypy==0.981
   - pre-commit==2.20
   - pytest==7.1.3
   - sphinx==1.*

- Rewrote data source name

## [ 0.4 ] -  Date 2021/10/08
### Changes
###### Added
- Added OptunaSearchCV for the MetaEstimator Pipeline
- Added an example section for OptunaCV in the README

###### Updated
- Rewrote parts of the README.md
- Rewrote structure of CHANGELOG.md
- Rewrote walkthrough.ipynb with an Optuna Example and another timedelta example
- Reduced redundant code in ModelPipe.py and ModelBaseline.py
- Rewrote function names in ModelBaseline.py to be more precise and meaningful
- Deleted check for splits != 0 since scikit-learn will automatically check for this and throws an descriptive error


### Refactored Files
###### Updated
- ModelPipe.py
- ModelBaseline.py
- ModelShowResults.py
- setup.py
- requirements.txt
- README.md
- walkthrough.ipynb
<br><br>
<hr>

## [ 0.3.b ] -  Date 2021/10/04
### Changes
###### Fixed
- TimeSeriesSplit with Sliding window now makes the correct amount of splits for the train data/cv


### Refactored Files
###### Updated
- ModelPipe.py
- ModelBaseline.py
- SplitMethods.py
- setup.py
<br><br>
<hr>

## [ 0.3.a ] -  Date 2021/10/01
### Changes
###### Fixed
- TimeSeriesSplit with Expanding window now makes the correct amount of splits for the train data/cv


### Refactored Files
###### Updated
- ModelPipe.py
- ModelBaseline.py
- setup.py
<br><br>
<hr>

## [ 0.3 ] -  Date 2021/09/14
### Changes
###### Updated
- Added ExampleTransformers to the buildpipe package, yet these were not tested yet, and could be unstable
- Added the Exception Class
- Added support for extra preprocessing steps in the pipeline like PCA and SVC
- Reduced boilerplate code for some classes with dataclasses decorator


### Refactored Files
###### Added
- Exceptions.py
- PreTransformers.py

###### Updated
- ModelPipe.py
- ModelBaseline.py
- __init__.py
- setup.py
- walkthrough.ipynb
- README
<br><br>
<hr>


## [ 0.2.a ] -  Date 2021/07/26
### Changes
###### Updated
- README (added sections for troubleshooting with mlflow and the pipe setup config, added a bug section)
- Rewrote release dates in ISO-8601: https://www.iso.org/iso-8601-date-and-time-format.html

###### Issues and Summary
- buildpipe now works also on GCP Jupyter notebooks.

- If there is no .env file found locally in the project folder of your project (which may be the case when using jupyter on GCP), the package now catches the FIleNotFound Error und encourgaes you to use jupyter magic to set the env variables

- If there is no pipe_setup.json in our root folder of your notebook (GCP) the buildpipe package will created a default json file at the first launch, and then
looks always for this root directory, instead for a project directory in pycharm

- There is a warning when using the gaussian seasonality with GCP while having mlflow turned on, it not affects your project,
yet it is annoying and confusing, and will be probably removed in the future

- Known Issues:
    - When using gaussian seasonality it throws an auth warning for MlFLow in GCP, altough there is no connection
made in the respective class
    - The mlflow decorator set's up a connection and closes it after each method. If you fit several models in no time (e.g. a few simple baselines) that results in several on/off connection statements for mlflow. that result in
a "Too many connections/max entries exceeded" error of mlflow (Error 100 class). This only occurs if your machine is extremely fast fitting the baselines (mostyl in GCP were you lend hardware power). The temporary solution to the latter is to wait a little bit before fitting another model (e.g. 30 seconds up to 1 minute)


### Refactored Files
###### Added
- default_dict_mlflow.py

###### Updated
- ModelPipe.py
- ModelBaseline.py
- MlflowDecorators.py
- __init__.py
<br><br>
<hr>


## [ 0.2 ] -  Date 2021/07/01
### Changes
###### Added
- CHANGELOG.md
###### Updated
- requirements.txt

### Refactored Files
###### Added
- MlflowDecorators.py

###### Deleted
- LoadGCPParquet.py

###### Updated
- ModelPipe.py
- ModelBaseline.py
- ModelShowResults.py
<br><br>
<hr>

## [ 0.1.a ] -  Date of last fix 2021/06/23
Initial Commit
### Added 1 level
- buildpipe folder
- test folder (not part of the package, only for documentation)
- README.md
- setup.py
- requirements.txt
- walkthrough_pipeline.ipynb
- .gitignore
- Metro_Traffic.csv (as dummy data set)

### Added 2 level
#### buildpipe folder with files
- BaselineEstimators.py
- LoadGCPDataset.py
- LoadGCPParquet.py
- MetaEstimator.py
- ModelBaseline.py
- ModelPipe.py
- ModelShowResults.py
- PreProcessing.py
- SplitMethods.py
- __init__.py
