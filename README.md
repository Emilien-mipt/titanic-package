# Titanic package
## Motivation
Many novice data scientists begin their journey in data science
by building models on the well-known [Titanic dataset](https://www.kaggle.com/c/titanic).
They tend to do that in jupyter notebooks, which is a nice tool for EDA and
building simple models. However, when it comes to pushing the built models to production
this tool becomes inconvenient.

In fact, there are some steps that should be done in order to prepare the model for
production such as organizing the code in modules, writing tests, adding linters
and type checks and e.t.c. However, I noticed that the majority of my students
are not aware of such steps.

Therefore, I created this repository to teach my students on how to switch from jupyter
notebooks to production code and wrap the models into python package, so that it could be used later
in different applications such as web application. As an example in this repo the model is built on the Titanic dataset,
therefore the built package is called "titanic_model".

This repo is heavily influenced by the excellent
course at Udemy
["Deployment of Machine Learning Models"](https://www.udemy.com/course/deployment-of-machine-learning-models/).

## Code structure
### Configs
The model parameters are set via configs. The configs are represented by yaml files. The values
for parameters can be set in `titanic_model/config.yml` file. The cofigs are parsed and validated
in `titanic_model/config/core.py` module using [StrictYaml](https://github.com/crdoconnor/strictyaml) lib for parsing
and [Pydantic](https://pydantic-docs.helpmanual.io/) lib for type checking the values.

### Setting the pipeline and training
The pipeline is set in `titanic_model/pipeline.py` file. Training is set in
`titanic_model/train_pipeline.py` file. All the data processing steps are made in the same
[SciKit-learn](https://scikit-learn.org/stable/) style including custom transformations, stored in
`titanic_model/processing/features.py` file.

### Making predictions
The code for prediction is set in `titanic_model/preict.py` file. Before every prediction
the validation of input data is made. The code for validation can be found in
`titanic_model/processing/validation.py` file.

## How to run the code
The code can be run via the [Tox](https://pypi.org/project/tox/) tool. Tox is a
convenient way to set up the environment and python paths automatically and run the
required commands from the command line. The file with description for tox can be found
in `tox.ini` file. The following commands can be run from the command line
using tox:

* Run training: `tox -e train`
* Run testing (via [pytest](https://docs.pytest.org/en/6.2.x/)): `tox -e test_package`
* Run typechecking (via [mypy](https://mypy.readthedocs.io/en/stable/)): `tox -e typechecks`
* Run style checks
(via [black](https://github.com/psf/black), [isort](https://github.com/PyCQA/isort),
[mypy](https://mypy.readthedocs.io/en/stable/)
and [flake8](https://pypi.org/project/flake8/)): `tox -e stylechecks`



## How to install the package
