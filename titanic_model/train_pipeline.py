import logging
from pathlib import Path

from config.core import LOG_DIR, config
from pipeline import titanic_pipe
from processing.data_manager import load_dataset, save_pipeline
from processing.validation import get_title
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from titanic_model import __version__ as _version


def run_training() -> None:
    """Train the model."""
    # Update logs
    log_path = Path(f"{LOG_DIR}/log_{_version}.log")
    if Path.exists(log_path):
        log_path.unlink()
    logging.basicConfig(filename=log_path, level=logging.DEBUG)

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    data["Title"] = data["Name"].apply(get_title)

    # cast numerical variables as floats
    data["Fare"] = data["Fare"].astype("float")
    data["Age"] = data["Age"].astype("float")

    data.drop(labels=config.model_config_params.variables_to_drop, axis=1, inplace=True)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config_params.features],  # predictors
        data[config.model_config_params.target],
        test_size=config.model_config_params.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config_params.random_state,
        stratify=data[config.model_config_params.target],
    )

    # fit model
    titanic_pipe.fit(X_train, y_train)

    # make predictions for train set
    class_train = titanic_pipe.predict(X_train)
    pred_train = titanic_pipe.predict_proba(X_train)[:, 1]

    # determine train accuracy and roc-auc
    train_f1 = f1_score(y_train, class_train)
    train_roc_auc = roc_auc_score(y_train, pred_train)

    print(f"train f1-score: {train_f1:.4f}")
    print(f"train roc-auc: {train_roc_auc:.4f}")
    print()

    logging.info(f"train f1-score: {train_f1:.4f}")
    logging.info(f"train roc-auc: {train_roc_auc:.4f}")

    # make predictions for test set
    class_test = titanic_pipe.predict(X_test)
    pred_test = titanic_pipe.predict_proba(X_test)[:, 1]

    # determine test accuracy and roc-auc
    test_f1 = f1_score(y_test, class_test)
    test_roc_auc = roc_auc_score(y_test, pred_test)

    print(f"test f1-score: {test_f1:.4f}")
    print(f"test roc-auc: {test_roc_auc:.4f}")
    print()

    logging.info(f"test f1-score: {test_f1:.4f}")
    logging.info(f"test roc-auc: {test_roc_auc:.4f}")

    # persist trained model
    save_pipeline(pipeline_to_persist=titanic_pipe)


if __name__ == "__main__":
    run_training()
