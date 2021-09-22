from config.core import config
from pipeline import titanic_pipe
from processing.data_manager import load_dataset, save_pipeline
from processing.validation import get_first_cabin, get_title
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    data["Cabin"] = data["Cabin"].apply(get_first_cabin)
    data["Title"] = data["Name"].apply(get_title)

    # cast numerical variables as floats
    data["Fare"] = data["Fare"].astype("float")
    data["Age"] = data["Age"].astype("float")

    data.drop(labels=config.model_config.variables_to_drop, axis=1, inplace=True)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )

    # fit model
    titanic_pipe.fit(X_train, y_train)

    # make predictions for train set
    class_ = titanic_pipe.predict(X_train)
    pred = titanic_pipe.predict_proba(X_train)[:, 1]

    # determine mse and rmse
    print("train roc-auc: {}".format(roc_auc_score(y_train, pred)))
    print("train accuracy: {}".format(accuracy_score(y_train, class_)))
    print()

    # make predictions for test set
    class_ = titanic_pipe.predict(X_test)
    pred = titanic_pipe.predict_proba(X_test)[:, 1]

    # determine mse and rmse
    print("test roc-auc: {}".format(roc_auc_score(y_test, pred)))
    print("test accuracy: {}".format(accuracy_score(y_test, class_)))
    print()

    # persist trained model
    save_pipeline(pipeline_to_persist=titanic_pipe)


if __name__ == "__main__":
    run_training()
