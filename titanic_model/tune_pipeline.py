import logging
from pathlib import Path

import pandas as pd
from processing.data_manager import load_dataset
from processing.validation import get_title
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate

from titanic_model import __version__ as _version
from titanic_model.config.core import LOG_DIR, config
from titanic_model.pipeline import titanic_pipe
from titanic_model.processing.data_manager import save_pipeline


def run_grid_search(X, y, pipeline, grid):
    """Запуск GridSearchCV"""
    search = GridSearchCV(estimator=pipeline, param_grid=grid, cv=5, scoring="f1", n_jobs=-1, verbose=1)
    search.fit(X, y)
    return search


def run_random_search(X, y, pipeline, grid, n_iter=20):
    """Запуск RandomizedSearchCV"""
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=grid,
        n_iter=n_iter,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        random_state=config.model_config_params.random_state,
    )
    search.fit(X, y)
    return search


if __name__ == "__main__":
    # Update logs
    log_path = Path(f"{LOG_DIR}/log_{_version}.log")
    if Path.exists(log_path):
        log_path.unlink()
    logging.basicConfig(filename=log_path, level=logging.DEBUG)

    # Загрузка данных
    data = load_dataset(file_name=config.app_config.training_data_file)

    data["Title"] = data["Name"].apply(get_title)

    # cast numerical variables as floats
    data["Fare"] = data["Fare"].astype("float")
    data["Age"] = data["Age"].astype("float")

    data.drop(labels=config.model_config_params.variables_to_drop, axis=1, inplace=True)

    X = data[config.model_config_params.features]
    Y = data[config.model_config_params.target]

    # Параметры для GridSearchCV/RandomizedSearchCV
    param_grid = config.tune_config.random_forest_params

    # Выбор метода поиска (раскомментируйте нужный)
    if config.tune_config.strategy == "grid":
        print("<UNK> Starting GridSearchCV...")
        search = run_grid_search(X, Y, titanic_pipe, param_grid)
    elif config.tune_config.strategy == "random":
        print("<UNK> Starting RandomizedSearchCV...")
        search = run_random_search(X, Y, titanic_pipe, param_grid, n_iter=20)
    else:
        raise ValueError(f"Unknown strategy {config.tune_config.strategy}")

    # 1. Вывод результатов
    print(f"Best params: {search.best_params_}")
    print(f"Best f1 score: {search.best_score_:.4f}")

    # 2. Результаты всех фолдов для лучшей комбинации
    best_idx = search.best_index_
    cv_results = pd.DataFrame(search.cv_results_)

    print("\nДетальные результаты для лучшей комбинации:")
    print(
        cv_results.loc[
            best_idx,
            [
                "mean_test_score",
                "std_test_score",
                "split0_test_score",
                "split1_test_score",
                "split2_test_score",
                "split3_test_score",
                "split4_test_score",
            ],
        ]
    )

    logging.info(f"Best params: {search.best_params_}")
    logging.info(f"Best f1 score: {search.best_score_:.4f}")
    logging.info("\nДетальные результаты для лучшей комбинации:")
    logging.info(
        cv_results.loc[
            best_idx,
            [
                "mean_test_score",
                "std_test_score",
                "split0_test_score",
                "split1_test_score",
                "split2_test_score",
                "split3_test_score",
                "split4_test_score",
            ],
        ]
    )

    # Сохранение лучшей модели
    best_model = search.best_estimator_

    # persist trained model
    print(f"Saved best model!")
    save_pipeline(pipeline_to_persist=best_model)
