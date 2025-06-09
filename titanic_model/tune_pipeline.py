from pathlib import Path

from cross_validation import CrossValidator
from processing.data_manager import load_dataset
from processing.validation import get_title

from titanic_model import __version__ as _version
from titanic_model.config.core import LOG_DIR, config
from titanic_model.pipeline import titanic_pipe
from titanic_model.processing.data_manager import save_pipeline

if __name__ == "__main__":
    # Загрузка данных
    data = load_dataset(file_name=config.app_config.training_data_file)

    data["Title"] = data["Name"].apply(get_title)

    # cast numerical variables as floats
    data["Fare"] = data["Fare"].astype("float")
    data["Age"] = data["Age"].astype("float")

    data.drop(labels=config.model_config_params.variables_to_drop, axis=1, inplace=True)

    X = data[config.model_config_params.features]
    Y = data[config.model_config_params.target]

    # Инициализация валидатора
    validator = CrossValidator(
        pipeline=titanic_pipe,
        param_grid=config.tune_config.random_forest_params,
        strategy=config.tune_config.strategy,
        random_state=config.model_config_params.random_state,
    )

    # Запуск кросс-валидации
    validator.fit(X, Y)

    # Получение результатов
    results = validator.get_best_results()
    print(f"Лучшие параметры: {results['best_params']}")
    print(f"Лучшая оценка: {results['best_score']:.4f}")

    # Логирование
    log_path = Path(f"logs/cv_results_{_version}.log")
    validator.log_results(log_path)

    # Сохранение лучшей модели
    print("Saved best model!")
    save_pipeline(pipeline_to_persist=validator.best_estimator_)
