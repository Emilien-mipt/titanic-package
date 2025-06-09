import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline


class CrossValidationResults(TypedDict):
    best_params: Dict[str, Any]
    best_score: float
    cv_results: pd.DataFrame


class CrossValidator:
    """Класс для проведения кросс-валидации и поиска гиперпараметров."""

    def __init__(
        self,
        pipeline: Pipeline,
        param_grid: Dict[str, List[Any]],
        strategy: str = "grid",
        cv: int = 5,
        scoring: str = "f1",
        n_iter: int = 20,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        verbose: int = 1,
    ):
        self.pipeline = pipeline
        self.param_grid = param_grid
        self.strategy = strategy
        self.cv = cv
        self.scoring = scoring
        self.n_iter = n_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.search_results_: Optional[Union[GridSearchCV, RandomizedSearchCV]] = None
        self.best_estimator_: Optional[Pipeline] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if self.strategy == "grid":
            self.search_results_ = GridSearchCV(
                estimator=self.pipeline,
                param_grid=self.param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )
        elif self.strategy == "random":
            self.search_results_ = RandomizedSearchCV(
                estimator=self.pipeline,
                param_distributions=self.param_grid,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self.search_results_.fit(X, y)
        self.best_estimator_ = self.search_results_.best_estimator_

    def get_best_results(self) -> CrossValidationResults:
        """Возвращает типизированный словарь с результатами."""
        if self.search_results_ is None:
            raise RuntimeError("Call fit() first")

        return CrossValidationResults(
            best_params=self.search_results_.best_params_,
            best_score=self.search_results_.best_score_,
            cv_results=pd.DataFrame(self.search_results_.cv_results_),
        )

    def log_results(self, log_path: Path) -> None:
        """Логирует результаты с гарантированной типизацией."""
        if self.search_results_ is None:
            raise RuntimeError("Call fit() first")

        if log_path.exists():
            log_path.unlink()

        logging.basicConfig(filename=log_path, level=logging.DEBUG)
        results = self.get_best_results()

        logging.info(f"\nBest params: {results['best_params']}")
        logging.info(f"\nBest {self.scoring} score: {results['best_score']:.4f}")

        # Теперь mypy точно знает, что это DataFrame
        cv_df = results["cv_results"]
        score_columns = ["mean_test_score", "std_test_score"] + [f"split{i}_test_score" for i in range(self.cv)]

        logging.info("\nДетальная информация по фолдам:")
        logging.info(cv_df.loc[self.search_results_.best_index_, score_columns])
