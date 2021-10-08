# for encoding categorical variables
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder

# for imputation
from feature_engine.imputation import AddMissingIndicator, CategoricalImputer, MeanMedianImputer
from sklearn.linear_model import LogisticRegression

# pipeline
from sklearn.pipeline import Pipeline

# feature scaling
from sklearn.preprocessing import StandardScaler

from titanic_model.config.core import config
from titanic_model.processing.features import ExtractLetterTransformer

# set up the pipeline
titanic_pipe = Pipeline(
    [
        # ===== IMPUTATION =====
        # impute categorical variables with string missing
        (
            "categorical_imputation",
            CategoricalImputer(imputation_method="missing", variables=config.model_config.categorical_vars_with_na),
        ),
        # add missing indicator to numerical variables
        ("missing_indicator", AddMissingIndicator(variables=config.model_config.numerical_vars_with_na)),
        # impute numerical variables with the median
        (
            "median_imputation",
            MeanMedianImputer(imputation_method="median", variables=config.model_config.numerical_vars_with_na),
        ),
        # Extract letter from cabin
        ("extract_letter", ExtractLetterTransformer(variables=config.model_config.var_for_letter_extraction)),
        # == CATEGORICAL ENCODING ======
        # remove categories present in less than 5% of the observations (0.05)
        # group them in one category called 'Rare'
        (
            "rare_label_encoder",
            RareLabelEncoder(tol=0.05, n_categories=1, variables=config.model_config.categorical_vars),
        ),
        # encode categorical variables using one hot encoding into k-1 variables
        ("categorical_encoder", OneHotEncoder(drop_last=True, variables=config.model_config.categorical_vars)),
        # scale
        ("scaler", StandardScaler()),
        ("Logit", LogisticRegression(C=0.0005, random_state=config.model_config.random_state)),
    ]
)
