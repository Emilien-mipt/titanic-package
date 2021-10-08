import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from titanic_model.config.core import config


# retain only the first cabin if more than
# 1 are available per passenger
def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return np.nan


def get_title(passenger):
    line = passenger
    if re.search("Mrs", line):
        return "Mrs"
    elif re.search("Mr", line):
        return "Mr"
    elif re.search("Miss", line):
        return "Miss"
    elif re.search("Master", line):
        return "Master"
    else:
        return "Other"


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    new_vars_with_na = [
        var
        for var in config.model_config.features
        if var not in config.model_config.categorical_vars_with_na + config.model_config.numerical_vars_with_na
        and validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""
    input_data["Cabin"] = input_data["Cabin"].apply(get_first_cabin)
    input_data["Title"] = input_data["Name"].apply(get_title)
    # cast numerical variables as floats
    input_data["Fare"] = input_data["Fare"].astype("float")
    input_data["Age"] = input_data["Age"].astype("float")

    input_data.drop(labels=config.model_config.variables_to_drop, axis=1, inplace=True)

    # Columns should coinside with config.model_config.feature
    assert input_data.columns.tolist() == config.model_config.features

    relevant_data = input_data[config.model_config.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)

    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleTitanicInputs(inputs=validated_data.replace({np.nan: None}).to_dict(orient="records"))
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class TitanicInputSchema(BaseModel):
    Pclass: Optional[int]
    Sex: Optional[str]
    Age: Optional[float]
    SibSp: Optional[int]
    Parch: Optional[int]
    Fare: Optional[float]
    Cabin: Optional[str]
    Embarked: Optional[str]
    Title: Optional[str]


class MultipleTitanicInputs(BaseModel):
    inputs: List[TitanicInputSchema]
