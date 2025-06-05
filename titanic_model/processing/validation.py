import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from titanic_model.config.core import config


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
        for var in config.model_config_params.features
        if var
        not in config.model_config_params.categorical_vars_with_na + config.model_config_params.numerical_vars_with_na
        and validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""
    input_data["Title"] = input_data["Name"].apply(get_title)
    # cast numerical variables as floats
    input_data["Fare"] = input_data["Fare"].astype("float")
    input_data["Age"] = input_data["Age"].astype("float")

    input_data.drop(labels=config.model_config_params.variables_to_drop, axis=1, inplace=True)

    # Columns should coinside with config.model_config_params.feature
    assert input_data.columns.tolist() == config.model_config_params.features

    relevant_data = input_data[config.model_config_params.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)

    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleTitanicInputs(inputs=validated_data.replace({np.nan: None}).to_dict(orient="records"))
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class TitanicInputSchema(BaseModel):
    PassengerId: Optional[int] = None
    Pclass: Optional[int]
    Name: Optional[str] = None
    Sex: Optional[str]
    Age: Optional[float]
    SibSp: Optional[int]
    Parch: Optional[int]
    Ticket: Optional[str] = None
    Fare: Optional[float]
    Cabin: Optional[str]
    Embarked: Optional[str]


class MultipleTitanicInputs(BaseModel):
    inputs: List[TitanicInputSchema]
