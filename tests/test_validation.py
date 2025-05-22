import pandas as pd

from titanic_model.processing.validation import get_first_cabin, get_title


def test_get_first_cabin(sample_input_data):
    assert sample_input_data["Cabin"].iat[12] == "B45"
    assert sample_input_data["Cabin"].iat[14] == "E31"
    assert pd.isna(sample_input_data["Cabin"].iat[0])

    sample_input_data["Cabin"] = sample_input_data["Cabin"].apply(get_first_cabin)

    assert sample_input_data["Cabin"].iat[12] == "B"
    assert sample_input_data["Cabin"].iat[14] == "E"
    assert pd.isna(sample_input_data["Cabin"].iat[0])


def test_get_title(sample_input_data):
    sample_input_data["Title"] = sample_input_data["Name"].apply(get_title)
    assert sample_input_data["Title"].iat[12] == "Mrs"
    assert sample_input_data["Title"].iat[13] == "Mr"
    assert sample_input_data["Title"].iat[36] == "Miss"
    assert sample_input_data["Title"].iat[55] == "Master"
    assert sample_input_data["Title"].iat[88] == "Other"
