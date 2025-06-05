from titanic_model.processing.validation import get_title


def test_get_title(sample_input_data):
    sample_input_data["Title"] = sample_input_data["Name"].apply(get_title)
    assert sample_input_data["Title"].iat[12] == "Mrs"
    assert sample_input_data["Title"].iat[13] == "Mr"
    assert sample_input_data["Title"].iat[36] == "Miss"
    assert sample_input_data["Title"].iat[55] == "Master"
    assert sample_input_data["Title"].iat[88] == "Other"
