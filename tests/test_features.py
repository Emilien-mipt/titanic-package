from titanic_model.config.core import config
from titanic_model.processing.features import ExtractLetterTransformer


def test_temporal_variable_transformer(sample_input_data):
    # Given
    transformer = ExtractLetterTransformer(variables=config.model_config_params.var_for_letter_extraction)

    assert sample_input_data["Cabin"].iat[12] == "B45"
    assert sample_input_data["Cabin"].iat[14] == "E31"

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    # test 1
    test_subject_1 = subject["Cabin"].iat[12]
    assert isinstance(test_subject_1, str)
    assert test_subject_1 == "B"
    assert len(test_subject_1) == 1

    # test 2
    test_subject_2 = subject["Cabin"].iat[14]
    assert isinstance(test_subject_2, str)
    assert test_subject_2 == "E"
    assert len(test_subject_2) == 1
