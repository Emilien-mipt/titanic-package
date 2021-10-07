import numpy as np

from titanic_package.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_shape = (418, 11)

    # When
    result = make_prediction(input_data=sample_input_data)
    # Then
    preds = result.get("preds")
    probs = result.get("probs")

    assert isinstance(preds, list)
    assert isinstance(probs, list)
    assert isinstance(preds[0], np.int64)
    assert isinstance(probs[0], np.float64)
    assert result.get("errors") is None
    assert len(preds) == len(probs) == expected_shape[0]
