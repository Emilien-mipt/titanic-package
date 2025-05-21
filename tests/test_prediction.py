import numpy as np
import pandas as pd

from titanic_model.predict import make_prediction


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

    # Given custom input
    custom_input = {
        "PassengerId": [0, 1],  # Теперь по 3 значения в каждом списке
        "Pclass": [1, 2],
        "Name": ["Mr. Emin Tagiev", "Miss Sofiya Tagieva"],
        "Sex": ["male", "female"],
        "Age": [32, 5],
        "SibSp": [1, np.nan],
        "Parch": [0, 0],
        "Ticket": [21228, 34567],
        "Fare": [82.2667, 26.0],
        "Cabin": ["B45", "C22"],
        "Embarked": ["S", "C"],
    }

    custom_data = pd.DataFrame(custom_input)

    # When
    result = make_prediction(input_data=custom_data)
    # Then
    preds = result.get("preds")
    probs = result.get("probs")

    assert isinstance(preds, list)
    assert isinstance(probs, list)
    assert isinstance(preds[0], np.int64)
    assert isinstance(probs[0], np.float64)
    assert result.get("errors") is None
    assert len(preds) == len(probs) == 1
