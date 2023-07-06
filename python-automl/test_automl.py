import pandas as pd
import pytest
from ludwig.datasets import agnews

from train_automl import AgnewsModel

# Test configuration
test_config = {
    "input_features": [
        {"name": "title", "type": "text", "encoder": {"type": "parallel_cnn"}}
    ],
    "output_features": [{"name": "class", "type": "category"}],
    "trainer": {"epochs": 1},  # use 1 epoch for test speed
}

# Test data to predict
text_to_predict = pd.DataFrame({
    "title": [
        "Google may spur cloud cybersecurity M&A with $5.4B Mandiant buy",
        "Europe struggles to meet mounting needs of Ukraine's fleeing millions",
        "How the pandemic housing market spurred buyer's remorse across America",
    ]
})



def test_model_prediction():
    train_df, test_df, _ = agnews.load(split=True)

    model = AgnewsModel(test_config, train_df=train_df, test_df=test_df)
    model.train()
    predictions = model.predict(text_to_predict)
    assert predictions is not None
    assert len(predictions) == len(text_to_predict)
