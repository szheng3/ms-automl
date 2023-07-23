import torch
import pandas as pd
from ludwig.models.inference import InferenceModule
import json
import click

from ludwig.utils.inference_utils import to_inference_module_input_from_dataframe

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
    # train_df, test_df, _ = agnews.load(split=True)
    #
    # model = AgnewsModel(test_config, train_df=train_df, test_df=test_df)
    # model.train()
    # predictions = model.predict(text_to_predict)
    # assert predictions is not None
    # assert len(predictions) == len(text_to_predict)
    text_to_predict=pd.read_csv("./data/text_to_predict.csv")

    inference_module = InferenceModule.from_directory('./model/')
    # output_df = inference_module.predict(text_to_predict)
    scripted_module = torch.jit.script(inference_module)

    with open(f"./model/model_hyperparameters.json") as f:
        config = json.load(f)

    input_sample_dict = to_inference_module_input_from_dataframe(text_to_predict, config)
    # output_df = scripted_module.predict(text_to_predict)
    output_df = scripted_module(input_sample_dict)
    assert len(text_to_predict) == len(output_df["class"]["predictions"])
