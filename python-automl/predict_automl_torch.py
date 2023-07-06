import torch
import pandas as pd
from ludwig.models.inference import InferenceModule
import json
import click

from ludwig.utils.inference_utils import to_inference_module_input_from_dataframe

@click.command()
@click.option(
    "--predict-path",
    type=click.Path(exists=True),
    help="data file path",
    required=True,
)
def main(predict_path):
    text_to_predict=pd.read_csv(predict_path)
    inference_module = InferenceModule.from_directory('./model/')
    # output_df = inference_module.predict(text_to_predict)
    scripted_module = torch.jit.script(inference_module)

    with open(f"./model/model_hyperparameters.json") as f:
        config = json.load(f)

    input_sample_dict = to_inference_module_input_from_dataframe(text_to_predict, config)
    # output_df = scripted_module.predict(text_to_predict)
    output_df = scripted_module(input_sample_dict)

    print(output_df)





if __name__ == '__main__':
    main()
