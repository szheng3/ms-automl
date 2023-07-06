import logging
import pandas as pd
from ludwig.api import LudwigModel
from ludwig.datasets import agnews
import click
import json
from sklearn.model_selection import train_test_split



class AgnewsModel:

    def __init__(self, config, dataset=None, train_df=None, test_df=None, logging_level=logging.INFO):
        self.config = config
        self.model = LudwigModel(config, logging_level=logging_level)
        self.train_df = train_df
        self.test_df = test_df
        self.train_stats = None
        self.test_stats = None
        self.predictions = None
        self.output_directory = None

    def train(self):
        self.train_stats, _, self.output_directory = self.model.train(dataset=self.train_df)

    def evaluate(self):
        self.test_stats, self.predictions, _ = self.model.evaluate(
            self.test_df, collect_predictions=True, collect_overall_stats=True
        )

    def predict(self, text_to_predict):
        self.predictions, _ = self.model.predict(text_to_predict)
        return self.predictions


@click.command()
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    help="data file path",
)
@click.option(
    "--config-path",
    type=click.Path(exists=True),
    help="Path config file",
)
def main(data_path, config_path):
    if data_path is None:
        train_df, test_df, _ = agnews.load(split=True)
    else:
        data_df = pd.read_csv(data_path)
        train_df, test_df = train_test_split(
            data_df, test_size=0.5, random_state=0
        )

    if config_path is None:
        config = {
            "input_features": [
                {
                    "name": "title",
                    "type": "text",
                    "encoder": {
                        "type": "parallel_cnn"
                    }
                }
            ],
            "output_features": [
                {
                    "name": "class",
                    "type": "category",
                }
            ],
            "trainer": {
                "epochs": 3,
            }
        }
    else:
        with open(config_path) as f:
            config = json.load(f)

    model = AgnewsModel(config, train_df=train_df, test_df=test_df)

    model.train()

    model.evaluate()


if __name__ == '__main__':
    main()
