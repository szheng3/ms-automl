[![Tests](https://github.com/szheng3/automl/actions/workflows/python-app.yml/badge.svg)](https://github.com/szheng3/automl/actions/workflows/python-app.yml)
[![Publish](https://github.com/szheng3/automl/actions/workflows/publish.yml/badge.svg)](https://github.com/szheng3/automl/actions/workflows/publish.yml)

# AutoML Project with Ludwig

This project showcases an automated machine learning (AutoML) solution using Ludwig, Uber's open source, deep learning
toolbox. Ludwig provides a unique and user-friendly interface to deep learning, which does not require the user to have
extensive knowledge about the inner workings of deep learning models.

## Features

1. **Easy-to-Use**: Ludwig is designed to be user-friendly and requires no programming experience. All you need to train
   a model and make predictions is a tabular dataset and a few lines of command-line code.
2. **Flexible**: Ludwig enables you to train a deep learning model using various types of input data such as text,
   images, and more. You can also combine different types of input data to create a hybrid model.
3. **Versatile**: Ludwig is suitable for a wide range of tasks such as image classification, text classification, and
   time series forecasting.

## Getting Started

### Prerequisites

- Python 3.7+

### Installation

Clone the repository to your local machine:

Install the dependencies:

```bash
pip install -r requirements.txt
```


### Data Preparation

Ludwig requires the data to be in a tabular format such as CSV. The column names of the CSV file will be used as the
feature names.

## Usage

How to run the code to train a model:

```bash
python train_automl.py --data-path ./data/agnews.csv --config-path ./config/config.json
```


And to generate with a pytorch model:


```bash
make gen
```


And to predict with a pre-trained model:


```bash

python predict_automl_torch.py --predict-path ./data/text_to_predict.csv

```

## Acknowledgements

This project makes use of the fantastic Ludwig toolbox developed by Uber's AI team. We thank them for their work and for
making it open source.

## Contact

If you have any questions or suggestions, please feel free to open an issue on this repository.