[![Tests](https://github.com/szheng3/ms-automl/actions/workflows/python-app.yml/badge.svg)](https://github.com/szheng3/ms-automl/actions/workflows/python-app.yml)
[![Publish](https://github.com/szheng3/ms-automl/actions/workflows/publish.yml/badge.svg)](https://github.com/szheng3/ms-automl/actions/workflows/publish.yml)


# Microservice-AutoML
> ### _Shuai Zheng, Yilun Wu | Summer '23 | Duke AIPI 561 Final Group Project_
&nbsp;

# An Microservice ML Endpoint based on AutoML Solution using Ludwig that uses Continuous Integration & Continuous Delivery 

This project showcases an Automated Machine Learning (AutoML) solution using Ludwig, Uber's open source deep learning toolbox. The advantages of using Ludwig are listed under the [Ludwig Features](#Features) section below. The project also showcases a microservice architecture that uses Continuous Integration & Continuous Delivery (CI/CD) to deploy the model as a Flask API endpoint. The CI/CD pipeline is implemented using GitHub Actions. The project also includes a frontend that allows users to interact with the model via a web interface using Streamlit in the [Usage](#Usage) section.

## Ludwig Features

1. **Easy-to-Use**: Ludwig is designed to be user-friendly and requires no programming experience. All you need to train
   a model and make predictions is a tabular dataset and a few lines of command-line code.
2. **Flexible**: Ludwig enables you to train a deep learning model using various types of input data such as text,
   images, and more. You can also combine different types of input data to create a hybrid model.
3. **Versatile**: Ludwig is suitable for a wide range of tasks such as image classification, text classification, and
   time series forecasting.

## Getting Started

### Requirements

- Python 3.7+
- See reqirements.txt for more details

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

- Model Training  (Optional):

```bash
python train_automl.py --data-path ./data/agnews.csv --config-path ./config/config.json
```


- Model Generation (Optional):


```bash
make gen
```

- Model prediction using pre-trained model:
```bash
python predict_automl_torch.py --predict-path ./data/text_to_predict.csv
```

- API server:
```bash
flask run --host=0.0.0.0 --port=80
```

- Docker Run:

```bash
docker run  -p 8080:8080 szheng3/automl-python
```

- Frontend:
```bash
streamlit run frontend.py
```
## Acknowledgements

This project makes use of the fantastic Ludwig toolbox developed by Uber's AI team. We thank them for their work and for
making it open source.

