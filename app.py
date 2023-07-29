from flask import Flask, request, jsonify
import pandas as pd
import torch
from ludwig.models.inference import InferenceModule
import json
from ludwig.utils.inference_utils import to_inference_module_input_from_dataframe
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

app = Flask(__name__)

# Load the model when the server starts
inference_module = InferenceModule.from_directory('./model/')
with open(f"model/model_hyperparameters.json") as f:
    config = json.load(f)

@app.route('/')
def health():
    return "OK"


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)

    # Make sure that 'title' key is in the data
    if 'title' not in data:
        return jsonify({"error": "Missing 'title' in request data"}), 400

    # Convert the data to a DataFrame
    text_to_predict = pd.DataFrame({
        "title": [data["title"]]  # Wrap the title string in a list to create a DataFrame
    })

    # Predict using the model
    input_sample_dict = to_inference_module_input_from_dataframe(text_to_predict, config)
    output_df = inference_module(input_sample_dict)

    # Return the prediction for the first (and only) item
    return jsonify({"prediction": output_df["class"]["predictions"][0]})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
