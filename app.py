import os
import numpy as np

from flask import Flask, request, abort
from flask_cors import CORS, cross_origin

from ModelManager import ModelManager

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
models_config_ = {}


def generate_models_from_csv():
    models_config = {}
    for root, dirs, files in os.walk("./licenta_data_preprocessing/"):
        for file in files:
            if file.endswith(".csv"):
                current_model = ModelManager(root + file)
                models_config[file] = current_model
    return models_config


def train_models(model_config):
    for file_name, model_name in model_config.items():
        try:
            model_config[file_name].prepare_model()
        except Exception as ex:
            print(ex)
            print(file_name)


@app.before_first_request
def prepare_models():
    global models_config_

    print("Preparing models...")
    models_config_ = generate_models_from_csv()
    print("DONE preparing models...")

    print("Training models...")
    train_models(models_config_)
    print("DONE training models... ")


@app.route('/predict')
def get_prediction():
    print(request.args)
    if "drug" in request.args and "timestamp" in request.args:
        drug_name = request.args.get("drug")
        timestamp = request.args.get("timestamp")
        if drug_name + ".csv" in models_config_:
            return str(models_config_.get(drug_name + ".csv").get_prediction(np.array(timestamp).reshape(-1, 1)))
        return "No data"
    else:
        abort(400, 'Invalid url format, provide valid drug name and timestamp')


if __name__ == '__main__':
    app.run()
