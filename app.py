import os

from flask import Flask, request
from ModelManager import ModelManager

app = Flask(__name__)
models_config_ = None


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
    print(models_config_)


@app.route('/predict')
def get_prediction():
    # print(request.args.get("drug"), request.args.get("timestamp"))
    return request.args


if __name__ == '__main__':
    app.run()
