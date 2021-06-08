import pandas as pd
import numpy as np
import os

from sklearn import linear_model
from sklearn.model_selection import train_test_split


class ModelManager:
    def __init__(self, data_source):
        self.data_source = data_source
        self.data_frame_x = pd.DataFrame([], columns=["TIMESTAMP"])
        self.data_frame_y = pd.DataFrame([], columns=["QUANTITY"])
        self.data_set_x = []
        self.data_set_y = []
        self.x_test = []
        self.x_train = []
        self.y_test = []
        self.y_train = []
        self.regression_model = linear_model.LinearRegression()

    def load_data_set(self):
        with open(self.data_source, 'r') as file:
            for line in file.readlines():
                y_column, x_column = map(float, line.split(','))
                self.data_set_x.append(x_column)
                self.data_set_y.append(y_column)

    def transform_dataset_to_dataframe(self):
        self.data_frame_x = pd.DataFrame(self.data_set_x, columns=["TIMESTAMP"])
        self.data_frame_y = pd.DataFrame(self.data_set_y, columns=["QUANTITY"])

    def print_dataset_description(self):
        print(self.data_frame_x.describe())
        print(self.data_frame_y.describe())

    def split_data_for_testing_and_training(self, percent, seed_value):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data_frame_x, self.data_frame_y, test_size=percent, random_state=seed_value)

    def train_model(self):
        self.regression_model.fit(self.x_train, self.y_train)

    def prepare_model(self):
        self.load_data_set()
        self.transform_dataset_to_dataframe()
        self.split_data_for_testing_and_training(0.23, 123)
        self.train_model()

    def get_prediction(self, x_test):
        return self.regression_model.predict(x_test)

    def get_x_test_data(self):
        return self.x_test
