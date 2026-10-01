import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(
            os.path.exists("breast_cancer.csv")
        )

    def test_model_created(self):
        self.assertTrue(
            os.path.exists("breast_cancer_model.pkl")
        )

    def test_metrics_created(self):
        self.assertTrue(
            os.path.exists("metrics.json")
        )

    def test_accuracy_is_valid(self):

        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]

        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):

        model = joblib.load(
            "breast_cancer_model.pkl"
        )

        data = pd.read_csv(
            "breast_cancer.csv"
        )

        X = data.drop("target", axis=1)

        sample = X.iloc[[0]]

        prediction = model.predict(sample)[0]

        self.assertIn(
            int(prediction),
            [2]
        )


if __name__ == "__main__":
    unittest.main()
