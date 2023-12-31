import unittest
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from challenge.model import DelayModel

class TestModel(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.model = DelayModel()
        self.data = pd.read_csv(filepath_or_buffer="data/data.csv")
        
    def test_model_preprocess_for_training(
        self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        assert isinstance(features, pd.DataFrame)
        assert features.columns.to_list() == [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        assert isinstance(target, pd.DataFrame)
        assert target.columns == [
            "delay"
        ]

    def test_model_preprocess_for_serving(
        self
    ):
        features = self.model.preprocess(
            data=self.data
        )

        assert isinstance(features, pd.DataFrame)
        assert features.columns.to_list() == [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]

    def test_model_fit(
        self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        _, features_validation, _, target_validation = train_test_split(features, target, test_size = 0.33, random_state = 42)

        self.model.fit(
            features=features,
            target=target
        )

        predicted_target = self.model._model.predict(
            features_validation
        )

        report = classification_report(target_validation, predicted_target, output_dict=True)
        
        assert report["0"]["recall"] < 0.60
        assert report["0"]["f1-score"] < 0.70
        assert report["1"]["recall"] > 0.60
        assert report["1"]["f1-score"] > 0.30

    def test_model_predict(
        self
    ):
        # x_train, y_train = self.model.preprocess(data=self.data, target_column='delay')
        # self.model.fit(x_train, y_train)
        features, target = self.model.preprocess(data=self.data, target_column='delay')
        self.model.fit(features, target)

        # features = self.model.preprocess(
        #     data=self.data
        # )

        predicted_targets = self.model.predict(
            features=features
        )

        assert isinstance(predicted_targets, list)
        assert len(predicted_targets) == features.shape[0]
        assert all(isinstance(predicted_target, int) for predicted_target in predicted_targets)