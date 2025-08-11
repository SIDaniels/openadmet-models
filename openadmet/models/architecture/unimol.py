
import torch
from unimol_tools.unimol_tools import MolTrain, MolPredict
from openadmet.models.architecture.model_base import ModelWrapper

class UniMolWrapper(ModelWrapper):
    def __init__(self, **kwargs):
        super(UniMolWrapper, self).__init__(**kwargs)
        self.task = kwargs.get("task", "classification")
        self.data_type = kwargs.get("data_type", "molecule")
        self.epochs = kwargs.get("epochs", 10)
        self.batch_size = kwargs.get("batch_size", 16)
        self.metrics = kwargs.get("metrics", "auc")
        self.model_name = kwargs.get("model_name", "unimolv1")
        self.model_size = kwargs.get("model_size", "84m")
        
        self.model = MolTrain(
            task=self.task,
            data_type=self.data_type,
            epochs=self.epochs,
            batch_size=self.batch_size,
            metrics=self.metrics,
            model_name=self.model_name,
            model_size=self.model_size,
        )

    def fit(self, train_data, valid_data=None):
        self.model.fit(data=train_data)

    def predict(self, data):
        predictor = MolPredict(load_model=self.model.exp_path)
        return predictor.predict(data=data)
