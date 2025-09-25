import os
import torch
from .train import ModelTrain
from ..dataset import DatasetFactory
from .net import NetFactory
from .utils import *


class ModelTest: 
    def __init__(self, model, dataset):
        self.model_name = model.name
        self.dataset_name = dataset.name
        self.model = model
        self.model.eval()
        self.dataset = dataset
        self.model_train = ModelTrain(model, dataset)

    @staticmethod
    def from_pth(
        model_path, 
        batch_size, 
        valsplit, 
        num_workers, 
        d_model,
        d_state,
        depth,
        dropout,
        norm,
        pre_norm
    ):
        basename = os.path.basename(model_path)
        basename_split = basename.split("_")
        model_name, dataset_name = basename_split[0], basename_split[1]

        dataset = DatasetFactory(dataset_name=dataset_name, batch_size=batch_size, valsplit=valsplit, num_workers=num_workers).get_dataset()
        num_classes = dataset.get_output_shape()[-1]
        model = NetFactory(
            model_name=model_name, 
            vocab_size=dataset.vocab_size,
            d_model=d_model, 
            d_state=d_state,
            depth=depth,
            dropout=dropout,
            num_classes=num_classes,
            norm=norm,
            pre_norm=pre_norm,
        ).get_net()
        model.load_state_dict(torch.load(model_path), strict=False)
        model.name = model_name

        return ModelTest(model, dataset)

    def test_step(self):
        return self.model_train.val_step()

    def run(self):
        self.model_train.valloader = self.model_train.dataset.get_testloader()
        accuracy = self.test_step()
        print("[Test] test_accuracy: {:.3f}".format(accuracy))
        return accuracy