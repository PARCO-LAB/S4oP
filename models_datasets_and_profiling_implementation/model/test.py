import os
import torch
from .train import ModelTrain, format_metrics
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
        num_classes = dataset.num_classes

        ckpt = torch.load(model_path, map_location=get_device())
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            active_idx_layers = ckpt["active_idx_layers"]
        else:
            active_idx_layers = None

        model = NetFactory(
            model_name=model_name,
            dataset_name=dataset_name,
            vocab_size=dataset.vocab_size,
            input_size=dataset.input_size,
            d_model=d_model,
            d_state=d_state,
            depth=depth,
            dropout=dropout,
            num_classes=num_classes,
            norm=norm,
            pre_norm=pre_norm,
            active_idx_layers=active_idx_layers,
            dual_stream=dataset.dual_stream,
            pool=getattr(dataset, "pool", "mean")
        ).get_net()

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
        else:
            model.load_state_dict(torch.load(model_path, map_location=get_device()), strict=True)
        model.name = model_name

        return ModelTest(model, dataset)

    def test_step(self):
        return self.model_train.val_step(loader=self.dataset.get_testloader())

    def run(self):
        metrics = self.test_step()
        print("[Test] {}".format(format_metrics(metrics, prefix="test_")))
        # restituisce la metrica su cui il dataset chiede di essere valutato
        return metrics[self.model_train.metric]