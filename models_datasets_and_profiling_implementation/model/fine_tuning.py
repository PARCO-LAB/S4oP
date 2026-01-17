import torch
from . import ModelTrain
from .utils import setup_optimizer

class FineTuning: 
    def __init__(self, 
                 model,
                 dataset,
                 epochs,
                 lr,
                 weight_decay,
                 checkpoint_folder,
                 patience):
        self.model_train = ModelTrain(model, dataset)
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.checkpoint_folder = checkpoint_folder
        self.patience = patience

    def run(self): 
        criterion = torch.nn.CrossEntropyLoss() if self.model_train.dataset.name != "ecg" else torch.nn.BCEWithLogitsLoss()

        optimizer, scheduler = setup_optimizer(
            model=self.model_train.model,
            lr=self.lr,
            weight_decay=self.weight_decay,
            epochs=self.epochs
        )

        self.model_train.run(
            optimizer=optimizer,
            scheduler=scheduler,
            loss_criterion=criterion,
            epochs=self.epochs,
            checkpoints_folder=self.checkpoint_folder,
            is_pruned=True,
            patience=self.patience
        )
        
