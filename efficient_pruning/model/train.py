import os
import torch
import torch_pruning as tp
import torch.nn.functional as F

from .net import NetFactory
from ..dataset import DatasetFactory, DatasetInterface
from . import utils

class ModelTrain: 
    def __init__(self, model, dataset):
        self.model_name = model.name
        self.dataset_name = dataset.name
        self.model = model
        self.dataset = dataset
        self.trainloader = self.dataset.get_trainloader()
        self.valloader = self.dataset.get_valloader()

    
    @staticmethod
    def from_scratch(
        model_name, 
        dataset_name, 
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
        dataset = DatasetFactory(dataset_name=dataset_name, batch_size=batch_size, valsplit=valsplit, num_workers=num_workers).get_dataset()
        num_classes = dataset.get_output_shape()[-1]
        print(f"Num classes: {num_classes}, Input shape: {dataset.input_shape}, d_model: {d_model}")
        model = NetFactory(
            model_name=model_name, 
            vocab_size=dataset.vocab_size, 
            d_model=d_model, 
            d_state=d_state,
            depth=depth,
            dropout=dropout,     
            num_classes=num_classes,
            norm=norm,
            pre_norm=pre_norm
        ).get_net()
        model.name = model_name
        return ModelTrain(model, dataset)

    def train_step(self, epoch, optimizer, loss_criterion):
        self.model.train()

        acc_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(self.trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(utils.get_device(), non_blocking=True), labels.to(utils.get_device(), non_blocking=True)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            acc_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            total += batch_size
            correct += (predicted == labels).sum().item()

            lr = optimizer.param_groups[0]["lr"]
            if i % 100 == 99:
                running_loss = acc_loss / total
                running_acc = 100.0 * correct / total
                print("[Epoch {}, Iteration {}] lr: {:.3f} loss: {:.3f} train_accuracy: {:.3f}".format(
                    epoch + 1, i + 1, lr, running_loss, running_acc))

        epoch_loss = acc_loss / total
        epoch_acc = 100.0 * correct / total
        if i % 100 != 99:
            print("[Epoch {}, Iteration {}] lr: {:.3f} loss: {:.3f} train_accuracy: {:.3f}".format(epoch + 1, i + 1, lr, epoch_loss, epoch_acc))
        return epoch_loss
    
    
    def val_step(self, loss_criterion=None):
        self.model.eval()
        correct = 0
        total = 0
        acc_loss = 0.0
        with torch.no_grad():
            for data in self.valloader:
                inputs, labels = data
                inputs, labels = inputs.to(utils.get_device(), non_blocking=True), labels.to(utils.get_device(), non_blocking=True)
                outputs = self.model(inputs)

                _, predicted = outputs.max(1)
                batch_size = labels.size(0)
                total += batch_size
                correct += (predicted == labels).sum().item()

                if loss_criterion is not None:
                    loss = loss_criterion(outputs, labels)
                    acc_loss += loss.item() * batch_size

        val_accuracy = 100.0 * correct / total
        if loss_criterion is not None:
            val_loss = acc_loss / total
            return val_accuracy, val_loss
        return val_accuracy
    

    def run(self, optimizer, scheduler, loss_criterion, epochs, checkpoints_folder=os.path.join(".", "checkpoints")):
        best_val_accuracy = 0.0
        for epoch in range(int(epochs)):
            epoch_loss = self.train_step(epoch, optimizer, loss_criterion)
            val_accuracy, val_loss = self.val_step(loss_criterion)
            print("[Epoch {} Summary] loss: {:.3f} val_loss {:.3f} val_accuracy: {:.3f}".format(
                epoch + 1, epoch_loss, val_loss, val_accuracy))
            
            # Step scheduler
            if scheduler is not None:
                scheduler.step()

            # Salvataggio modello migliore
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save("{}_{}_best".format(self.model_name, self.dataset_name), checkpoints_folder=checkpoints_folder)

            # Checkpoint periodici
            if (epoch+1) % 100 == 0:
                self.save("{}_{}_epoch{}".format(self.model_name, self.dataset_name, epoch), checkpoints_folder=checkpoints_folder)

    def save(self, name, checkpoints_folder=os.path.join(".", "checkpoints")):
        os.makedirs(checkpoints_folder, exist_ok=True)
        model_path = os.path.join(checkpoints_folder, "{}.pth".format(name))
        self.model.zero_grad()
        torch.save(self.model.state_dict(), model_path)
