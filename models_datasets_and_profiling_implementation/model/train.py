import os
import torch
from .net import NetFactory
from ..dataset import DatasetFactory
from . import utils

class ModelTrain: 
    def __init__(self, model, dataset):
        self.model_name = model.name
        self.dataset_name = dataset.name
        self.model = model
        self.dataset = dataset
        self.trainloader = self.dataset.get_trainloader()
        self.valloader = self.dataset.get_valloader()
        self.eps = 1e-8

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
        pre_norm,
        active_idx_layers=None
    ):
        dataset = DatasetFactory(dataset_name=dataset_name, batch_size=batch_size, valsplit=valsplit, num_workers=num_workers).get_dataset()
        num_classes = dataset.get_output_shape()[-1]
        print(f"Num classes: {num_classes}, Input shape: {dataset.input_shape}, d_model: {d_model}")
        model = NetFactory(
            model_name=model_name, 
            dataset_name=dataset_name,
            vocab_size=dataset.vocab_size if hasattr(dataset, 'vocab_size') else dataset.input_shape[-1],
            d_model=d_model, 
            d_state=d_state,
            depth=depth,
            dropout=dropout,     
            num_classes=num_classes,
            norm=norm,
            pre_norm=pre_norm,
            active_idx_layers=active_idx_layers
        ).get_net()
        model.name = model_name
        return ModelTrain(model, dataset)

    def train_step(self, epoch, optimizer, loss_criterion):
        self.model.train()

        acc_loss = 0.0
        correct = 0
        total = 0
        precision_list_train = []
        recall_list_train = []
        f1_list_train = []

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
            if isinstance(loss_criterion, torch.nn.BCEWithLogitsLoss):
                # Multi-label prediction (threshold = 0)
                predicted = (outputs > 0).float()

                # Accuracy multilabel: confronto elemento per elemento
                correct += (predicted == labels).sum().item()
                total += labels.numel()   # NOTA: numel, non batch_size
                tp = torch.zeros(6, dtype=torch.float)
                fp = torch.zeros(6, dtype=torch.float)
                fn = torch.zeros(6, dtype=torch.float)
                tn = torch.zeros(6, dtype=torch.float)
                for c in range(6):
                    tp[c] = ((predicted[:, c] == 1) & (labels[:, c] == 1)).sum().item()
                    fp[c] = ((predicted[:, c] == 1) & (labels[:, c] == 0)).sum().item()
                    fn[c] = ((predicted[:, c] == 0) & (labels[:, c] == 1)).sum().item()
                    tn[c] = ((predicted[:, c] == 0) & (labels[:, c] == 0)).sum().item()
                precision = tp.sum() / (tp.sum() + fp.sum() + self.eps)
                recall = tp.sum() / (tp.sum() + fn.sum() + self.eps)  
                f1 = 2 * (precision * recall) / (precision + recall + self.eps)
                precision_list_train.append(precision.item())
                recall_list_train.append(recall.item())
                f1_list_train.append(f1.item())
            else:
                _, predicted = outputs.max(1)
                total += batch_size
                correct += (predicted == labels).sum().item()


            lr = optimizer.param_groups[0]["lr"]
            if i % 100 == 99:
                running_loss = acc_loss / total
                running_acc = 100.0 * correct / total
                if isinstance(loss_criterion, torch.nn.BCEWithLogitsLoss):
                    precision = sum(precision_list_train) / len(precision_list_train)
                    recall = sum(recall_list_train) / len(recall_list_train)
                    f1 = sum(f1_list_train) / len(f1_list_train)
                    print("[Epoch {}, Iteration {}] loss: {:.4f} | train_accuracy: {:.3f} | precision: {:.3f} | recall: {:.3f} | f1: {:.3f}]".format(epoch + 1, i + 1, running_loss, running_acc, precision * 100, recall * 100, f1 * 100))
                else:
                    print("[Epoch {}, Iteration {}] loss: {:.4f} | train_accuracy: {:.3f}]".format(epoch + 1, i + 1, running_loss, running_acc))

        epoch_loss = acc_loss / total
        epoch_acc = 100.0 * correct / total
        if i % 100 != 99:
            if isinstance(loss_criterion, torch.nn.BCEWithLogitsLoss):
                precision = sum(precision_list_train) / len(precision_list_train)
                recall = sum(recall_list_train) / len(recall_list_train)
                f1 = sum(f1_list_train) / len(f1_list_train)
                print("[Epoch {}, Iteration {}] loss: {:.4f} | train_accuracy: {:.3f} | precision: {:.3f} | recall: {:.3f} | f1: {:.3f}]".format(epoch + 1, i + 1, epoch_loss, epoch_acc, precision * 100, recall * 100, f1 * 100))
            else:
                print("[Epoch {}, Iteration {}] loss: {:.4f} | train_accuracy: {:.3f}]".format(epoch + 1, i + 1, epoch_loss, epoch_acc))
        return epoch_loss
    
    
    def val_step(self, loss_criterion=None):
        self.model.eval()
        correct = 0
        total = 0
        acc_loss = 0.0
        precision_list_train = []
        recall_list_train = []
        f1_list_train = []
        with torch.no_grad():
            for data in self.valloader:
                inputs, labels = data
                inputs, labels = inputs.to(utils.get_device(), non_blocking=True), labels.to(utils.get_device(), non_blocking=True)
                outputs = self.model(inputs)
                batch_size = labels.size(0)

                if self.dataset_name == "ecg":
                    predicted = (outputs > 0).float()
                    correct += (predicted == labels).sum().item()
                    total += labels.numel()
                    tp = torch.zeros(6, dtype=torch.float)
                    fp = torch.zeros(6, dtype=torch.float)
                    fn = torch.zeros(6, dtype=torch.float)
                    tn = torch.zeros(6, dtype=torch.float)
                    for c in range(6):
                        tp[c] = ((predicted[:, c] == 1) & (labels[:, c] == 1)).sum().item()
                        fp[c] = ((predicted[:, c] == 1) & (labels[:, c] == 0)).sum().item()
                        fn[c] = ((predicted[:, c] == 0) & (labels[:, c] == 1)).sum().item()
                        tn[c] = ((predicted[:, c] == 0) & (labels[:, c] == 0)).sum().item()
                    precision = tp.sum() / (tp.sum() + fp.sum() + self.eps)
                    recall = tp.sum() / (tp.sum() + fn.sum() + self.eps)  
                    f1 = 2 * (precision * recall) / (precision + recall + self.eps)
                    precision_list_train.append(precision.item())
                    recall_list_train.append(recall.item())
                    f1_list_train.append(f1.item())
                else:
                    _, predicted = outputs.max(1)
                    total += batch_size
                    correct += (predicted == labels).sum().item()

                if loss_criterion is not None:
                    loss = loss_criterion(outputs, labels)
                    acc_loss += loss.item() * batch_size

        val_accuracy = 100.0 * correct / total
        if loss_criterion is not None:
            val_loss = acc_loss / total
            if isinstance(loss_criterion, torch.nn.BCEWithLogitsLoss):
                precision = sum(precision_list_train) / len(precision_list_train)
                recall = sum(recall_list_train) / len(recall_list_train)
                f1 = sum(f1_list_train) / len(f1_list_train)
                #print(f"[Validation] val_loss: {val_loss:.4f} | val_accuracy: {val_accuracy:.3f} | precision: {precision:.3f} | recall: {recall:.3f} | f1: {f1:.3f}]")
                return val_accuracy, val_loss, f1
            else:
                #print(f"[Validation] val_loss: {val_loss:.4f} | val_accuracy: {val_accuracy:.3f}]")
                return val_accuracy, val_loss
        if self.dataset_name == "ecg":
            f1 = sum(f1_list_train) / len(f1_list_train)
            return val_accuracy, f1
        else:
            return val_accuracy

    def run(self, optimizer, scheduler, loss_criterion, epochs, checkpoints_folder, is_pruned=False, patience=None):
        best_val_accuracy = 0.0
        best_f1_score = 0.0

        # Determina il nome base del file per questa run
        if is_pruned:
            model_path = checkpoints_folder
        else:
            model_path = os.path.join(checkpoints_folder, f"{self.model_name}_{self.dataset_name}_best.pth")

        i = 0
        for epoch in range(int(epochs)):
            epoch_loss = self.train_step(epoch, optimizer, loss_criterion)
            if isinstance(loss_criterion, torch.nn.BCEWithLogitsLoss):
                val_accuracy, val_loss, val_f1 = self.val_step(loss_criterion)
                print(f"[Epoch {epoch+1} Summary] loss: {epoch_loss:.3f} | val_loss {val_loss:.3f} | val_accuracy: {val_accuracy:.3f} | val_f1: {val_f1 * 100:.3f}]")
            else:
                val_accuracy, val_loss = self.val_step(loss_criterion)
                print(f"[Epoch {epoch+1} Summary] loss: {epoch_loss:.3f} | val_loss {val_loss:.3f} | val_accuracy: {val_accuracy:.3f}]")

            if scheduler is not None:
                scheduler.step()

            # Salvataggio best: sempre nello stesso file di questa run
            if isinstance(loss_criterion, torch.nn.BCEWithLogitsLoss):
                if val_f1 > best_f1_score:
                    best_f1_score = val_f1
                    self.save(model_path)
                    i = 0
                else:
                    if patience is not None:
                        i += 1
                        if i > patience:
                            print(f"Early stopping at epoch {epoch+1} due to no improvement in F1 score for {patience} consecutive epochs.")
                            break
            else:
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    self.save(model_path)
                    i = 0
                else:
                    if patience is not None:
                        i += 1
                        if i > patience:
                            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation accuracy for {patience} consecutive epochs.")
                            break

    def save(self, path):
        #torch.save(self.model.state_dict(), path) 
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "active_idx_layers": self.model.active_idx_layers
        }, path)