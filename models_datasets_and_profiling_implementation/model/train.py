import os
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from .net import NetFactory
from ..dataset import DatasetFactory
from . import utils

@torch.no_grad()
def grad_report(model):
    total = 0.0
    rows = []
    for name, p in model.named_parameters():
        if p.grad is None:
            rows.append((name, None))
            continue
        n = p.grad.detach().norm(2).item()
        total += n ** 2
        rows.append((name, n))
    total = total ** 0.5
    return total, rows


class MetricTracker:

    def __init__(self, multilabel, num_classes):
        self.multilabel = multilabel
        self.num_classes = num_classes
        # l'auprc binaria ha senso solo con due classi; per listops/image resta nan
        self.ranked = multilabel or num_classes == 2
        self.scores = []
        self.targets = []
        self.correct = 0
        self.total = 0
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.loss_sum = 0.0
        self.has_loss = False 
        self.n = 0
        self.eps = 1e-8

    @torch.no_grad()
    def update(self, outputs, labels, loss=None):
        batch_size = labels.size(0)
        self.n += batch_size
        if loss is not None:
            self.loss_sum += float(loss) * batch_size
            self.has_loss = True

        outputs = outputs.detach().float()

        if self.multilabel:
            predicted = (outputs > 0).float()
            self.correct += (predicted == labels).sum().item()
            self.total += labels.numel()
            self.tp += ((predicted == 1) & (labels == 1)).sum().item()
            self.fp += ((predicted == 1) & (labels == 0)).sum().item()
            self.fn += ((predicted == 0) & (labels == 1)).sum().item()
            self.scores.append(torch.sigmoid(outputs).cpu().numpy())
            self.targets.append(labels.detach().to(torch.uint8).cpu().numpy())
        else:
            self.correct += (outputs.argmax(1) == labels).sum().item()
            self.total += batch_size
            if self.ranked:
                self.scores.append(torch.softmax(outputs, dim=-1)[:, 1].cpu().numpy())
                self.targets.append(labels.detach().to(torch.uint8).cpu().numpy())

    def _auprc(self):
        """Average precision: media, sulle posizioni dei positivi nella classifica
        ordinata per punteggio, della precisione cumulata fino a quella posizione.
        Multi-label: una classifica per traccia, poi media (macro)."""
        if not self.ranked or not self.scores:
            return float("nan")
        scores = np.concatenate(self.scores)
        targets = np.concatenate(self.targets)
        if self.multilabel:
            aps = [
                average_precision_score(targets[:, c], scores[:, c])
                for c in range(targets.shape[1])
                if 0 < targets[:, c].sum() < len(targets)  # tracce con una sola classe: indefinite
            ]
            return float(np.mean(aps)) if aps else float("nan")
        if 0 < targets.sum() < len(targets):
            return float(average_precision_score(targets, scores))
        return float("nan")

    def compute(self):
        metrics = {"accuracy": 100.0 * self.correct / max(self.total, 1)}
        if self.n > 0 and self.has_loss:
            metrics["loss"] = self.loss_sum / self.n
        if self.multilabel:
            precision = self.tp / (self.tp + self.fp + self.eps)
            recall = self.tp / (self.tp + self.fn + self.eps)
            metrics["precision"] = 100.0 * precision
            metrics["recall"] = 100.0 * recall
            metrics["f1"] = 100.0 * 2 * precision * recall / (precision + recall + self.eps)
        metrics["auprc"] = 100.0 * self._auprc()
        return metrics


def format_metrics(metrics, prefix=""):
    order = ["loss", "accuracy", "auprc", "f1", "precision", "recall"]
    parts = []
    for k in order:
        if k not in metrics or (isinstance(metrics[k], float) and np.isnan(metrics[k])):
            continue
        parts.append(f"{prefix}{k}: {metrics[k]:.4f}" if k == "loss" else f"{prefix}{k}: {metrics[k]:.3f}")
    return " | ".join(parts)


class ModelTrain:
    def __init__(self, model, dataset):
        self.model_name = model.name
        self.dataset_name = dataset.name
        self.model = model
        self.dataset = dataset
        self.trainloader = self.dataset.get_trainloader()
        self.valloader = self.dataset.get_valloader()
        self.eps = 1e-8
        # Il dataset dichiara su cosa vuole essere valutato (self.metric).
        # Se non lo dichiara si torna al comportamento precedente.
        self.metric = getattr(dataset, "metric", None) or ("f1" if dataset.multilabel else "accuracy")

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
        active_idx_layers
    ):
        dataset = DatasetFactory(dataset_name=dataset_name, batch_size=batch_size, valsplit=valsplit, num_workers=num_workers).get_dataset()
        num_classes = dataset.num_classes
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
        model.name = model_name
        return ModelTrain(model, dataset)

    def new_tracker(self):
        return MetricTracker(self.dataset.multilabel, self.dataset.num_classes)

    def train_step(self, epoch, optimizer, loss_criterion):
        self.model.train()
        tracker = self.new_tracker()

        for i, data in enumerate(self.trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(utils.get_device(), non_blocking=True), labels.to(utils.get_device(), non_blocking=True)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            tracker.update(outputs, labels, loss=loss.item())

            if i % 100 == 99:
                # a meta' epoca l'auprc sarebbe calcolata su punteggi di modelli diversi: la ometto
                running = {k: v for k, v in tracker.compute().items() if k != "auprc"}
                print("[Epoch {}, Iteration {}] {}".format(epoch + 1, i + 1, format_metrics(running, prefix="train_")))

        metrics = tracker.compute()
        print("[Epoch {} Train] {}".format(epoch + 1, format_metrics(metrics, prefix="train_")))
        return metrics

    def val_step(self, loss_criterion=None, loader=None):
        self.model.eval()
        loader = self.valloader if loader is None else loader
        tracker = self.new_tracker()

        with torch.no_grad():
            for data in loader:
                inputs, labels = data
                inputs, labels = inputs.to(utils.get_device(), non_blocking=True), labels.to(utils.get_device(), non_blocking=True)
                outputs = self.model(inputs)
                loss = loss_criterion(outputs, labels).item() if loss_criterion is not None else None
                tracker.update(outputs, labels, loss=loss)

        return tracker.compute()

    def run(self, optimizer, scheduler, loss_criterion, epochs, checkpoints_folder, is_pruned=False, patience=None):
        best_score = -float("inf")

        if is_pruned:
            model_path = checkpoints_folder
        else:
            model_path = os.path.join(checkpoints_folder, f"{self.model_name}_{self.dataset_name}_best.pth")

        print(f"[Selezione del best checkpoint su: val_{self.metric}]")

        i = 0
        for epoch in range(int(epochs)):
            train_metrics = self.train_step(epoch, optimizer, loss_criterion)
            val_metrics = self.val_step(loss_criterion)
            print("[Epoch {} Summary] train_loss: {:.4f} | {}".format(
                epoch + 1, train_metrics["loss"], format_metrics(val_metrics, prefix="val_")))

            if scheduler is not None:
                scheduler.step()

            score = val_metrics.get(self.metric, float("nan"))
            if np.isnan(score):
                raise ValueError(
                    f"metrica '{self.metric}' non disponibile per il dataset {self.dataset_name} "
                    f"(metriche calcolate: {sorted(val_metrics)})"
                )

            if score > best_score:
                best_score = score
                self.save(model_path)
                i = 0
            else:
                if patience is not None:
                    i += 1
                    if i > patience:
                        print(f"Early stopping at epoch {epoch+1} due to no improvement in {self.metric} for {patience} consecutive epochs.")
                        break

        print(f"[Best] val_{self.metric}: {best_score:.3f}")
        return best_score

    def save(self, path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "active_idx_layers": self.model.active_idx_layers if self.model_name != "mamba2" else self.model.active_heads_layers
        }, path)