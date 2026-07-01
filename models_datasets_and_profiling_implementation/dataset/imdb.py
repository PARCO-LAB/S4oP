import torch
from torch.utils.data import TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer
from .interface import DatasetInterface


class IMDB(DatasetInterface):
    def __init__(self, batch_size, valsplit, num_workers):
        super().__init__("imdb", batch_size, num_workers)

        # Carica dataset IMDB
        tokenizer_name="bert-base-uncased"
        raw_train = load_dataset("stanfordnlp/imdb", split="train")
        raw_test  = load_dataset("stanfordnlp/imdb", split="test")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        def tokenize_dataset(dataset):
            texts = list(dataset["text"])
            labels = list(dataset["label"])
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=4096,
            )
            X = torch.tensor(encodings["input_ids"], dtype=torch.long)
            y = torch.tensor(labels, dtype=torch.long)
            return X, y

        X_train, y_train = tokenize_dataset(raw_train)
        X_test, y_test   = tokenize_dataset(raw_test)
        self.vocab_size = len(self.tokenizer)
        print(f"Numero di sample per ciascuna classe: {torch.bincount(y_test)}")

        # Train/val split
        val_size = int(valsplit * len(X_train))
        train_size = len(X_train) - val_size

        indices = torch.randperm(len(X_train))
        train_idx, val_idx = indices[:train_size], indices[train_size:]

        self.trainset = TensorDataset(X_train[train_idx], y_train[train_idx])
        self.valset   = TensorDataset(X_train[val_idx], y_train[val_idx])
        self.testset  = TensorDataset(X_test, y_test)

        # Labels
        self.labels = [0, 1]

        # Input shape
        self.input_shape = (batch_size, X_train.size(1))

        print(f"IMDB: {len(self.trainset)} train, {len(self.valset)} val, {len(self.testset)} test, seq_len={X_train.size(1)}, vocab_size={self.vocab_size}")