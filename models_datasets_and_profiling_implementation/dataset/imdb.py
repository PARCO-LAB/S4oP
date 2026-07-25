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
        raw_test = load_dataset("stanfordnlp/imdb", split="test")

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
        X_test, y_test = tokenize_dataset(raw_test)

        # Train/val split
        val_size = int(valsplit * len(X_train))
        train_size = len(X_train) - val_size

        perm = torch.randperm(len(X_train), generator=torch.Generator().manual_seed(42))
        train_idx, val_idx = perm[:train_size], perm[train_size:]

        self.trainset = TensorDataset(X_train[train_idx], y_train[train_idx])
        self.valset = TensorDataset(X_train[val_idx], y_train[val_idx])
        self.testset = TensorDataset(X_test, y_test)

        self.labels = [0, 1]
        self.seq_len = X_train.shape[1] # 4096
        self.input_size = 1 # un token per timestep
        self.num_classes = 2
        self.vocab_size = len(self.tokenizer) # tokenizzato -> nn.Embedding
        self.input_shape = (batch_size, self.seq_len)  # 2D per input tokenizzato

        self.summary(extra=f"pos/classe={torch.bincount(y_test).tolist()}")