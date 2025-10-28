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
        raw_train = load_dataset("imdb", split="train")
        raw_test  = load_dataset("imdb", split="test")

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

""" import torch
from torch.utils.data import TensorDataset
import tensorflow_datasets as tfds
from .interface import DatasetInterface
from collections import Counter

class IMDB(DatasetInterface):
    def __init__(self, batch_size, valsplit, num_workers, max_length=1024):
        super().__init__("imdb", batch_size, num_workers)

        # Carica dataset IMDb da TFDS
        train_raw, val_raw, test_raw = self.get_imdb_dataset()

        # Costruzione del vocabolario
        self.vocab = self.build_vocab(train_raw)
        self.vocab_size = len(self.vocab)
        self.pad_idx = 0
        self.unk_idx = 1
        self.max_length = max_length

        # Tokenizzazione custom
        X_train, y_train = self.tokenize_dataset(train_raw)
        X_val, y_val     = self.tokenize_dataset(val_raw)
        X_test, y_test   = self.tokenize_dataset(test_raw)

        # Train/val split opzionale se val_raw è vuoto
        if valsplit > 0 and len(X_val) == 0:
            val_size = int(valsplit * len(X_train))
            train_size = len(X_train) - val_size
            indices = torch.randperm(len(X_train))
            train_idx, val_idx = indices[:train_size], indices[train_size:]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            X_train, y_train = X_train[train_idx], y_train[train_idx]

        # TensorDataset
        self.trainset = TensorDataset(X_train, y_train)
        self.valset   = TensorDataset(X_val, y_val)
        self.testset  = TensorDataset(X_test, y_test)

        # Labels
        self.labels = [0, 1]

        # Input shape
        self.input_shape = (batch_size, X_train.size(1))

        print(f"IMDB: {len(self.trainset)} train, {len(self.valset)} val, {len(self.testset)} test")
        print(f"Seq length={X_train.size(1)}, vocab_size={self.vocab_size}")
        print(f"Numero di sample per ciascuna classe: {torch.bincount(y_test)}")

    def get_imdb_dataset(self):
        Carica IMDb da TFDS e adatta gli esempi a Source/Target.    
        data = tfds.load('imdb_reviews')
        train_raw = data['train']
        test_raw  = data['test']

        def adapt_example(ex):
            return {'Source': ex['text'], 'Target': ex['label']}

        train = train_raw.map(adapt_example)
        val   = test_raw.map(adapt_example)  # IMDb non ha val set
        test  = test_raw.map(adapt_example)

        # Converti a numpy per PyTorch
        train = list(tfds.as_numpy(train))
        val   = list(tfds.as_numpy(val))
        test  = list(tfds.as_numpy(test))
        return train, val, test

    def build_vocab(self, dataset):
        Costruisce un vocabolario word-level dal training set.
        counter = Counter()
        for example in dataset:
            text = example['Source']
            if isinstance(text, bytes):
                text = text.decode('utf-8')
            tokens = text.lower().split()
            counter.update(tokens)
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for idx, tok in enumerate(counter.keys(), start=2):
            vocab[tok] = idx
        return vocab

    def tokenize_dataset(self, dataset):
        Tokenizza dataset e fa padding fino a max_length.
        X = []
        y = []
        for example in dataset:
            text = example['Source']
            if isinstance(text, bytes):
                text = text.decode('utf-8')
            label = int(example['Target'])

            # tokenizzazione
            tokens = text.lower().split()
            token_ids = [self.vocab.get(tok, self.unk_idx) for tok in tokens]

            # truncation / padding
            token_ids = token_ids[:self.max_length]
            pad_len = self.max_length - len(token_ids)
            token_ids += [self.pad_idx] * pad_len

            X.append(token_ids)
            y.append(label)

        X_tensor = torch.tensor(X, dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return X_tensor, y_tensor
 """