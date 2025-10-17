import torch
from torch.utils.data import TensorDataset, random_split
import pandas as pd
import os
from .interface import DatasetInterface

class LRAListOps(DatasetInterface):
    def __init__(self, batch_size, valsplit, num_workers):
        super().__init__("listops", batch_size, num_workers)

        root_data = "data/" 

        # Carica i file TSV
        train_path = os.path.join(root_data, "listops/train.tsv")
        test_path  = os.path.join(root_data, "listops/test.tsv")

        train_df = pd.read_csv(train_path, sep="\t", header=0)
        test_df  = pd.read_csv(test_path, sep="\t", header=0)

        # Vocabolario dinamico 
        unique_tokens = set()
        for seq in list(train_df.iloc[:,0].values) + list(test_df.iloc[:,0].values):
            unique_tokens.update(seq.split())

        # Aggiungi token PAD = 0
        self.vocab = {"<PAD>": 0}
        for idx, tok in enumerate(sorted(unique_tokens), start=1):
            self.vocab[tok] = idx

        self.vocab_size = len(self.vocab)

        # Numero totale di classi (0–9)
        self.labels = list(range(10))

        # Funzione di tokenizzazione delle sequenze
        def tokenize_sequence(seq):
            tokens = seq.split()
            return [self.vocab[t] for t in tokens]

        X_train = [tokenize_sequence(seq) for seq in train_df.iloc[:,0].values]
        y_train = torch.tensor(train_df.iloc[:,1].values, dtype=torch.long)

        X_test  = [tokenize_sequence(seq) for seq in test_df.iloc[:,0].values]
        y_test  = torch.tensor(test_df.iloc[:,1].values, dtype=torch.long)
        print(f"Numero di sample per ciascuna classe: {torch.bincount(y_test)}")

        # Padding
        max_len_train = max(len(x) for x in X_train)
        max_len_test  = max(len(x) for x in X_test)
        max_len = max(max_len_train, max_len_test)

        def pad_sequence(seq, length):
            return seq + [self.vocab["<PAD>"]] * (length - len(seq))

        X_train = torch.tensor([pad_sequence(seq, max_len) for seq in X_train], dtype=torch.long)
        X_test  = torch.tensor([pad_sequence(seq, max_len) for seq in X_test], dtype=torch.long)

        # Split train/val
        val_size = int(valsplit * len(X_train))
        train_size = len(X_train) - val_size
        full_train = TensorDataset(X_train, y_train)
        self.trainset, self.valset = random_split(full_train, [train_size, val_size])

        self.testset = TensorDataset(X_test, y_test)

        # Input shape
        self.input_shape = (batch_size, X_train.size(1))

        print(f"LRAListOps: {len(self.trainset)} train, {len(self.valset)} val, {len(self.testset)} test, seq_len={max_len}, vocab_size={self.vocab_size}")
