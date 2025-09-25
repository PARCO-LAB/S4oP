import torch
from torch.utils.data import TensorDataset
import pandas as pd
import os
from .interface import DatasetInterface

class LRAListOps(DatasetInterface):
    def __init__(self, batch_size, valsplit, num_workers):
        super().__init__("listops", batch_size, num_workers)

        # Parametri per diminuzione del dataset per test veloci
        root_data="data/" 
        #max_seq_len=1500
        #subset_size=20000

        # Definisci il vocabolario
        self.vocab = {
            "(": 0, ")": 1, "[": 2, "]": 3,
            "MIN": 4, "MAX": 5, "MED": 6, "SM": 7
        }
        # aggiungi numeri da 0 a 9 come token
        for i in range(10):
            self.vocab[str(i)] = 8 + i  # 8->0 fino a 17

        # Numero totale di classi
        self.labels = list(range(10))
        self.vocab_size = len(self.vocab)

        # Leggi i file TSV
        train_path = os.path.join(root_data, "listops/train.tsv")
        test_path  = os.path.join(root_data, "listops/test.tsv")

        train_df = pd.read_csv(train_path, sep="\t", header=0)  # header=0 perché la prima riga è "Source Target"
        test_df  = pd.read_csv(test_path, sep="\t", header=0)

        # Tokenizza le sequenze
        def tokenize_sequence(seq):
            tokens = seq.split()
            return [self.vocab[t] for t in tokens if t in self.vocab]

        # Applica tokenizzazione e taglia lunghezza massima
        X_train = [tokenize_sequence(seq) for seq in train_df.iloc[:,0].values]
        y_train = torch.tensor(train_df.iloc[:,1].values, dtype=torch.long)

        X_test  = [tokenize_sequence(seq) for seq in test_df.iloc[:,0].values]
        y_test  = torch.tensor(test_df.iloc[:,1].values, dtype=torch.long)

        # Padding sequenze corte
        max_len_train = max(len(x) for x in X_train)
        max_len_test  = max(len(x) for x in X_test)
        max_len = max(max_len_train, max_len_test)

        def pad_sequence(seq, length):
            return seq + [0]*(length - len(seq))

        X_train = torch.tensor([pad_sequence(seq, max_len) for seq in X_train], dtype=torch.long)
        X_test  = torch.tensor([pad_sequence(seq, max_len) for seq in X_test], dtype=torch.long)

        # Train/validation split
        val_size = int(valsplit * len(X_train))
        train_size = len(X_train) - val_size

        self.trainset = TensorDataset(X_train[:train_size], y_train[:train_size])
        self.valset   = TensorDataset(X_train[train_size:], y_train[train_size:])
        self.testset  = TensorDataset(X_test, y_test)

        # Input shape
        self.input_shape = (batch_size, X_train.size(1))

        print(f"LRAListOps: {len(self.trainset)} train, {len(self.valset)} val, {len(self.testset)} test, seq_len={max_len}")
