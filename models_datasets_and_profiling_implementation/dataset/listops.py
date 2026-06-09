import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from .interface import DatasetInterface


class ListOpsLazy(Dataset):
    def __init__(self, tsv_path, vocab, max_len):
        self.df = pd.read_csv(tsv_path, sep="\t", header=0)
        self.vocab = vocab
        self.pad_idx = vocab["<PAD>"]
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = self.df.iloc[idx, 0]
        label = int(self.df.iloc[idx, 1])

        tokens = [self.vocab[t] for t in seq.split()]
        tokens = tokens[:self.max_len]
        pad_len = self.max_len - len(tokens)
        if pad_len > 0:
            tokens += [self.pad_idx] * pad_len

        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class LRAListOps(DatasetInterface):
    def __init__(self, batch_size, valsplit, num_workers):
        super().__init__("listops", batch_size, num_workers)

        root_data = "data"
        train_path = os.path.join(root_data, "listops/train.tsv")
        val_path = os.path.join(root_data, "listops/val.tsv")
        test_path = os.path.join(root_data, "listops/test.tsv")

        train_df = pd.read_csv(train_path, sep="\t", header=0)
        val_df = pd.read_csv(val_path,   sep="\t", header=0)
        test_df = pd.read_csv(test_path,  sep="\t", header=0)

        unique_tokens = set()
        for df in (train_df, val_df, test_df):
            for seq in df.iloc[:, 0].values:
                unique_tokens.update(seq.split())

        self.vocab = {"<PAD>": 0}
        for idx, tok in enumerate(sorted(unique_tokens), start=1):
            self.vocab[tok] = idx

        self.vocab_size = len(self.vocab)
        self.labels = list(range(10))

        # Lunghezza massima di sequenza su tutti gli split (per il padding)
        max_len = max(
            max(len(seq.split()) for seq in train_df.iloc[:, 0].values),
            max(len(seq.split()) for seq in val_df.iloc[:, 0].values),
            max(len(seq.split()) for seq in test_df.iloc[:, 0].values),
        )

        # Tre dataset separati: niente più random_split
        self.trainset = ListOpsLazy(train_path, self.vocab, max_len)
        self.valset = ListOpsLazy(val_path,   self.vocab, max_len)
        self.testset = ListOpsLazy(test_path,  self.vocab, max_len)

        self.input_shape = (batch_size, max_len)

        print(f"LRAListOps: {len(self.trainset)} train, {len(self.valset)} val, {len(self.testset)} test, seq_len={max_len}, vocab_size={self.vocab_size}")