import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from .interface import DatasetInterface


class ListOpsLazy(Dataset):
    def __init__(self, df, vocab, max_len):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.pad_idx = vocab["<PAD>"]
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = self.df.iloc[idx, 0]
        label = int(self.df.iloc[idx, 1])
        tokens = [self.vocab[t] for t in seq.split()][:self.max_len]
        tokens += [self.pad_idx] * (self.max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class LRAListOps(DatasetInterface):
    def __init__(self, batch_size, valsplit, num_workers):
        super().__init__("listops", batch_size, num_workers)

        root_data = "data"
        paths = {s: os.path.join(root_data, f"listops/{s}.tsv")
                 for s in ("train", "val", "test")}
        dfs = {s: pd.read_csv(p, sep="\t", header=0) for s, p in paths.items()}

        # vocab da tutti gli split (val/test non devono avere token sconosciuti)
        unique_tokens = set()
        for df in dfs.values():
            for seq in df.iloc[:, 0].values:
                unique_tokens.update(seq.split())
        self.vocab = {"<PAD>": 0}
        for i, tok in enumerate(sorted(unique_tokens), start=1):
            self.vocab[tok] = i

        max_len = max(
            max(len(seq.split()) for seq in df.iloc[:, 0].values)
            for df in dfs.values()
        )

        self.trainset = ListOpsLazy(dfs["train"], self.vocab, max_len)
        self.valset = ListOpsLazy(dfs["val"], self.vocab, max_len)
        self.testset = ListOpsLazy(dfs["test"], self.vocab, max_len)

        # num_classes derivato dalle label reali
        all_labels = set()
        for df in dfs.values():
            all_labels.update(int(v) for v in df.iloc[:, 1].values)
        self.labels = sorted(all_labels)

        self.seq_len = max_len
        self.input_size = 1 # un token per timestep
        self.num_classes = len(self.labels) # 10
        self.vocab_size = len(self.vocab)
        self.input_shape = (batch_size, max_len)

        self.summary()