import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from .interface import DatasetInterface

class RetrievalPickleLazy(Dataset):
    def __init__(self, pickle_path):
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        # due documenti per campione -> (N, 2, seq_len), fork-safe (array in RAM)
        self.X = np.stack(
            [np.stack([s["input_ids_0"], s["input_ids_1"]]) for s in data]
        ).astype(np.int16) # token 0..127 -> int16 basta
        self.y = np.array([int(s["label"]) for s in data], dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx].astype(np.int64))   # long per l'embedding
        return x, torch.tensor(self.y[idx])

class RetrievalDataset(DatasetInterface):
    def __init__(self, batch_size, valsplit, num_workers):
        super().__init__("retrieval", batch_size, num_workers)

        self.dual_stream = True
        
        base = "data/retrieval/lra-retrieval.{}.pickle"
        self.trainset = RetrievalPickleLazy(base.format("train"))
        self.valset = RetrievalPickleLazy(base.format("dev"))
        self.testset = RetrievalPickleLazy(base.format("test"))

        self.labels = sorted(set(self.trainset.y.tolist()))
        self.seq_len = self.trainset.X.shape[-1] # per-documento (derivato)
        self.input_size = 1 # un token per timestep
        self.num_classes = len(self.labels) # 2
        self.vocab_size = 256 # max osservato 127
        self.input_shape = (batch_size, 2, self.seq_len) # 2 = dual-stream

        self.summary(extra="dual-stream (2 doc/campione)")