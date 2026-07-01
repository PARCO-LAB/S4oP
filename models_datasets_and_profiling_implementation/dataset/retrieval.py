import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from .interface import DatasetInterface

class RetrievalPickleLazy(Dataset):
    def __init__(self, pickle_path):
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        # due documenti per campione -> (N, 2, seq_len), fork-safe
        self.X = np.stack(
            [np.stack([s["input_ids_0"], s["input_ids_1"]]) for s in data]
        ).astype(np.int16)                                    # token 0..127 entrano in int16
        self.y = np.array([int(s["label"]) for s in data], dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx].astype(np.int64))    # cast a long per l'embedding
        return x, torch.tensor(self.y[idx])

class RetrievalDataset(DatasetInterface):
    def __init__(self, batch_size, valsplit, num_workers):
        super().__init__("retrieval", batch_size, num_workers)
        base = "data/retrieval/lra-retrieval.{}.pickle"
        self.trainset = RetrievalPickleLazy(base.format("train"))
        self.valset = RetrievalPickleLazy(base.format("dev"))
        self.testset = RetrievalPickleLazy(base.format("test"))
        self.vocab_size = 256          # sicuro (max osservato 127)
        self.labels = list(range(2))   # 2 classi -> get_output_shape() -> 2
        self.input_shape = (batch_size, 2, 4096)
        print(f"RETRIEVAL: {len(self.trainset)} train, {len(self.valset)} val, "
              f"{len(self.testset)} test, seq_len=4096, vocab=256, classes=2, dual-stream")