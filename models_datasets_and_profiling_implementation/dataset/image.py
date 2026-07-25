import pickle
import torch
import numpy as np
from torch.utils.data import TensorDataset
from .interface import DatasetInterface

class ImageDataset(DatasetInterface):
    def __init__(self, batch_size, valsplit, num_workers):
        super().__init__("image", batch_size, num_workers)

        def load_split(split):
            with open(f"data/image/lra-image.{split}.pickle", "rb") as f:
                data = pickle.load(f)
            X = torch.from_numpy(np.stack([s["input_ids_0"] for s in data])).float()  # (N, 1024)  
            X = (X / 255.0).unsqueeze(-1) # (N, 1024, 1) in [0,1]
            y = torch.from_numpy(np.array([s["label"] for s in data], dtype=np.int64))
            return TensorDataset(X, y)

        self.trainset = load_split("train")
        self.valset = load_split("dev")
        self.testset = load_split("test")

        y_train = self.trainset.tensors[1]
        self.labels = list(range(int(y_train.max().item()) + 1))
        self.num_classes = len(self.labels)
        self.seq_len = self.trainset.tensors[0].shape[1] # 1024
        self.input_size = self.trainset.tensors[0].shape[2]  # 1 (pixel scalare)
        self.vocab_size = None # continuo -> nn.Linear
        self.input_shape = (batch_size, self.seq_len, self.input_size)

        self.summary()