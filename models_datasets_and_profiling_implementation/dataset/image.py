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
            X = (X / 255.0).unsqueeze(-1) 
            y = torch.from_numpy(np.array([s["label"] for s in data], dtype=np.int64))
            return TensorDataset(X, y)

        self.trainset = load_split("train")
        self.valset = load_split("dev")
        self.testset = load_split("test")

        self.labels = list(range(10))
        self.input_shape = (batch_size, 1024, 1)

        print(f"IMAGE: {len(self.trainset)} train, {len(self.valset)} val, "
              f"{len(self.testset)} test, seq_len=1024, classes=10")