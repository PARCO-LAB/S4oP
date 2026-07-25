import torch
from torch.utils.data import Dataset, TensorDataset
import h5py
import numpy as np
import pandas as pd
from .interface import DatasetInterface

class ECGMultiPartLazy(Dataset):
    def __init__(self, df, hdf5_paths, labels_cols):
        self.df = df
        self.labels_cols = labels_cols 
        self._files = None  # aperti lazy, uno per processo/worker
        self.hdf5_paths = list(hdf5_paths)
        # indicizzazione: exam_id → (index_in_file, file_id)
        self.index_map = {}

        # apre i file senza caricare nulla
        for file_idx, path in enumerate(self.hdf5_paths):
            with h5py.File(path, "r") as f:
                ids = np.asarray(f["exam_id"], dtype=int)
            for local_idx, exam_id in enumerate(ids):
                self.index_map[int(exam_id)] = (file_idx, local_idx)

        # prende SOLO gli exam_id che sono sia negli HDF5 che nel CSV
        csv_ids = set(self.df["exam_id"].values)
        h5_ids = set(self.index_map.keys())

        valid_ids = sorted(h5_ids & csv_ids)

        self.df = self.df.set_index("exam_id").loc[valid_ids]

        self.exam_ids = valid_ids

    def _get_files(self):
        if self._files is None:  # apertura ritardata: avviene dentro ogni worker
            self._files = [h5py.File(p, "r") for p in self.hdf5_paths]
        return self._files

    def __len__(self):
        return len(self.exam_ids)

    def __getitem__(self, idx):
        exam_id = self.exam_ids[idx]
        file_idx, local_idx = self.index_map[exam_id]

        f = self._get_files()[file_idx]

        x = f["tracings"][local_idx].astype(np.float32)
        x *= 1000.0  # scaling

        y = self.df.loc[exam_id, self.labels_cols].to_numpy(dtype=np.float32)

        return torch.tensor(x), torch.tensor(y)

class ECGDataset(DatasetInterface):
    def __init__(self, batch_size, valsplit, num_workers, num_parts=5):
        super().__init__("ecg", batch_size, num_workers)

        self.multilabel = True

        labels_cols = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]

        df = pd.read_csv("data/ecg/exams.csv")

        hdf5_paths = [f"data/ecg/exams_part{i}.hdf5" for i in range(num_parts)]

        full_dataset = ECGMultiPartLazy(df, hdf5_paths, labels_cols)

        # train/val split
        N = len(full_dataset)
        val_size = int(N * valsplit)
        train_size = N - val_size

        train_set, val_set = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        self.trainset = train_set
        self.valset = val_set

        with h5py.File("data/ecg/ecg_test.hdf5", "r") as f:
            X_test = torch.tensor(f["tracings"][:], dtype=torch.float) * 1000.0

        df_test = pd.read_csv("data/ecg/gold_standard.csv")
        y_test = torch.tensor(df_test[labels_cols].values, dtype=torch.float)

        self.testset = TensorDataset(X_test, y_test)
        
        self.seq_len = X_test.shape[1] # 4096 
        self.input_size = X_test.shape[2] # 12 derivazioni
        self.num_classes = len(labels_cols) # 6 (multi-label)
        self.vocab_size = None # input continuo -> nn.Linear
        self.input_shape = (batch_size, self.seq_len, self.input_size)

        self.summary(extra=f"pos/classe={y_test.sum(dim=0).tolist()}")