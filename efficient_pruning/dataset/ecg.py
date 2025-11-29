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

        # indicizzazione: exam_id → (index_in_file, file_id)
        self.index_map = {}   # exam_id → (file_idx, local_idx)

        self.files = []  # file h5py aperti

        # apre i file senza caricare nulla
        for file_idx, path in enumerate(hdf5_paths):
            f = h5py.File(path, "r")
            self.files.append(f)

            ids = np.array(f["exam_id"], dtype=int)
            for local_idx, exam_id in enumerate(ids):
                self.index_map[exam_id] = (file_idx, local_idx)

        # prende SOLO gli exam_id che sono sia negli HDF5 che nel CSV
        csv_ids = set(self.df["exam_id"].values)
        h5_ids  = set(self.index_map.keys())

        valid_ids = sorted(h5_ids & csv_ids)

        self.df = self.df.set_index("exam_id").loc[valid_ids]

        self.exam_ids = valid_ids

    def __len__(self):
        return len(self.exam_ids)

    def __getitem__(self, idx):
        exam_id = self.exam_ids[idx]
        file_idx, local_idx = self.index_map[exam_id]

        f = self.files[file_idx]

        x = f["tracings"][local_idx].astype(np.float32)
        x *= 1000.0  # scaling

        y = self.df.loc[exam_id, self.labels_cols].values.astype(np.float32)

        return torch.tensor(x), torch.tensor(y)

class ECGDataset(DatasetInterface):
    def __init__(self, batch_size, valsplit, num_workers, num_parts=5):
        super().__init__("ecg", batch_size, num_workers)

        labels_cols = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]

        df = pd.read_csv("data/ecg/exams.csv")

        hdf5_paths = [
            f"data/ecg/exams_part{i}.hdf5"
            for i in range(num_parts)
        ]

        full_dataset = ECGMultiPartLazy(df, hdf5_paths, labels_cols)

        # train/val split
        N = len(full_dataset)
        val_size = int(N * valsplit)
        train_size = N - val_size

        train_set, val_set = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

        self.trainset = train_set
        self.valset = val_set

        with h5py.File("data/ecg/ecg_test.hdf5", "r") as f:
            X_test = torch.tensor(f["tracings"][:], dtype=torch.float) * 1000.0

        df_test = pd.read_csv("data/ecg/gold_standard.csv")
        y_test = torch.tensor(df_test[labels_cols].values, dtype=torch.float)

        self.testset = TensorDataset(X_test, y_test)
        print(f"ECG Sample per classe: {torch.sum(y_test, dim=0)}")

        self.labels = labels_cols
        self.input_shape = (batch_size, 4096, 12)

        print(f"ECG: {len(self.trainset)} train, {len(self.valset)} val, {len(self.testset)} test, seq_len={4096}, num_leads=12")

""" import torch
from torch.utils.data import TensorDataset
import h5py
import numpy as np
import pandas as pd
from .interface import DatasetInterface


class ECGDataset(DatasetInterface):
    def __init__(self, batch_size, valsplit, num_workers):
        super().__init__("ecg", batch_size, num_workers)

        # --- CARICO HDF5 ---
        with h5py.File("data/ecg/exams_part0.hdf5", "r") as f:
            tracings = f["tracings"]
            exam_ids = np.array(f["exam_id"], dtype=int)  # attenzione ai valori zero

            N = tracings.shape[0]

            X_trainval = np.array(tracings, dtype=np.float32)

        # --- SCALING ---
        X_trainval *= 1000.0
        X_trainval = torch.tensor(X_trainval, dtype=torch.float)

        # --- CARICO CSV ---
        df = pd.read_csv("data/ecg/exams.csv")
        labels_cols = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]

        # solo le righe di questo file
        df_part0 = df[df["trace_file"] == "exams_part0.hdf5"]

        # prendo solo gli exam_id che esistono nel CSV
        valid_ids = df_part0["exam_id"].values
        mask = np.isin(exam_ids, valid_ids)
        exam_ids_filtered = exam_ids[mask]
        X_trainval = X_trainval[mask]  # mantieni allineamento tra tracings e labels

        # ora pandas non sbaglia più
        df_part0 = df_part0.set_index("exam_id").loc[exam_ids_filtered]
        y_trainval = torch.tensor(df_part0[labels_cols].values, dtype=torch.float)

        # --- TRAIN/VAL SPLIT ---
        N = len(exam_ids_filtered)
        val_size = int(valsplit * N)
        train_size = N - val_size

        indices = torch.randperm(N)
        train_idx, val_idx = indices[:train_size], indices[train_size:]

        self.trainset = TensorDataset(X_trainval[train_idx], y_trainval[train_idx])
        self.valset   = TensorDataset(X_trainval[val_idx],   y_trainval[val_idx])

        # --- TEST SET ---
        with h5py.File("data/ecg/ecg_test.hdf5", "r") as f:
            X_test = np.array(f["tracings"], dtype=np.float32)

        X_test *= 1000.0
        X_test = torch.tensor(X_test, dtype=torch.float)

        df_test = pd.read_csv("data/ecg/gold_standard.csv")
        y_test = torch.tensor(df_test[labels_cols].values, dtype=torch.float)

        self.testset = TensorDataset(X_test, y_test)

        self.labels = labels_cols
        self.input_shape = (batch_size, 4096, 12)

        print(f"[TNMG] Train: {len(self.trainset)}, Val: {len(self.valset)}")
        print(f"[ECG]  Test:  {len(self.testset)}") """

""" import torch
from torch.utils.data import TensorDataset
import h5py
import numpy as np
import pandas as pd
from .interface import DatasetInterface


class ECGDataset(DatasetInterface):
    def __init__(self, batch_size, valsplit, num_workers, num_parts=4):
        super().__init__("ecg", batch_size, num_workers)

        labels_cols = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]

        all_X = []
        all_ids = []

        df = pd.read_csv("data/ecg/exams.csv")

        for i in range(num_parts):
            path = f"data/ecg/exams_part{i}.hdf5"
            print(f"[ECG] Carico {path} ...")

            with h5py.File(path, "r") as f:
                tracings = np.array(f["tracings"], dtype=np.float32)
                exam_ids = np.array(f["exam_id"], dtype=int)

            # scaling
            tracings *= 1000.0

            all_X.append(tracings)
            all_ids.append(exam_ids)

        # concateno tutti i pezzi
        X_trainval = np.concatenate(all_X, axis=0)
        exam_ids = np.concatenate(all_ids, axis=0)

        X_trainval = torch.tensor(X_trainval, dtype=torch.float)

        # filtro solo gli exam_id presenti nel csv
        valid_df = df[df["trace_file"].isin(
            [f"exams_part{i}.hdf5" for i in range(num_parts)]
        )]

        # match exam_id con tracings
        mask = np.isin(exam_ids, valid_df["exam_id"].values)
        exam_ids_filtered = exam_ids[mask]
        X_trainval = X_trainval[mask]

        # ordino il dataframe nel giusto ordine
        print("[ECG] Allineo labels...")
        valid_df = valid_df.set_index("exam_id").loc[exam_ids_filtered]
        y_trainval = torch.tensor(valid_df[labels_cols].values, dtype=torch.float)

        # train/val split
        N = len(exam_ids_filtered)
        val_size = int(valsplit * N)
        train_size = N - val_size

        indices = torch.randperm(N)
        train_idx, val_idx = indices[:train_size], indices[train_size:]

        self.trainset = TensorDataset(X_trainval[train_idx], y_trainval[train_idx])
        self.valset   = TensorDataset(X_trainval[val_idx],   y_trainval[val_idx])

        # test set
        with h5py.File("data/ecg/ecg_test.hdf5", "r") as f:
            X_test = np.array(f["tracings"], dtype=np.float32)

        X_test *= 1000.0
        X_test = torch.tensor(X_test, dtype=torch.float)

        df_test = pd.read_csv("data/ecg/gold_standard.csv")
        y_test = torch.tensor(df_test[labels_cols].values, dtype=torch.float)

        self.testset = TensorDataset(X_test, y_test)

        self.labels = labels_cols
        self.input_shape = (batch_size, 4096, 12)

        print(f"[TNMG] Train: {len(self.trainset)}, Val: {len(self.valset)}")
        print(f"[ECG]  Test:  {len(self.testset)}")
 """