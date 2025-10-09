import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, random_split
from torchvision import transforms
from PIL import Image
from .interface import DatasetInterface

class LRAPathfinder(DatasetInterface):
    def __init__(self, batch_size, valsplit, num_workers):
        super().__init__("pathfinder", batch_size, num_workers)

        root_data = "data/pathfinder"
        imgs_dir = os.path.join(root_data, "imgs")
        metadata_dir = os.path.join(root_data, "metadata")

        # Trasformazioni base: ridimensiona e converte in tensore normalizzato
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        img_tensors = []
        labels = []

        # Scorri tutte le sottocartelle di imgs/
        for folder_name in sorted(os.listdir(imgs_dir)):
            folder_path = os.path.join(imgs_dir, folder_name)
            meta_path = os.path.join(metadata_dir, f"{folder_name}.npy")

            if not os.path.isdir(folder_path) or not os.path.exists(meta_path):
                continue

            # Carica metadati (ogni riga: [idx, label, ...])
            meta = np.load(meta_path)
            if meta.ndim == 1:
                meta = np.expand_dims(meta, axis=0)

            for row in meta:
                idx, label = int(row[0]), int(row[1])
                img_path = os.path.join(folder_path, f"sample_{idx}.png")

                if os.path.exists(img_path):
                    # Apri immagine e applica trasformazioni
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = self.transform(img)
                    img_tensors.append(img_tensor)
                    labels.append(label)

        # Converti in tensori
        X = torch.stack(img_tensors)
        y = torch.tensor(labels, dtype=torch.long)

        # Split train/val/test
        val_size = int(valsplit * len(X))
        test_size = int(0.1 * len(X))  # 10% per test (puoi cambiare)
        train_size = len(X) - val_size - test_size

        full_dataset = TensorDataset(X, y)
        self.trainset, self.valset, self.testset = random_split(full_dataset, [train_size, val_size, test_size])

        # Imposta forma di input
        self.input_shape = (batch_size,) + tuple(X.shape[1:])
        self.labels = sorted(list(set(labels)))

        print(f"LRAPathfinder: {len(self.trainset)} train, {len(self.valset)} val, {len(self.testset)} test")
        print(f"Input shape: {self.input_shape}, num_classes={len(self.labels)}")

