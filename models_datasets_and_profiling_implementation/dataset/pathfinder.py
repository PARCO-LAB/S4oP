import os
import torch
from torch.utils.data import TensorDataset, random_split
from torchvision import transforms
from PIL import Image
from .interface import DatasetInterface


class LRAPathfinder(DatasetInterface):
    def __init__(self, batch_size, valsplit, num_workers):
        super().__init__("pathfinder", batch_size, num_workers)

        # Percorso base
        root_data = "../pathfinder"
        imgs_dir = os.path.join(root_data, "imgs")
        metadata_dir = os.path.join(root_data, "metadata")

        if not os.path.exists(imgs_dir) or not os.path.exists(metadata_dir):
            raise FileNotFoundError(f"Cartelle imgs/ o metadata/ non trovate in {root_data}")

        # Trasformazioni immagini
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

        img_tensors = []
        labels = []

        for folder_name in sorted(os.listdir(imgs_dir)):
            folder_path = os.path.join(imgs_dir, folder_name)
            meta_path = os.path.join(metadata_dir, f"{folder_name}.txt")  # ora .txt

            if not os.path.isdir(folder_path) or not os.path.exists(meta_path):
                continue

            # Lettura file txt
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
            except UnicodeDecodeError:
                with open(meta_path, "r", encoding="latin-1", errors="ignore") as f:
                    lines = [line.strip() for line in f if line.strip()]

            for line in lines:
                parts = line.split()
                if len(parts) < 4:
                    continue

                # Combina cartella + file
                img_rel_path = os.path.join(parts[0], parts[1])  # es: 'imgs/0/sample_0.png'
                img_path = os.path.join(root_data, img_rel_path)

                try:
                    label = int(parts[3])
                except ValueError:
                    continue

                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path).convert("RGB")
                        img_tensor = self.transform(img)  # [C, H, W]

                        # Trasforma in sequenza [L, C]: [H*W, C]
                        C, H, W = img_tensor.shape
                        img_tensor = img_tensor.permute(1, 2, 0).contiguous().view(H*W, C)

                        img_tensors.append(img_tensor)
                        labels.append(label)
                    except Exception as e:
                        print(f"Immagine saltata: {img_path} ({e})")
                        continue

        if len(img_tensors) == 0:
            raise RuntimeError("Nessuna immagine trovata! Controlla la struttura delle cartelle e i nomi dei file.")

        # Split
        X = torch.stack(img_tensors)  # [N, L, C]
        y = torch.tensor(labels, dtype=torch.long)

        val_size = int(valsplit * len(X))
        test_size = int(0.1 * len(X))
        train_size = len(X) - val_size - test_size

        full_dataset = TensorDataset(X, y)
        self.trainset, self.valset, self.testset = random_split(full_dataset, [train_size, val_size, test_size])

        self.input_shape = (batch_size,) + tuple(X.shape[1:])  # [B, L, C]
        self.labels = sorted(list(set(labels)))
        print(f"Il numero di sample per ciascuna classe: {torch.bincount(self.testset[:][1])}")

        print(f"LRAPathfinder: {len(self.trainset)} train, {len(self.valset)} val, {len(self.testset)} test")
        print(f"Input shape: {self.input_shape}, num_classes={len(self.labels)}")
