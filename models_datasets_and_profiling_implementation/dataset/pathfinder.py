import os
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image
from .interface import DatasetInterface


class PathfinderLazy(Dataset):

    def __init__(self, root_data, imgs_dir, metadata_dir, transform):
        self.root_data = root_data
        self.transform = transform
        self.samples = []  # lista di (img_path, label)

        for folder_name in sorted(os.listdir(imgs_dir)):
            folder_path = os.path.join(imgs_dir, folder_name)
            meta_path = os.path.join(metadata_dir, f"{folder_name}.txt")

            if not os.path.isdir(folder_path) or not os.path.exists(meta_path):
                continue

            # lettura metadata (con fallback di encoding)
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
                # parts[0]='imgs/<n>', parts[1]='sample_*.png', parts[3]=label
                img_path = os.path.join(root_data, parts[0], parts[1])
                try:
                    label = int(parts[3])
                except ValueError:
                    continue
                self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                "Nessuna immagine trovata!"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)  # [C, H, W]
        C, H, W = img_tensor.shape
        # sequenza [L, C] = [H*W, C]
        img_tensor = img_tensor.permute(1, 2, 0).contiguous().view(H * W, C)
        return img_tensor, torch.tensor(label, dtype=torch.long)


class LRAPathfinder(DatasetInterface):
    def __init__(self, batch_size, valsplit, num_workers):
        super().__init__("pathfinder", batch_size, num_workers)

        # Percorso base (relativo: lanciare il training dalla cartella S4oP/)
        root_data = "data/pathfinder"
        imgs_dir = os.path.join(root_data, "imgs")
        metadata_dir = os.path.join(root_data, "metadata")

        if not os.path.exists(imgs_dir) or not os.path.exists(metadata_dir):
            raise FileNotFoundError(f"Cartelle imgs/ o metadata/ non trovate in {root_data}")

        # Trasformazioni immagini
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

        # Dataset lazy completo (solo indice in RAM)
        full_dataset = PathfinderLazy(root_data, imgs_dir, metadata_dir, self.transform)

        # Labels ricavate dall'indice, senza decodificare immagini
        self.labels = sorted({label for _, label in full_dataset.samples})

        # Split train/val/test (test = 10%, come nella versione originale)
        N = len(full_dataset)
        val_size = int(valsplit * N)
        test_size = int(0.1 * N)
        train_size = N - val_size - test_size

        self.trainset, self.valset, self.testset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

        # Input shape da un singolo sample (carica una sola immagine)
        sample_x, _ = full_dataset[0]
        self.input_shape = (batch_size,) + tuple(sample_x.shape)  # [B, L, C]

        # Distribuzione classi nel test set, calcolata dall'indice (niente immagini)
        test_labels = torch.tensor(
            [full_dataset.samples[i][1] for i in self.testset.indices]
        )
        print(f"Il numero di sample per ciascuna classe (test): {torch.bincount(test_labels)}")

        print(f"LRAPathfinder (lazy): {len(self.trainset)} train, {len(self.valset)} val, {len(self.testset)} test")
        print(f"Input shape: {self.input_shape}, num_classes={len(self.labels)}")