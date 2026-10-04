import torch

class DatasetInterface: 
    def __init__(self, name, batch_size, num_workers): 
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_shape = None
        self.vocab_size = None
        self.seq_len = None
        self.input_size = None
        self.metric = None
        self.num_classes = None
        self.trainset = None
        self.valset = None
        self.testset = None

        self.multilabel = False # task multi-label -> BCEWithLogitsLoss invece di CrossEntropy
        self.dual_stream = False # input a due documenti [B,2,L] -> testa "match"

    def summary(self, extra=None):
        kind = (f"vocab_size={self.vocab_size}" if self.vocab_size is not None else f"input_size={self.input_size}")
        line = (f"[{self.name.upper()}] "
                f"train={len(self.trainset)} val={len(self.valset)} test={len(self.testset)} "
                f"| seq_len={self.seq_len} | {kind} | num_classes={self.num_classes}")
        if extra:
            line += f" | {extra}"
        print(line)

    @staticmethod
    def get_loader(dataset, batch_size, num_workers, shuffle):
        if dataset is None: 
            raise ValueError("Unexpected get_loader: 'dataset' argument at None")
        else:
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    def get_trainloader(self, shuffle=True):
        return DatasetInterface.get_loader(self.trainset, self.batch_size, self.num_workers, shuffle)

    def get_valloader(self, shuffle=False):
        return DatasetInterface.get_loader(self.valset, self.batch_size, self.num_workers, shuffle)

    def get_testloader(self, shuffle=False):
        return DatasetInterface.get_loader(self.testset, self.batch_size, self.num_workers, shuffle)