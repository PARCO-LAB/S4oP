import torch
import numpy as np


class DatasetInterface: 
    def __init__(self, name, batch_size, num_workers=1, input_shape=(512, 512)): 
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_shape = input_shape
        self.trainset = None
        self.valset = None
        self.testset = None


    @staticmethod
    def get_loader(dataset, batch_size, num_workers, shuffle, subset_perc):
        if dataset is None: 
            raise ValueError("Unexpected get_loader 'dataset' argument at None")
        if subset_perc is not None: 
            if subset_perc >= 1: 
                amount_samples = int(subset_perc)
            else: 
                amount_samples = int(len(dataset) * subset_perc)
            indices = np.random.choice(len(dataset), amount_samples, replace=False)
            dataset = torch.utils.data.Subset(dataset, indices)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


    def get_input_shape(self):
        dataiter = iter(self.get_trainloader())
        inputs, _ = next(dataiter)
        input_shape = (1, *inputs.size()[1:])
        return input_shape
    

    def get_output_shape(self):
        output_shape = (1, len(self.labels))
        return output_shape


    def get_trainloader(self, shuffle=True, subset_perc=None):
        return DatasetInterface.get_loader(self.trainset, self.batch_size, self.num_workers, shuffle, subset_perc)


    def get_valloader(self, shuffle=False, subset_perc=None):
        return DatasetInterface.get_loader(self.valset, self.batch_size, self.num_workers, shuffle, subset_perc)


    def get_testloader(self, shuffle=False, subset_perc=None):
        return DatasetInterface.get_loader(self.testset, self.batch_size, self.num_workers, shuffle, subset_perc)
