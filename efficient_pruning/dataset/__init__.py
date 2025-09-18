from .cifar import *
from .mnist import *
from .imagenet import *

__all__ = {
    "cifar10": Cifar10,
    "cifar100": Cifar100,
    "mnist": Mnist,
    "fashionmnist": FashionMnist,
    "imagenet": ImageNet,
}

class DatasetFactory: 
    def __init__(self, dataset_name, batch_size=128, valsplit=0.2, transform=None, num_workers=1, input_shape=None):
        if dataset_name not in __all__:
            raise NotImplementedError("dataset_name {} not recognized".format(dataset_name)) 
        self.dataset = __all__[dataset_name](
            batch_size=batch_size, valsplit=valsplit, transform=transform, num_workers=num_workers, input_shape=input_shape,
        )
        
    def get_dataset(self): 
        return self.dataset