from .listops import *
from .imdb import *
from .pathfinder import *
from .ecg import *
from .image import *
from .retrieval import *
from .genomics import *

__all__ = {
    "listops": LRAListOps,
    "imdb": IMDB,
    "pathfinder": LRAPathfinder,
    "ecg": ECGDataset,
    "image": ImageDataset,
    "retrieval": RetrievalDataset,
    "promoter": LRBPromoter,
    "enhancer": LRBEnhancer,
    "histone": LRBHistone,
    "dnase": LRBDnase,
}

class DatasetFactory: 
    def __init__(self, dataset_name, batch_size, valsplit, num_workers):
        if dataset_name not in __all__:
            raise NotImplementedError("dataset_name {} not recognized".format(dataset_name)) 
        self.dataset = __all__[dataset_name](batch_size=batch_size, valsplit=valsplit, num_workers=num_workers)
        
    def get_dataset(self): 
        return self.dataset