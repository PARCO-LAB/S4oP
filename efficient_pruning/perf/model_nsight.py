import torch
import pandas as pd

class ModelNsight: 
    def __init__(self):
        pass

    def get_state(self):
        return 
    
    def append_state(self, new_data):
        pass
    
    @torch.no_grad()
    def run(self, name, model, get_example_input): 

        torch.cuda.nvtx.range_push(name)
        torch.cuda.cudart().cudaProfilerStart()
        
        output = model(get_example_input())
        output

        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.nvtx.range_pop()

    def add(self, name):
        pass

    def info(self, name=None):
        print("=========== ModelNsight ===========")
        print("===================================")    

    def dump(self, filename):
        pass

    def dataframe(self):
        df = pd.DataFrame()
        return df