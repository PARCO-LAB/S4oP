import torch
import json
import os
import pandas as pd
import numpy as np

from .time_profile import TimeProfile
from ..model.utils import get_device

import torch.utils.benchmark as benchmark


class ModelTime: 
    def __init__(self, iterations=100):
        self.data = {}
        self.time_profiles = {}
        self.iterations = iterations

    
    def add(self, name):
        self.data[name] = {}
        self.time_profiles[name] = TimeProfile(self.iterations)


    def get_state(self):
        return self.data
    
    
    def append_state(self, new_data):
        for name in new_data: 
            self.data[name] = new_data[name]
            self.time_profiles[name].data = new_data[name]


    @torch.no_grad()
    def run(self, name, model, get_example_input, iterations=None): 
        if iterations is None: 
            iterations = self.iterations
        example_input = get_example_input()

        m = lambda example_input: model(example_input)

        self.time_profiles[name].add("tot")
        self.time_profiles[name].run("tot", lambda: m(example_input), iterations)
        
        self.data[name] = self.time_profiles[name].data

        def model_forward():
            with torch.no_grad():
                return model(example_input)

        timer = benchmark.Timer(
            stmt='model_forward()',
            globals={'model_forward': model_forward},
            num_threads=1,
        )
        measure = timer.timeit(iterations)

        self.data[name]["tot"]["torch_tot"] = np.sum(measure.times) * 1e9
        self.data[name]["tot"]["torch_mean"] = measure.mean * 1e9
        self.data[name]["tot"]["torch_std"] = np.std(measure.times) * 1e9
        self.data[name]["tot"]["torch_median"] = measure.median * 1e9

    def info(self, name=None):
        print("===========  ModelTime  ===========")
        data_names = self.data if name is None else [name]
        for name in data_names:
            print("[[ {} ]]".format(name))
            self.time_profiles[name].info(header=False)
        print("===================================")
    

    def dump(self, filename):
        with open("{}.json".format(filename), "w") as json_file:
            json.dump(self.data, json_file, indent=4)


    def dataframe(self):
        df_tot = pd.DataFrame()
        for name in self.time_profiles:
            df = self.time_profiles[name].dataframe()
            for data_name in self.data[name]:
                curr_df = pd.DataFrame({
                    "Section": ["Time"] * 4,
                    "Name": [data_name] * 4,
                    "Metric": [
                        "TorchTot", 
                        "TorchMean", 
                        "TorchMedian", 
                        "TorchStd", 
                        # "TorchProfCPU", 
                        # "TorchProfGPU",
                    ],
                    "Unit": ["ns"] * 4,
                    "Value": [
                        self.data[name][data_name]["torch_tot"],
                        self.data[name][data_name]["torch_mean"],
                        self.data[name][data_name]["torch_median"],
                        self.data[name][data_name]["torch_std"],
                        # self.data[name][data_name]["torch_prof_cuda_time"],
                        # self.data[name][data_name]["torch_prof_cpu_time"],
                    ]
                })

                df = pd.concat([
                    df, 
                    curr_df
                ], axis=0).reset_index(drop=True)

            df = pd.concat([pd.DataFrame({"Test": [name] * len(df.index)}), df], axis=1)
            df_tot = pd.concat([df_tot, df], axis=0).reset_index(drop=True)
        return df_tot