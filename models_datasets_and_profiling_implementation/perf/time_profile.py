import time
import json
import numpy as np
import pandas as pd
import torch


class TimeProfile: 
    def __init__(self, iterations=100):
        self.iterations = iterations
        self.data = {}

    
    @staticmethod
    def _get_time():
        return time.time() * 1e9
    

    @staticmethod
    def pretty_time(time_ns):
        if time_ns < 1e3: 
            return "{} ns".format(int(time_ns))
        elif time_ns < 1e6:
            return "{} us".format(time_ns/1e3)
        elif time_ns < 1e9:
            return "{} ms".format(time_ns/1e6)
        elif time_ns < 1e12:
            return "{} sec".format(time_ns/1e9)
        else:
            return "{} min".format(time_ns/(1e9 * 60))


    def add(self, name):
        self.data[name] = {
            "time_vec": None,
            "tot": None,
            "mean": None,
            "median": None,
            "std": None
        }


    def run(self, name, function, iterations=None):
        if iterations is None:
            iterations = self.iterations
        use_cuda = torch.cuda.is_available()
        timings = np.zeros(iterations)

        for _ in range(50):  # warmup
            ret = function()

        if use_cuda:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            for i in range(iterations):
                starter.record()
                ret = function()
                ender.record()
                torch.cuda.synchronize()
                timings[i] = starter.elapsed_time(ender) * 1e6   # ms -> ns
        else:
            for i in range(iterations):
                t0 = time.perf_counter_ns()
                ret = function()
                timings[i] = time.perf_counter_ns() - t0

        self.data[name]["tot"] = np.sum(timings)
        self.data[name]["mean"] = np.mean(timings)
        self.data[name]["median"] = np.median(timings)
        self.data[name]["std"] = np.std(timings)
        self.data[name]["time_vec"] = timings.tolist()
        return ret

    def info(self, name=None, header=True):
        if header: 
            print("=========== TimeProfile ===========")
        data_names = self.data if name is None else [name]
        for name in data_names:
            for data_type in self.data[name]:
                if data_type == "time_vec":
                    continue
                print("[{}] {}: {}".format(name, data_type, TimeProfile.pretty_time(self.data[name][data_type])))
        if header: 
            print("===================================")


    def dump(self, filename):
        with open("{}.json".format(filename), "w") as json_file:
            json.dump(self.data, json_file, indent=4)


    def dataframe(self):
        df = pd.DataFrame()
        for name in self.data:
            curr_df = pd.DataFrame({
                "Section": ["Time"] * 2,
                "Name": [name] * 2,
                "Metric": [
                    "Tot", 
                    "Mean", 
                    # "Median", 
                    # "Std",
                ],
                "Unit": ["ns"] * 2,
                "Value": [
                    self.data[name]["tot"],
                    self.data[name]["mean"],
                    # self.data[name]["median"],
                    # self.data[name]["std"],
                ]
            })

            df = pd.concat([
                df, 
                curr_df
            ], axis=0).reset_index(drop=True)
        return df
            

