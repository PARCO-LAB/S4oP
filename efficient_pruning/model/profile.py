import os
import json
import pandas as pd
from ..perf import ModelTime, ModelNsight
from .utils import set_benchmark


class ModelProfile: 
    def __init__(self, iterations=100):
        self.iterations = iterations
        self.model_time = ModelTime(self.iterations)
        self.model_nsight = ModelNsight()
        self.tests = {}

    def get_state(self):
        state = {}
        state["time"] = self.model_time.get_state()
        state["nsight"] = self.model_nsight.get_state()
        state["tests"] = self.tests
        return state
    
    def append_state(self, new_state):
        if "time" in new_state and new_state["time"]: 
            self.model_time.append_state(new_state["time"])
        if "nsight" in new_state and new_state["nsight"]: 
            self.model_nsight.append_state(new_state["nsight"])
        if "tests" in new_state and new_state["tests"]: 
            self.tests = new_state["tests"]


    def add(self, name):
        self.model_time.add(name)
        self.model_nsight.add(name)

    def add_test(self, name, data):
        self.tests[name] = data

    def run(self, name, model, get_example_input, iterations=None):
        model.eval()
        self.model_time.run(name, model, get_example_input, iterations)
        self.model_nsight.run(name, model, get_example_input)

    def _info_test(self, name=None):
        print("============ ModelTest ============")
        data_names = self.tests if name is None else [name]
        for name in data_names:
            if name not in self.tests: 
                continue
            print("[[ {} ]]".format(name))
            for key, value in self.tests[name].items():
                print("[{}] : {}%".format(key, value))
        print("===================================")

    def info(self, name=None):
        self.model_time.info(name)
        self.model_nsight.info(name)
        self._info_test(name)

    def _dump_test(self, filename):
        with open("{}.json".format(filename), "w") as json_file:
            json.dump(self.tests, json_file, indent=4)

    def dump(self, filename):
        self.model_time.dump("{}_time".format(filename))
        self.model_nsight.dump("{}_nsight".format(filename))
        self._dump_test("{}_tests".format(filename))

        
    def _dataframe_test(self):
        df = pd.DataFrame()
        for test_name in self.tests:
            for key in self.tests[test_name]:
                curr_df = pd.DataFrame({
                    "Test": [test_name], 
                    "Section": ["Tests"],
                    "Name": [key],
                    "Metric": ["Accuracy"],
                    "Unit": ["%"],
                    "Value": [
                        self.tests[test_name][key],
                    ]
                })

                df = pd.concat([
                    df, 
                    curr_df
                ], axis=0).reset_index(drop=True)
        return df

    
    def dataframe(self):
        df_time = self.model_time.dataframe()
        df_nsight = self.model_nsight.dataframe()
        df_tests = self._dataframe_test()

        return pd.concat([
            df_time,
            df_nsight,
            df_tests,
        ], axis=0).sort_values(by=["Test", "Section", "Metric"]).reset_index(drop=True)
