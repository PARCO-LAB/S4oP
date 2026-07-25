import json
import torch
from multiprocessing import queues
import pandas as pd

from ..perf import ModelTime, MemoryProfile
from .utils import set_benchmark, clean_memory, get_device, set_seed


class ModelProfile: 
    def __init__(self, iterations=100):
        self.iterations = iterations
        self.model_time = ModelTime(self.iterations)
        self.memory_profile = MemoryProfile()
        self.tests = {}

    def get_state(self):
        state = {}
        state["time"] = self.model_time.get_state()
        state["tests"] = self.tests
        return state
    
    def append_state(self, new_state):
        if "time" in new_state and new_state["time"]: 
            self.model_time.append_state(new_state["time"])
        if "tests" in new_state and new_state["tests"]: 
            self.tests = new_state["tests"]


    def add(self, name):
        self.model_time.add(name)
        self.memory_profile.add(name)

    def add_test(self, name, data):
        self.tests[name] = data

    def run(self, name, model, get_example_input, iterations=None):
        model.eval()
        self.model_time.run(name, model, get_example_input, iterations)
        self.memory_profile.run(name, model, get_example_input)

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
        self.memory_profile.info(name)
        #self._info_test(name)

    def _dump_test(self, filename):
        with open("{}.json".format(filename), "w") as json_file:
            json.dump(self.tests, json_file, indent=4)

    def dump(self, filename):
        self.model_time.dump("{}_time".format(filename))
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
        df_tests = self._dataframe_test()
        df_memory = self.memory_profile.dataframe()

        return pd.concat([
            df_time,
            df_tests,
            df_memory,
        ], axis=0).sort_values(by=["Test", "Section", "Metric"]).reset_index(drop=True)

def _get_model_profile_by_name(model_profile_name, iterations=100):
    if model_profile_name == "time": 
        return ModelProfile(iterations)
    else: 
        raise Exception("Not recognized model profile name {}".format(model_profile_name))

def _task_run(model_profile_info, data_queue, result_queue, stop_event):
    set_seed(42)
    set_benchmark(False)
    model_profile_name, iterations = model_profile_info
    model_profile = _get_model_profile_by_name(model_profile_name, iterations)
    while not stop_event.is_set():
        try:
            name, model, example_input_shape, custom_iterations = data_queue.get(timeout=0.1)
            model_profile.add(name)
            model_profile.run(name, model.to(get_device()), lambda m=None: torch.randn(*example_input_shape if m is None else m.input_shape).to(get_device()), custom_iterations)
            result_queue.put(model_profile.get_state())
            clean_memory()
        except queues.Empty:
            pass