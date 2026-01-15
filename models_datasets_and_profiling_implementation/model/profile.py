import json
import torch
import torch.multiprocessing as multiprocessing
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
    
class ModelProfileSelection(ModelProfile):
    def __init__(self, selection, iterations=100):
        super(ModelProfileSelection, self).__init__(iterations=iterations)
        self.selection = selection
        self.profile_selection = {
            "time": self.model_time,
        }

        if selection not in self.profile_selection: 
            raise Exception("Not recognized selection field: {}".format(selection))
        
        self.model_profile_selected = self.profile_selection[selection]


    def get_state(self):
        return {self.selection: self.model_profile_selected.get_state()}
    

    def append_state(self, new_state):
        if self.selection in new_state: 
            self.model_profile_selected.append_state(new_state[self.selection])


    @property
    def data(self):
        return self.model_profile_selected.data


    def add(self, name):
        self.model_profile_selected.add(name)


    def run(self, name, model, get_example_input, iterations=None):
        if hasattr(model, "eval"):
            model.eval()

        if self.selection in ["time"]:
            self.model_profile_selected.run(name, model, get_example_input, iterations)
        else:
            raise Exception("Not recognized selection field: {}".format(self.selection))

    def info(self, name=None):
        self.model_profile_selected.info(name)

    
    def dump(self, filename):
        self.model_profile_selected.dump("{}_{}".format(filename, self.selection))

    
    def dataframe(self):
        return self.model_profile_selected.dataframe().reset_index(drop=True)

class ModelMultiprocessing:
    def __init__(self, model_profile_name, iterations):
        self.model_profile_name = model_profile_name
        self.iterations = iterations
        self.model_profile = _get_model_profile_by_name(self.model_profile_name, self.iterations)
        self.started = False

    def start(self):
        self.started = True
        ctx = multiprocessing.get_context('spawn')

        self.data_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        self.stop_event = ctx.Event()

        # Start the worker process
        model_profile_info = (self.model_profile_name, self.iterations)
        self.worker = ctx.Process(
            target=_task_run, 
            args=(model_profile_info, self.data_queue, self.result_queue, self.stop_event))
        self.worker.start()

    @property
    def data(self):
        return self.model_profile.data
    
    def add(self, name):
        return self.model_profile.add(name)
    
    def add_test(self, name, data):
        return self.model_profile.add_test(name, data)
    
    def run(self, name, model, get_example_input, iterations=None):
        if not self.started: 
            raise Exception("Error: model profile task not started")
        model.cpu()
        self.data_queue.put((name, model, get_example_input().shape, iterations))
        # clean_memory()
        model_profile_state = self.result_queue.get()
        self.model_profile.append_state(model_profile_state)
        model.to(get_device())

    def stop(self):
        self.started = False
        self.stop_event.set()
        self.worker.join()
        self.worker.terminate()
        self.worker.close()
        self.data_queue.close()
        self.result_queue.close()
        # self.stop_event.close()
        clean_memory()

    def info(self, name=None):
        return self.model_profile.info(name)

    def dump(self, filename):
        return self.model_profile.dump(filename)

    def dataframe(self):
        return self.model_profile.dataframe()

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