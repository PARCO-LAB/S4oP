import json
import os
import torch
import torch.nn as nn
import torch.multiprocessing as multiprocessing
from multiprocessing import queues

from ...model.profile import ModelProfileSelection
from ...model.net.s4 import LayerS4D, S4Block, DropoutNd
from ...model.utils import get_device, singleton, clean_memory, set_benchmark, set_seed

def _get_s4d_attributes(layer):
    return {
        "d_model": layer.h,
        "d_state": layer.n,
        "dropout": layer.dropout.p if isinstance(layer.dropout, DropoutNd) else 0.0,
        "extra": (layer.kernel.C, layer.kernel.log_dt, layer.kernel.log_A_real, layer.kernel.A_imag)
    }

def _create_s4d_from_attributes(attr):
    return LayerS4D(
        d_model=attr["d_model"],
        d_state=attr["d_state"],
        dropout=attr["dropout"]
    ).to(get_device())

def _get_s4_attributes(layer):
    return {
        "d_model": layer.d_model,
        "d_state": layer.layer.kernel.d_state,
        "dropout": 0.0 if isinstance(layer.drop, nn.Identity) else layer.drop.p,
        "extra": layer.layer.kernel._get_params(),
    }

def _create_s4_from_attributes(attr):
    return S4Block(
        d_model=attr["d_model"],
        d_state=attr["d_state"],
        dropout=attr["dropout"]
    ).to(get_device())

def _get_attributes(layer):
    if isinstance(layer, LayerS4D):
        target_type = "S4DLayer"
        attr = _get_s4d_attributes(layer)
    elif isinstance(layer, S4Block):
        target_type = "S4Block"
        attr = _get_s4_attributes(layer)
    else:
        raise Exception(f"Tipo di layer non riconosciuto: {type(layer)}")

    return target_type, attr

def _create_from_attributes(target_type, attr):
    if target_type == "S4Block":
        return _create_s4_from_attributes(attr)
    elif target_type == "S4DLayer":
        return _create_s4d_from_attributes(attr)
    else:
        raise Exception(f"Tipo di layer non riconosciuto: {target_type}")

def _profile_task(data_queue, result_queue, stop_event, seq_len, batch_size):
    set_seed(42)
    set_benchmark(False)

    while not stop_event.is_set():
        try:
            config_list, model_profile = data_queue.get(timeout=0.1)
            results = []

            for config in config_list:
                target_type, target_name, new_d_model, base_attr = config

                # Costruisci il nuovo layer con H prunato
                base_attr["d_model"] = new_d_model

                module = _create_from_attributes(target_type, base_attr)

                # Input coerente con dataset
                example_input = lambda _: torch.randn(
                    batch_size, new_d_model, seq_len, device=get_device()
                )

                # Profiling
                model_profile.add("tmp")
                model_profile.run("tmp", module, example_input)

                data = model_profile.data["tmp"]["tot"]
                if "tot" in data:
                    data = data["tot"]
                else:
                    data = list(data.values())[0]

                results.append(data)

            result_queue.put(results)
            clean_memory()

        except queues.Empty:
            pass

@singleton
class PruningProfiler:

    DEFAULT_ITERATIONS = 10
    DEFAULT_MODEL_PROFILE_CLASS = lambda iterations: ModelProfileSelection("time", iterations)

    def __init__(
        self,
        iterations=DEFAULT_ITERATIONS,
        model_profile_class=DEFAULT_MODEL_PROFILE_CLASS,
        backup_file=None,
        seq_len=1024,
        batch_size=1
    ):
        self.iterations = iterations
        self.model_profile = model_profile_class(iterations)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.profiling_configurations = {}
        self.one_shot_profile = True
        self.started = False
        self.set_backup_file(backup_file)

    def set_backup_file(self, backup_file):
        self.backup_file = backup_file
        if self.backup_file and os.path.exists(self.backup_file):
            self.load(self.backup_file)

    def save(self, output_path):
        dump = [{"key": k, "value": v} for k, v in self.profiling_configurations.items()]
        with open(output_path, "w") as f:
            json.dump(dump, f, indent=4)

    def load(self, input_path):
        with open(input_path, "r") as f:
            data = json.load(f)
        for e in data:
            self.profiling_configurations[tuple(e["key"])] = e["value"]

    def start(self):
        self.started = True
        ctx = multiprocessing.get_context("spawn")
        self.data_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        self.stop_event = ctx.Event()
        self.worker = ctx.Process(target=_profile_task, args=(self.data_queue, self.result_queue, self.stop_event, self.seq_len, self.batch_size))
        self.worker.start()

    def stop(self):
        if not self.started:
            return
        self.stop_event.set()
        self.worker.join()
        self.worker.terminate()
        self.worker.close()
        self.data_queue.close()
        self.result_queue.close()
        clean_memory()
        self.started = False

    def config_to_profile_key(self, config):
        target_type, target_name, new_d_model, attr = config
        if self.one_shot_profile:
            return (target_name, new_d_model)
        else:
            return config

    def run(self, config_list):
        config_list_todo = []
        for config in config_list:
            if self.config_to_profile_key(config) not in self.profiling_configurations:
                config_list_todo.append(config)

        if not config_list_todo:
            return [self.profiling_configurations[self.config_to_profile_key(c)] for c in config_list]

        if not self.started:
            self.start()

        print("P4P: {}".format(config_list_todo[0]))
        self.data_queue.put((config_list_todo, self.model_profile))
        profile_data_list = self.result_queue.get()

        for config, data in zip(config_list_todo, profile_data_list):
            key = self.config_to_profile_key(config)
            self.profiling_configurations[key] = data

        if self.backup_file:
            self.save(self.backup_file)

        return [self.profiling_configurations[self.config_to_profile_key(c)] for c in config_list]

    def get_config(self, layer):
        target_type, attr = _get_attributes(layer)
        target_name = getattr(layer, "name", "unnamed_layer")
        d_model = attr["d_model"]

        config_list = []
        for new_d_model in range(d_model, 0, -1):
            config_list.append((target_type, target_name, new_d_model, attr))

        return config_list