import os
import json

import torch
import torchinfo
from torchsummary import summary
from torchviz import make_dot

from . import utils
from net.s4.layer_s4d import LayerS4D, S4DKernel

def _attributes_linear(layer):
    return {
        "in_features": layer.in_features,
        "out_features": layer.out_features,
    }

def _attributes_batchnorm(layer):
    return {
        "num_features": layer.num_features,
        "eps": layer.eps,
        "momentum": layer.momentum,
        "affine": layer.affine,
        "track_running_stats": layer.track_running_stats,
    }

def _attributes_layernorm(layer):
    return {
        "normalized_shape": layer.normalized_shape,
        "eps": layer.eps,
        "elementwise_affine": layer.elementwise_affine,
    }

def _attributes_embedding(layer):
    return {
        "num_embeddings": layer.num_embeddings,
        "embedding_dim": layer.embedding_dim,
        "padding_idx": layer.padding_idx,
    }

def _attributes_s4d(layer):
    if isinstance(layer, LayerS4D):
        return {
            "d_model": layer.h,
            "d_state": layer.n,
            "dropout": str(layer.dropout),
            "activation": layer.activation.__class__.__name__,
        }
    elif isinstance(layer, S4DKernel):
        return {
            "N": layer.n,
            "params": [p.shape for p in layer.parameters()],
        }
    return {}


def _attributes(layer):
    if isinstance(layer, torch.nn.Linear):
        return _attributes_linear(layer)
    elif isinstance(layer, torch.nn.BatchNorm1d):
        return _attributes_batchnorm(layer)
    elif isinstance(layer, torch.nn.LayerNorm):
        return _attributes_layernorm(layer)
    elif isinstance(layer, torch.nn.Embedding):
        return _attributes_embedding(layer)
    elif isinstance(layer, (LayerS4D, S4DKernel)):
        return _attributes_s4d(layer)
    else:
        return {}

@torch.no_grad()
def input_output_hook(module, inputs, output):
    if isinstance(inputs, tuple):
        if len(inputs) == 0:
            inp = None
        elif len(inputs) == 1:
            inp = inputs[0]
        else:
            inp = inputs
    else:
        inp = inputs

    if isinstance(output, tuple):
        if len(output) == 1:
            out = output[0]
        else:
            out = output
    else:
        out = output

    module._input = inp
    module._output = out

    if inp is not None and hasattr(inp, "shape"):
        module.input_shape = tuple(inp.shape)
    else:
        module.input_shape = None

    if out is not None and hasattr(out, "shape"):
        module.output_shape = tuple(out.shape)
    else:
        module.output_shape = None


@torch.no_grad()
def augment_names(model):
    i = 0
    for name, module in model.named_modules():
        module.name = model.name if module == model and hasattr(model, "name") else name
        module.layer_name = module.__class__.__name__
        module.parent = ".".join(name.split(".")[:-1])
        module.key = "{:03d}_{}".format(i, name.split(".")[-1])
        module.class_name = str(module.__class__)
        i += 1
    return model


@torch.no_grad()
def augment_childs(model):
    childs = []
    for name, submodule in model.named_children():
        childs.append(name)
        submodule_childs = augment_childs(submodule)
        submodule.childs = submodule_childs
    return childs


@torch.no_grad()
def augment_input_output(model, example_input):
    handles = []
    for _, layer in model.named_modules():
        handles.append(layer.register_forward_hook(input_output_hook))

    # Esegui forward pass
    with torch.no_grad():
        model(example_input)

    # Rimuovi gli hook
    for h in handles:
        h.remove()

""" def find_layers_connection_on(summary, direction):
    if direction == "input":
        opposite_direction = "output"
    else:
        direction = "output"
        opposite_direction = "input"

    for id_src in summary:
        id_src_list = summary[id_src][direction]
        layers = []
        for id_src_elem in id_src_list:
            found_out = False
            for id_out in summary:
                id_out_list = summary[id_out][opposite_direction]
                for id_out_elem in id_out_list:
                    if id_src_elem.shape == id_out_elem.shape:
                        found_out = True
                        layers.append(id_out)
                        break
                if found_out:
                    break
            if not found_out:
                layers.append(direction)
        summary[id_src][direction + "_layers"] = layers """


""" def find_layers_connection(summary):
    find_layers_connection_on(summary, "input")
    find_layers_connection_on(summary, "output")
    for id_in in summary:
        # Pulizia: tieni solo connessioni tra layer
        del summary[id_in]["input"]
        del summary[id_in]["output"] """

class ModelInfo: 
    def __init__(self, model, vocab_size, seq_len, batch_size, dataset_name="dummy"):
        self.model_name = model.name if hasattr(model, "name") else model.__class__.__name__
        self.dataset_name = dataset_name

        self.model = model
        self.model.eval()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size

        if vocab_size is not None:
            self.input_shape = (batch_size, seq_len)
        else:
            self.input_shape = (batch_size, seq_len, 3)

        self.augment()


    def set_input_shape(self, seq_len=None, batch_size=None):
        if seq_len is not None:
            self.seq_len = seq_len
        if batch_size is not None:
            self.batch_size = batch_size
        self.input_shape = (self.batch_size, self.seq_len)
        self.augment()


    @torch.no_grad()
    def augment(self):
        augment_names(self.model)
        self.model.childs = augment_childs(self.model)

        if self.vocab_size is not None:
            input_sample = torch.randint(
                0, self.vocab_size, self.input_shape, dtype=torch.long
            ).to(utils.get_device())
        else:
            input_sample = torch.randn(*self.input_shape).to(utils.get_device())

        augment_input_output(self.model, input_sample)
    

    def torchviz(self, output_dir="."):
        input = torch.randint(0, self.vocab_size, self.input_shape, dtype=torch.long).to(utils.get_device())
        output = self.model(input)
        model_graph = make_dot(output, params=dict(self.model.named_parameters()))
        model_graph.render(
            os.path.join(output_dir, "{}_{}_torchviz".format(self.model_name, self.dataset_name)), format="png")

    @torch.no_grad()
    def torchinfo(self, output_dir, mode):
        info = torchinfo.summary( 
            self.model, 
            input_size=(self.batch_size, self.seq_len) if self.vocab_size is not None else (self.batch_size, self.seq_len, 3),
            col_names=("input_size", "output_size", "num_params", "mult_adds"), 
            verbose=0, 
            dtypes=[torch.long] if self.vocab_size is not None else [torch.float],)

        if mode is not None:
            with open(os.path.join(output_dir, f"{self.model_name}_{self.dataset_name}_torchinfo_{mode}.txt"), "w") as text_file:
                text_file.write(str(info))
        else:
            with open(os.path.join(output_dir, f"{self.model_name}_{self.dataset_name}_torchinfo.txt"), "w") as text_file:
                text_file.write(str(info))

    @torch.no_grad()
    def torchsummary(self, output_dir="."):
        info = summary(self.model, (self.batch_size, self.seq_len))

        with open(os.path.join(output_dir, f"{self.model_name}_{self.dataset_name}_torchsummary.txt"), "w") as text_file:
            text_file.write(str(info))

    """ @torch.no_grad()
    def summary(self, output_dir="."):
        
        data = {}
        for name, module in self.model.named_modules():
            if not hasattr(module, "input_shape") or not hasattr(module, "output_shape"):
                continue
            data[name] = {
                # augment_names
                "name": module.name,
                "layer_name": module.layer_name,
                "parent": module.parent,
                "key": module.key,
                "class_name": module.class_name,
                # augment_childs
                "childs": module.childs,
                # augment input output
                "input_shape": module.input_shape,
                "output_shape": module.output_shape,
                # other
                "nb_params": sum(p.numel() for p in module.parameters()),
                "weights": [list(p.size()) for p in module.parameters()],
                "attributes": _attributes(module)
            }
    
        with open(os.path.join(output_dir, "{}_{}_summary.json".format(self.model_name, self.dataset_name)), "w") as json_file:
            json.dump(data, json_file, indent=4)
            
        return data """


    def get_example_input_data(self):
        example_input_data = {}
        for name, module in self.model.named_modules():
            if not hasattr(module, "input_shape"):
                continue
            example_input_data[module.name] = module.input_shape
        example_input_data[None] = self.input_shape
        return example_input_data


    def get_example_input(self, search_module=None):
        if search_module == None: 
            input_shape = self.input_shape
        else: 
            input_shape = search_module.input_shape
            
        if self.vocab_size is not None:
            example_input = torch.randint(0, self.vocab_size, input_shape, dtype=torch.long).to(utils.get_device())
        else:
            example_input = torch.randn(input_shape).to(utils.get_device())
        return example_input
