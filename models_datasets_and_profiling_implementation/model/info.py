import os
import torch
import torchinfo
from torchsummary import summary
from torchviz import make_dot
from . import utils

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

class ModelInfo: 
    def __init__(self, model, vocab_size, seq_len, batch_size, dataset_name="dummy"):
        self.model_name = model.name if hasattr(model, "name") else model.__class__.__name__
        self.dataset_name = dataset_name

        self.model = model
        self.model.eval()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size

        if self.dataset_name in ["listops", "imdb"]:
            self.input_shape = (batch_size, seq_len)
        elif dataset_name == "retrieval":
            self.input_shape = (batch_size, 2, seq_len)
        else:
            self.input_shape = (batch_size, seq_len, 1)

        self.augment()


    def set_input_shape(self, seq_len=None, batch_size=None):
        if seq_len is not None:
            self.seq_len = seq_len
        if batch_size is not None:
            self.batch_size = batch_size
        if self.dataset_name == "retrieval":
            self.input_shape = (batch_size, 2, seq_len)
        elif self.dataset_name in ["listops", "imdb"]:
            self.input_shape = (batch_size, seq_len)
        else:
            self.input_shape = (batch_size, seq_len, 1) if self.dataset_name == "image" else (batch_size, seq_len, 12)
        self.augment()


    @torch.no_grad()
    def augment(self):
        augment_names(self.model)
        self.model.childs = augment_childs(self.model)

        if self.dataset_name in ["listops", "imdb", "retrieval"]:
            input_sample = torch.randint(
                0, self.vocab_size, self.input_shape, dtype=torch.long
            ).to(utils.get_device())
        else:
            input_sample = torch.randn(*self.input_shape).to(utils.get_device())

        augment_input_output(self.model, input_sample)
    

    def torchviz(self, output_dir="."):
        if self.dataset_name in ["listops", "imdb", "retrieval"]:
            input = torch.randint(0, self.vocab_size, self.input_shape, dtype=torch.long).to(utils.get_device())
        else:
            input = torch.randn(*self.input_shape).to(utils.get_device())
        output = self.model(input)
        model_graph = make_dot(output, params=dict(self.model.named_parameters()))
        model_graph.render(
            os.path.join(output_dir, "{}_{}_torchviz".format(self.model_name, self.dataset_name)), format="png")

    @torch.no_grad()
    def torchinfo(self, output_dir, name):
        if self.dataset_name == "retrieval":
            input_size = (self.batch_size, 2, self.seq_len)
            dtypes = [torch.long]
        elif self.dataset_name in ["listops", "imdb"]:
            input_size = (self.batch_size, self.seq_len)
            dtypes = [torch.long]
        else:
            input_size = (self.batch_size, self.seq_len, 1) if self.dataset_name == "image" else (self.batch_size, self.seq_len, 12)
            dtypes = [torch.float]

        info = torchinfo.summary(
            self.model,
            input_size=input_size,
            col_names=("input_size", "output_size", "num_params", "mult_adds"),
            verbose=0,
            dtypes=dtypes,
        )
        with open(os.path.join(output_dir, f"{name}_torchinfo.txt"), "w") as text_file:
            text_file.write(str(info))

    @torch.no_grad()
    def torchsummary(self, output_dir=".", name=None):
        info = summary(self.model, (self.batch_size, self.seq_len))

        with open(os.path.join(output_dir, f"{name}_torchsummary.txt"), "w") as text_file:
            text_file.write(str(info))


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
            
        if self.dataset_name in ["listops", "imdb", "retrieval"]:
            example_input = torch.randint(0, self.vocab_size, input_shape, dtype=torch.long).to(utils.get_device())
        else:
            example_input = torch.randn(input_shape).to(utils.get_device())
        return example_input
