from functools import partial
from . import s4

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import utils

__all__ = {
    #"s4": s4.s4,
    "s4d": s4.S4D,
}

class NetFactory:
    def __init__(self, model_name, d_model, num_classes, vocab_size):
        if model_name == "s4d":
            self.net = __all__["s4d"](d_model=d_model, vocab_size=vocab_size, num_classes=num_classes)
        else:
            raise ValueError(f"Model {model_name} not recognized")

        self.name = model_name
        self.net.name = model_name

    def get_net(self):
        self.net.to(utils.get_device())
        return self.net