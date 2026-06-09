from . import s4
from . import mamba

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import utils

__all__ = {
    "s4_old": s4.S4_old,
    "s4d_old": s4.S4D_old,
    "s4d": s4.S4D,
    "s4": s4.S4,
    "mamba": mamba.Mamba
}

class NetFactory:
    def __init__(
        self, 
        model_name, 
        dataset_name,
        vocab_size, 
        d_model, 
        d_state,
        depth,
        dropout,
        num_classes,
        norm,
        pre_norm,
        active_idx_layers=None, # lista di indici attivi per S4D pruned
    ):
        if model_name == "s4d_old":
            self.net = __all__["s4d_old"](vocab_size=vocab_size,
                                      dataset_name=dataset_name,
                                      d_model=d_model,
                                      d_state=d_state,
                                      depth=depth,
                                      dropout=dropout,
                                      num_classes=num_classes,
                                      norm=norm,
                                      pre_norm=pre_norm)
        elif model_name == "s4_old":
            self.net = __all__["s4_old"](vocab_size=vocab_size,
                                     dataset_name=dataset_name,
                                     d_model=d_model,
                                     d_state=d_state,
                                     depth=depth,
                                     dropout=dropout,
                                     num_classes=num_classes,
                                     norm=norm,
                                     pre_norm=pre_norm)  
        elif model_name == "s4d":
            self.net = __all__["s4d"](vocab_size=vocab_size,
                                            dataset_name=dataset_name,
                                            d_model=d_model,
                                            d_state=d_state,
                                            depth=depth,
                                            dropout=dropout,
                                            num_classes=num_classes,
                                            active_idx_layers=active_idx_layers,
                                            norm=norm,
                                            pre_norm=pre_norm)
        elif model_name == "s4":
            self.net = __all__["s4"](vocab_size=vocab_size,
                                            dataset_name=dataset_name,
                                            d_model=d_model,
                                            d_state=d_state,
                                            depth=depth,
                                            dropout=dropout,
                                            num_classes=num_classes,
                                            active_idx_layers=active_idx_layers,
                                            norm=norm,
                                            pre_norm=pre_norm)
        elif model_name == "mamba":
            self.net = __all__["mamba"](vocab_size=vocab_size,
                                            dataset_name=dataset_name,
                                            d_model=d_model,
                                            d_state=d_state,
                                            depth=depth,
                                            dropout=dropout,
                                            num_classes=num_classes,
                                            active_idx_layers=active_idx_layers,
                                            norm=norm,
                                            pre_norm=pre_norm)
        else:
            raise ValueError(f"Model {model_name} not recognized")
        

        self.name = model_name
        self.net.name = model_name

    def get_net(self):
        self.net.to(utils.get_device())
        return self.net