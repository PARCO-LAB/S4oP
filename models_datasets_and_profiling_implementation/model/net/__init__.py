import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import utils

class NetFactory:
    def __init__(
        self, 
        model_name, 
        dataset_name,
        vocab_size, 
        input_size,
        d_model, 
        d_state,
        depth,
        dropout,
        num_classes,
        norm,
        pre_norm,
        active_idx_layers,
        dual_stream = False
    ):
        if model_name == "s4d":
            from . import s4
            if dual_stream:
                self.net = s4.RetrievalS4D(vocab_size=vocab_size,
                                            input_size=input_size,
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
                self.net = s4.S4D(vocab_size=vocab_size,
                                    input_size=input_size,
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
            from . import s4
            if dual_stream:
                self.net = s4.RetrievalS4(vocab_size=vocab_size,
                                            input_size=input_size,
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
                self.net = s4.S4(vocab_size=vocab_size,
                                    input_size=input_size,
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
            from . import mamba 
            if dual_stream:
                self.net = mamba.RetrievalMamba(vocab_size=vocab_size,
                                                input_size=input_size,
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
                self.net = mamba.Mamba(vocab_size=vocab_size,
                                        input_size=input_size,
                                        dataset_name=dataset_name,
                                        d_model=d_model,
                                        d_state=d_state,
                                        depth=depth,
                                        dropout=dropout,
                                        num_classes=num_classes,
                                        active_idx_layers=active_idx_layers,
                                        norm=norm,
                                        pre_norm=pre_norm)
        elif model_name == "mamba2":
            from . import mamba 
            if dual_stream:
                self.net = mamba.RetrievalMamba2(vocab_size=vocab_size,
                                                input_size=input_size,
                                                dataset_name=dataset_name,
                                                d_model=d_model,
                                                d_state=d_state,
                                                depth=depth,
                                                dropout=dropout,
                                                num_classes=num_classes,
                                                active_heads_layers=active_idx_layers,
                                                norm=norm,
                                                pre_norm=pre_norm)
            else:
                self.net = mamba.Mamba2(vocab_size=vocab_size,
                                        input_size=input_size,
                                        dataset_name=dataset_name,
                                        d_model=d_model,
                                        d_state=d_state,
                                        depth=depth,
                                        dropout=dropout,
                                        num_classes=num_classes,
                                        active_heads_layers=active_idx_layers,
                                        norm=norm,
                                        pre_norm=pre_norm)
        else:
            raise ValueError(f"Model {model_name} not recognized")

        self.name = model_name
        self.net.name = model_name

    def get_net(self):
        self.net.to(utils.get_device())
        return self.net