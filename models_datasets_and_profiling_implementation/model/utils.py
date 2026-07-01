import torch
import random
import signal
import numpy as np
import gc
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def setup_optimizer(model, lr, weight_decay, epochs, warmup_epochs=5):
    all_parameters = list(model.parameters())

    # Speciali S4 (hanno _optim): lr/wd custom definiti dal layer
    special_s4 = [p for p in all_parameters if hasattr(p, "_optim")]
    # Mamba da NON decadere (A_log, D: hanno _no_weight_decay)
    no_decay = [p for p in all_parameters
                if not hasattr(p, "_optim") and getattr(p, "_no_weight_decay", False)]
    # Tutto il resto: weight decay normale
    base_params = [p for p in all_parameters
                   if not hasattr(p, "_optim") and not getattr(p, "_no_weight_decay", False)]

    optimizer = optim.AdamW(base_params, lr=lr, weight_decay=weight_decay)

    # Gruppo no-weight-decay (A_log/D di Mamba) -> stesso lr, wd=0
    if no_decay:
        optimizer.add_param_group({"params": no_decay, "weight_decay": 0.0})

    # Gruppi speciali S4 (_optim)
    hps = [getattr(p, "_optim") for p in special_s4]
    hps = [dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))]
    for hp in hps:
        params = [p for p in special_s4 if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group({"params": params, **hp})

    # Scheduler: warmup lineare + cosine
    warmup_epochs = max(1, min(warmup_epochs, int(epochs) // 10))
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, int(epochs) - warmup_epochs))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    return optimizer, scheduler


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True


def set_benchmark(value=True):
    torch.backends.cudnn.benchmark = value


def clean_memory():
    gc.collect()
    torch.cuda.memory_allocated(get_device())
    torch.cuda.memory_reserved(get_device())
    torch.cuda.empty_cache()


class GracefulKiller:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True



def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance