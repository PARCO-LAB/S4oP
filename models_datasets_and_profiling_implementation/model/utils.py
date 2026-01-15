import torch
import random
import signal
import numpy as np
import gc
import torch.optim as optim

def setup_optimizer(model, lr, weight_decay, epochs):
    """
    Setup dell'optimizer per S4 coerente con la repo ufficiale.

    - Parametri speciali (A, B, C, dt) hanno _optim settato
    e usano lr più piccolo (~1e-3) e no weight decay.
    - Tutti gli altri parametri usano lr più grande (es. 1e-2) e weight decay.
    """

    # Tutti i parametri del modello
    all_parameters = list(model.parameters())

    # Parametri generali (senza attributo _optim)
    base_params = [p for p in all_parameters if not hasattr(p, "_optim")]
    optimizer = optim.AdamW(base_params, lr=lr, weight_decay=weight_decay)

    # Raggruppa i parametri speciali (_optim)
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    # Elimina duplicati mantenendo ordine
    hps = [
        dict(s) for s in sorted(
            list(dict.fromkeys(frozenset(hp.items()) for hp in hps))
        )
    ]

    # Aggiunge ogni gruppo speciale all'optimizer
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group({"params": params, **hp})

    # Scheduler: CosineAnnealingLR (come nella repo ufficiale)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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