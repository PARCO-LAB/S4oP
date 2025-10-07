PRUNING_CONFIG = {
    "iterative1": 
    {
        "iterations": 3,
        "finetune_epochs": 5,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "perc_channels": 0.5,
    },
    "iterative2": 
    {
        "iterations": 5,
        "finetune_epochs": 5,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "perc_channels": 0.3,
    },
    "iterative3": 
    {
        "iterations": 10,
        "finetune_epochs": 10,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "perc_channels": 0.3,
    },
    "one-shot1": 
    {
        "finetune_epochs": 10,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "perc_channels": 0.5,
    },
    "one-shot2": 
    {
        "finetune_epochs": 15,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "perc_channels": 0.7,
    },
    "one-shot3": 
    {
        "finetune_epochs": 15,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "perc_channels": 0.9,
    },
}