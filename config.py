# Patience: parametro per l'early stopping

PRUNING_DEFAULT_CONFIG = {
    "s4d":
    {
        "text":
        {
            "depth": 4,
            "features": 64,
            "norm": "BN",
            "pre-norm": True,
            "dropout": 0.0,
            "lr": 0.001,
            "batch_size": 50,
            "epochs": 20,
            "wd": 0,
            "patience": 5
        },
        "listops":
        {
            "depth": 8,
            "features": 128,
            "norm": "BN",
            "pre-norm": False,
            "dropout": 0.0,
            "lr": 0.001,
            "batch_size": 50,
            "epochs": 40,
            "wd": 0.05,
            "patience": 5
        },
        "imdb":
        {
            "depth": 2,
            "features": 128,
            "norm": "BN",
            "pre-norm": False,
            "dropout": 0.0,
            "lr": 0.001,
            "batch_size": 50,
            "epochs": 20,
            "wd": 0,
            "patience": 5
        },
    },
    "s4":
    {
        "text":
        {
            "depth": 4,
            "features": 64,
            "norm": "BN",
            "pre-norm": True,
            "dropout": 0.0,
            "lr": 0.001,
            "batch_size": 50,
            "epochs": 20,
            "wd": 0,
            "patience": 5
        },
        "listops":
        {
            "depth": 6,
            "features": 128,
            "norm": "BN",
            "pre-norm": False,
            "dropout": 0.0,
            "lr": 0.01,
            "batch_size": 50,
            "epochs": 50,
            "wd": 0.01,
            "patience": 5
        },
        "imdb":
        {
            "depth": 4,
            "features": 128,
            "norm": "LN",
            "pre-norm": False,
            "dropout": 0.0,
            "lr": 0.001,
            "batch_size": 40,
            "epochs": 20,
            "wd": 0,
            "patience": 5
        },
    },
    "num_workers": 4,
    "val_split": 0.2
}