MODELS_CONFIG = {
    "s4d":
    {
        "ecg":
        {
            "depth": 4,
            "features": 128,
            "norm": "LN",
            "pre-norm": False,
            "dropout": 0.2,
            "lr": 0.001,
            "batch_size": 32,
            "epochs": 200,
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
            "depth": 4,
            "features": 256,
            "norm": "LN",
            "pre-norm": True,
            "dropout": 0.2,
            "lr": 0.001,
            "batch_size": 32,
            "epochs": 30,
            "wd": 1e-4,
            "patience": 5
        },
        "pathfinder":
        {
            "depth": 6,
            "features": 256,
            "norm": "BN",
            "pre-norm": True,
            "dropout": 0.1,
            "lr": 0.004,
            "batch_size": 64,
            "epochs": 200,
            "wd": 0.03,
            "patience": 20
        }
    },
    "s4":
    {
        "ecg":
        {
            "depth": 4,
            "features": 128,
            "norm": "LN",
            "pre-norm": False,
            "dropout": 0.2,
            "lr": 0.001,
            "batch_size": 32,
            "epochs": 200,
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
            "features": 256,
            "norm": "LN",
            "pre-norm": True,
            "dropout": 0.2,
            "lr": 0.001,
            "batch_size": 32,
            "epochs": 30,
            "wd": 1e-4,
            "patience": 5
        },
        "pathfinder":
        {
            "depth": 6,
            "features": 256,
            "norm": "BN",
            "pre-norm": True,
            "dropout": 0.1,
            "lr": 0.004,
            "batch_size": 100,
            "epochs": 200,
            "wd": 0,
            "patience": 20
        }
    },
    "num_workers": 4,
    "val_split": 0.2
}