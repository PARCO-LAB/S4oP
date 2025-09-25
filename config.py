PRUNING_DEFAULT_CONFIG = {
    "datasetLRA":
    {
        "imdb":
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
            "batch_size": 100,
            "epochs": 50,
            "wd": 0.01,
            "patience": 5
        }
    },
    "num_workers": 4,
}