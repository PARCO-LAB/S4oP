MODELS_CONFIG = {
    "s4d":
    {
        "ecg":
        {
            "d_state": 64,
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
            "d_state": 64,
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
            "d_state": 64,
            "depth": 6,
            "features": 256,
            "norm": "BN",
            "pre-norm": True,
            "dropout": 0.0,
            "lr": 0.01,
            "batch_size": 16,
            "epochs": 32,
            "wd": 0.05,
            "patience": 5
        },
        "pathfinder":
        {
            "d_state": 64,
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
        },
        "image":
        {
            "d_state": 64,
            "depth": 6,
            "features": 512,
            "norm": "LN",
            "pre-norm": False,
            "dropout": 0.1,
            "lr": 0.01,
            "batch_size": 50,
            "epochs": 200,
            "wd": 0.05,
            "patience": 20
        },
        "retrieval":
        {
            "d_state": 64,
            "depth": 6,
            "features": 256,
            "norm": "BN",
            "pre-norm": True,
            "dropout": 0.0,
            "lr": 0.002,
            "batch_size": 64,
            "epochs": 20,
            "wd": 0.05,
            "patience": 5
        }
    },
    "s4":
    {
        "ecg":
        {
            "d_state": 64,
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
            "d_state": 64,
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
            "d_state": 64,
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
        "pathfinder":
        {
            "d_state": 64,
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
        },
        "image":
        {
            "d_state": 64,
            "depth": 6,
            "features": 512,
            "norm": "LN",
            "pre-norm": False,
            "dropout": 0.2,
            "lr": 0.004,
            "batch_size": 50,
            "epochs": 200,
            "wd": 0.01,
            "patience": 20
        },
        "retrieval":
        {
            "d_state": 64,
            "depth": 6,
            "features": 256,
            "norm": "BN",
            "pre-norm": True,
            "dropout": 0.0,
            "lr": 0.002,
            "batch_size": 64,
            "epochs": 20,
            "wd": 0.0,
            "patience": 5
        }
    },
    "mamba":
    {
        "ecg":
        {
            "d_state": 64,
            "depth": 6,
            "features": 13, # 256
            "norm": "LN",
            "pre-norm": True,
            "dropout": 0.2,
            "lr": 0.001,
            "batch_size": 16,
            "epochs": 100,
            "wd": 0.01,
            "patience": 50
        },
        "listops":
        {
            "d_state": 64,
            "depth": 12,
            "features": 13, # 256
            "norm": "LN",
            "pre-norm": True,
            "dropout": 0.3,
            "lr": 0.0001,
            "batch_size": 32,
            "epochs": 100,
            "wd": 0.01,
            "patience": 20
        },
        "imdb":
        {
            "d_state": 64,
            "depth": 4,
            "features": 13, # 256
            "norm": "LN",
            "pre-norm": True,
            "dropout": 0.1,
            "lr": 0.001,
            "batch_size": 32,
            "epochs": 32,
            "wd": 0.01,
            "patience": 4
        },
        "pathfinder":
        {
            "d_state": 64,
            "depth": 6,
            "features": 13, # 256
            "norm": "LN",
            "pre-norm": True,
            "dropout": 0.3,
            "lr": 0.0001,
            "batch_size": 32,
            "epochs": 100,
            "wd": 0.01,
            "patience": 20
        },
        "image":
        {
            "d_state": 64,
            "depth": 12,
            "features": 13, # 256
            "norm": "LN",
            "pre-norm": True,
            "dropout": 0.3,
            "lr": 0.0001,
            "batch_size": 32,
            "epochs": 200,
            "wd": 0.01,
            "patience": 20
        },
        "retrieval":
        {
            "d_state": 64,
            "depth": 6,
            "features": 13, # 256
            "norm": "LN",
            "pre-norm": True,
            "dropout": 0.0,
            "lr": 0.0001,
            "batch_size": 16,
            "epochs": 50,
            "wd": 0.01,
            "patience": 10
        },
    },
    "mamba2":
    {
        "ecg":
        {
            "d_state": 64,
            "depth": 6,
            "features": 256, # 256
            "headdim": 16, 
            "norm": "LN",
            "pre-norm": True,
            "dropout": 0.2,
            "lr": 0.001,
            "batch_size": 16,
            "epochs": 100,
            "wd": 0.01,
            "patience": 50
        },
        "listops":
        {
            "d_state": 64,
            "depth": 12,
            "features": 256, # 256
            "headdim": 16,
            "norm": "LN",
            "pre-norm": True,
            "dropout": 0.3,
            "lr": 0.0001,
            "batch_size": 32,
            "epochs": 100,
            "wd": 0.01,
            "patience": 20
        },
        "imdb":
        {
            "d_state": 64,
            "depth": 4,
            "features": 256, # 256
            "headdim": 16,
            "norm": "LN",
            "pre-norm": True,
            "dropout": 0.1,
            "lr": 0.001,
            "batch_size": 32,
            "epochs": 32,
            "wd": 0.01,
            "patience": 4
        },
        "pathfinder":
        {
            "d_state": 64,
            "depth": 6,
            "features": 256, # 256
            "headdim": 16,
            "norm": "LN",
            "pre-norm": True,
            "dropout": 0.3,
            "lr": 0.0001,
            "batch_size": 32,
            "epochs": 100,
            "wd": 0.01,
            "patience": 20
        },
        "image":
        {
            "d_state": 64,
            "depth": 12,
            "features": 256, # 256
            "headdim": 16,
            "norm": "LN",
            "pre-norm": True,
            "dropout": 0.3,
            "lr": 0.0001,
            "batch_size": 32,
            "epochs": 200,
            "wd": 0.01,
            "patience": 20
        },
        "retrieval":
        {
            "d_state": 64,
            "depth": 6,
            "features": 256, # 256
            "headdim": 16,
            "norm": "LN",
            "pre-norm": True,
            "dropout": 0.0,
            "lr": 0.0001,
            "batch_size": 16,
            "epochs": 50,
            "wd": 0.01,
            "patience": 10
        },
    },
    "num_workers": 4,
    "val_split": 0.2
}