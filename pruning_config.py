PRUNING_CONFIG = {
    "s4":
    {
        "listops":
        {
            "finetune_epochs": 6,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "early_stopping": 2
        },
        "pathfinder":
        {
            "finetune_epochs": 20,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "early_stopping": 6
        },
        "imdb":
        {
            "finetune_epochs": 3,
            "lr": 1e-5,
            "weight_decay": 1e-5,
            "early_stopping": 1
        },
        "ecg":
        {
            "finetune_epochs": 20,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "early_stopping": 6
        }
    },
    "s4d":
    {
        "listops":
        {
            "finetune_epochs": 5,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "early_stopping": 2
        },
        "pathfinder":
        {
            "finetune_epochs": 20,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "early_stopping": 6
        },
        "imdb":
        {
            "finetune_epochs": 4,
            "lr": 1e-5,
            "weight_decay": 1e-5,
            "early_stopping": 1
        },
        "ecg":
        {
            "finetune_epochs": 20,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "early_stopping": 6
        }
    },
    "perc": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7],
    #"perc": [0.1, 0.3, 0.5, 0.7, 0.9],
    #"seeds": [7, 42, 123, 2024, 314159]
    "seeds": [42]
}