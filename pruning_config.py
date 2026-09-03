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
        },
        "image":
        {
            "finetune_epochs": 20,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "early_stopping": 2
        },
        "retrieval":
        {
            "finetune_epochs": 4,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "early_stopping": 1
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
        },
        "image":
        {
            "finetune_epochs": 20,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "early_stopping": 2
        },
        "retrieval":
        {
            "finetune_epochs": 4,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "early_stopping": 1
        }
    },
    "mamba":
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
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "early_stopping": 2
        },
        "ecg":
        {
            "finetune_epochs": 20,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "early_stopping": 6
        },
        "image":
        {
            "finetune_epochs": 5,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "early_stopping": 2
        },
        "retrieval":
        {
            "finetune_epochs": 6,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "early_stopping": 2
        }
    },
    "mamba2":
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
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "early_stopping": 2
        },
        "ecg":
        {
            "finetune_epochs": 20,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "early_stopping": 6
        },
        "image":
        {
            "finetune_epochs": 5,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "early_stopping": 2
        },
        "retrieval":
        {
            "finetune_epochs": 6,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "early_stopping": 2
        }
    },
    "perc": [0.1, 0.3, 0.5, 0.7, 0.9],
    "seeds": [7, 42, 123, 2024, 31650]
}