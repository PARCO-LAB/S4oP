PRUNING_CONFIG = {
    "s4":
    {
        "listops":
        {
            "finetune_epochs": 6,
            "lr": 1e-4,
            "weight_decay": 1e-5,
        },
        "pathfinder":
        {
            "finetune_epochs": 20,
            "lr": 1e-4,
            "weight_decay": 1e-5,
        },
        "imdb":
        {
            "finetune_epochs": 4,
            "lr": 1e-5,
            "weight_decay": 1e-5,
        },
        "ecg":
        {
            "finetune_epochs": 20,
            "lr": 1e-4,
            "weight_decay": 1e-5,
        }
    },
    "s4d":
    {
        "listops":
        {
            "finetune_epochs": 5,
            "lr": 1e-4,
            "weight_decay": 1e-5,
        },
        "pathfinder":
        {
            "finetune_epochs": 20,
            "lr": 1e-4,
            "weight_decay": 1e-5,
        },
        "imdb":
        {
            "finetune_epochs": 4,
            "lr": 1e-5,
            "weight_decay": 1e-5,
        },
        "ecg":
        {
            "finetune_epochs": 20,
            "lr": 1e-4,
            "weight_decay": 1e-5,
        }
    }
}