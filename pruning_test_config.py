PRUNING_CONFIG = {
    "s4":
    {
        "listops":
        {
            "finetune_epochs": 3,
            "lr": 1e-4,
            "weight_decay": 1e-5,
        },
        "pathfinder":
        {
            "finetune_epochs": 12,
            "lr": 1e-4,
            "weight_decay": 1e-5,
        },
        "imdb":
        {
            "finetune_epochs": 2,
            "lr": 1e-5,
            "weight_decay": 1e-5,
        },
        "ecg":
        {
            "finetune_epochs": 25,
            "lr": 1e-4,
            "weight_decay": 1e-5,
        }
    },
    "s4d":
    {
        "listops":
        {
            "finetune_epochs": 3,
            "lr": 1e-4,
            "weight_decay": 1e-5,
        },
        "pathfinder":
        {
            "finetune_epochs": 12,
            "lr": 1e-4,
            "weight_decay": 1e-5,
        },
        "imdb":
        {
            "finetune_epochs": 2,
            "lr": 1e-5,
            "weight_decay": 1e-5,
        },
        "ecg":
        {
            "finetune_epochs": 25,
            "lr": 1e-4,
            "weight_decay": 1e-5,
        }
    }
}

""" PRUNING_CONFIG = {
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
            "finetune_epochs": 25,
            "lr": 1e-4,
            "weight_decay": 1e-5,
        },
        "imdb":
        {
            "finetune_epochs": 4,
            "lr": 1e-5,
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
            "finetune_epochs": 25,
            "lr": 1e-4,
            "weight_decay": 1e-5,
        },
        "imdb":
        {
            "finetune_epochs": 4,
            "lr": 1e-5,
            "weight_decay": 1e-5,
        }
    }
} """

""" "iterative1": 
    {
        "iterations": 3,
        "finetune_epochs": 10, # 5, ho messo 50 per i test con pathfinder
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "perc_channels": 0.1, # 0.5, ho messo 0.1 per i test esponenziali
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
        "finetune_epochs": 10, # 10, ho messo 50 per i test con pathfinder
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "perc_channels": 0.1, #0.5, ho messo 0.3 per i test esponenziali
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
    }, """