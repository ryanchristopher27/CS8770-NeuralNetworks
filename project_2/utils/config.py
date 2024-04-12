# Config File
# Contains Parameters for Current Run

config = {}


#==========================================================
# General Parameters
config["experiment"] = 0

#==========================================================
# System Parameters
config["system"] = {
    "accelerator": "gpu",
    "strategty": "auto",
    "num_deviced": 1,
    "num_workers": 8,
}

# Data Parameters
#==========================================================
config["data"] = {
    "num_sequences": 3,
    "num_samples": 2000,
    "num_features": 100,
}

# Model Paramters
#==========================================================
config["model"] = {
    "num_layers": 3,
    "hidden_size": 512,
}

# Hyper Paramters
#==========================================================
config["hyper_parameters"] = {
    "batch_size": 64,
    "learning_rate": 3e-3,
    "num_epochs": 100,
}

# Evaluation Parameters
#==========================================================
config["evaluation"] = {
    "tags": [
        "train_error_epoch",
        "valid_error_epoch",
        "lr-Adam"
    ]
}

# Path Parameters
#==========================================================
config["path"] = {
    "results": "results",
    "version": 0,
}
