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
    "strategy": "auto",
    "num_devices": 1,
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
    "type": "LSTM",
    "num_layers": 2,
    "hidden_size": 64,
    "input_size": 10,
}

# Hyper Paramters
#==========================================================
config["hyper_parameters"] = {
    "batch_size": 64,
    "learning_rate": 3e-3,
    "num_epochs": 100,
    "objective": "mse_loss",
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
config["paths"] = {
    "results": "results",
    "version": 0,
}
