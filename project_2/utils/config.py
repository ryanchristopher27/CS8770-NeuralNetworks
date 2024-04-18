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
    "num_samples": 555,
    "num_sequences": 50,
    "num_features": 1,
}

# Model Paramters
#==========================================================
config["model"] = {
    "type": "RNN",
    "num_layers": 2,
    "hidden_size": 50,
    "input_size": 1,
}

# Hyper Paramters
#==========================================================
config["hyper_parameters"] = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 50,
    "objective": "mse_loss",
    "optimizer": "Adam", # Adam, SGD
}

# Evaluation Parameters
#==========================================================
config["evaluation"] = {
    "tags": [
        "train_error_epoch",
        "valid_error_epoch",
        "lr-" + config["hyper_parameters"]["optimizer"]
    ]
}

# Path Parameters
#==========================================================
config["paths"] = {
    "results": "results",
    "version": 0,
}
