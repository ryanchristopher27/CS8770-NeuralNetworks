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
    "num_sequences": 1,
    "num_samples": 2000,
    "num_features": 36,
}

# Model Paramters
#==========================================================
config["model"] = {
    "type": "LSTM",
    "num_layers": 1,
    "hidden_size": 36,
    "input_size": 36,
}

# Hyper Paramters
#==========================================================
config["hyper_parameters"] = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 100,
    "objective": "mse_loss",
    "optimizer": "SGD",
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
