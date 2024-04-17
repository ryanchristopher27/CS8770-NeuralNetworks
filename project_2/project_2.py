import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from utils.models import Network
from utils.data import Dataset


from utils.config import config

# Fetch the dataset
air_quality = fetch_ucirepo(id=360)
X = air_quality.data.features

# Assuming 'Date' is the first column
X = X.iloc[:, 1:]

# Convert non-numeric columns to numeric using one-hot encoding
X = pd.get_dummies(X)

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a PyTorch dataset
dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))

class AirQualityDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, dataset):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = dataset

    def setup(self, stage=None):
        self.train_dataset = dataset

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

# class RNNModel(pl.LightningModule):
#     def __init__(self, input_size, hidden_size, num_layers):
#         super().__init__()
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, input_size)

#     def forward(self, x):
#         out, _ = self.rnn(x)
#         out = self.fc(out)
#         return out

#     def training_step(self, batch, batch_idx):
#         x = batch[0]
#         x = x.unsqueeze(1)  # Add sequence length dimension
#         x_hat = self(x)
#         loss = nn.MSELoss()(x_hat, x)
#         self.log('train_loss', loss)
#         return loss

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=0.001)

# Set hyperparameters
# input_size = X_scaled.shape[1]
# hidden_size = 64
# num_layers = 2
# batch_size = 32
# num_epochs = 10

# # Create data module and model
# data_module = AirQualityDataModule(batch_size)
# # model = RNNModel(input_size, hidden_size, num_layers)
# model = Network(config)

# # Train the model
# trainer = pl.Trainer(max_epochs=num_epochs)
# trainer.fit(model, data_module)

config["model"]["input_size"] = X_scaled.shape[1]

task = config["experiment"]
path_results = config["paths"]["results"]

num_epochs = config["hyper_parameters"]["num_epochs"]
batch_size = config["hyper_parameters"]["batch_size"]

strategy = config["system"]["strategy"]
accelerator = config["system"]["accelerator"]
num_devices = config["system"]["num_devices"]

# Data Module
data_module = AirQualityDataModule(batch_size, dataset)

# Create Model
model = Network(config)

# Create: Logger

path_save = path_results + "/exp_%s" % task
exp_logger = CSVLogger(save_dir=path_save)

# Create: Trainer

lr_monitor = LearningRateMonitor(logging_interval="epoch")

trainer = L.Trainer(callbacks=[lr_monitor], 
                    logger=exp_logger, devices=num_devices, 
                    log_every_n_steps=1, max_epochs=num_epochs, 
                    strategy=strategy, accelerator=accelerator)

# Train: Model

trainer.fit(model=model, train_dataloaders=data_module.train_dataloader())