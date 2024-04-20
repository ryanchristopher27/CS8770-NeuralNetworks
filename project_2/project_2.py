from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader, TensorDataset
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor
import os

from utils.models import Network
from utils.data import Dataset, create_stock_dataset, StockDataModule


from utils.config import config
from utils.helpers import *
from utils.plots import *


# Fetch the dataset
air_quality = fetch_ucirepo(id=360)
X = air_quality.data.features

# Assuming 'Date' is the first column
X = X.iloc[:, 1:]

Y = X['T']

X = X.loc[:, X.columns != 'T']

# Convert non-numeric columns to numeric using one-hot encoding
X = pd.get_dummies(X)

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a PyTorch dataset
dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))

class AirQualityDataModule(L.LightningDataModule):
    def __init__(self, batch_size, dataset, val_batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = dataset
        self.val_batch_size = val_batch_size

    def setup(self, stage=None):
        dataset_size = len(self.dataset)
        val_size = int(0.2 * dataset_size)
        train_size = dataset_size - val_size

        # Split the dataset into train and validation sets
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size)


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

X_train, y_train, X_test, y_test = create_stock_dataset()

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))


config["model"]["input_size"] = X_scaled.shape[1]

task = config["experiment"]
path_results = config["paths"]["results"]

num_epochs = config["hyper_parameters"]["num_epochs"]
batch_size = config["hyper_parameters"]["batch_size"]

strategy = config["system"]["strategy"]
accelerator = config["system"]["accelerator"]
num_devices = config["system"]["num_devices"]

