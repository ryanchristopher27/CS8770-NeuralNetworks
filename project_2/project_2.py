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
from stock_experiment import stock_experiment


def main():

    seq_lengths = [1, 5, 10, 25, 50]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, seq_length in enumerate(seq_lengths):
        config['data']['num_sequences'] = seq_length

        df_predictions, df_actual = stock_experiment(config, plot_results=False)

        df_predictions.reset_index(drop=True, inplace=True)
        df_actual.reset_index(drop=True, inplace=True)

        # Only Add Actual Once
        if i == 0:
            ax.plot(df_actual.index, df_actual, label='Actual')
        
        ax.plot(df_predictions.index, df_predictions, label=f'Predicted (Seq={seq_length})')


    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.set_title('Predicted vs Actual Values')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
