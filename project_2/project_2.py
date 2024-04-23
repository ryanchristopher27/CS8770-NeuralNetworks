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
import sys

from utils.models import Network
from utils.data import Dataset, create_stock_dataset, StockDataModule


from utils.config import config
from utils.helpers import *
from utils.plots import *
from stock_experiment import stock_experiment


def main():
    num_args = len(sys.argv)

    experiment = 'dynamic'
    exp_type = 'learning_rate'
    iterable = [0.1, 0.01, 0.001, 0.0001]

    plot_results = False

    if num_args >= 2:
        if sys.argv[1].lower() == 'dynamic':
            experiment = 'dynamic'
        elif sys.argv[1].lower() =='single':
            experiment ='single'

    if experiment !='single':
        dynamic_stock_experiment(exp_type=exp_type, iterable=iterable, plot_results=plot_results)
    else:
        single_stock_experiment(plot_results=plot_results)

def dynamic_stock_experiment(exp_type, iterable, plot_results):

    print("===" * 30)
    print(f"{exp_type.capitalize()} Experiment")
    print(f"Options: {iterable}")
    print("===" * 30)

    # seq_lengths = [1, 5, 10, 25, 50]
    # seq_lengths = [1, 3, 5]

    fig, ax = plt.subplots(figsize=(10, 6))

    test_errors = np.zeros(len(iterable))
    # test_errors = np.zeros(len(seq_lengths))

    # for i, seq_length in enumerate(seq_lengths):
    for i, val in enumerate(iterable):
        if exp_type == 'sequence':
            config['data']['num_sequences'] = val
        elif exp_type == 'learning_rate':
            config['hyper_parameters']['learning_rate'] = val
        elif exp_type == 'hidden_size':
            config['model']['hidden_size'] = val
        else:
            raise ValueError(f'Invalid Experiment Type: {exp_type}')
        
        print("===" * 30)
        print(f'{exp_type.capitalize()} Experiment: ({val}) [{i+1}/{len(iterable)}] ')
        print("===" * 30)

        df_predictions, df_actual, test_error = stock_experiment(config, plot_results=plot_results)

        test_errors[i] = test_error

        df_predictions.reset_index(drop=True, inplace=True)
        df_actual.reset_index(drop=True, inplace=True)

        # Only Add Actual Once
        if i == 0:
            ax.plot(df_actual.index, df_actual, label='Actual')
        
        ax.plot(df_predictions.index, df_predictions, label=f'Predicted ({val})')
        # ax.plot(df_predictions.index, df_predictions, label=f'Predicted (Seq={seq_length})')

    # Plot Predictions
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.set_title(f'{exp_type.capitalize()} Predicted vs Actual Values')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Test Errors
    plt.figure(figsize=(8, 6))
    plt.plot(iterable, test_errors, marker='o')
    plt.xlabel(f'{exp_type.capitalize()}')
    plt.ylabel('Test Error')
    plt.title(f'Test Error vs. {exp_type.capitalize()}')
    plt.grid(True)
    plt.show()



def single_stock_experiment(plot_results):

    df_predictions, df_actual, test_error = stock_experiment(config, plot_results=plot_results)
    df_predictions.reset_index(drop=True, inplace=True)
    df_actual.reset_index(drop=True, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df_actual.index, df_actual, label='Actual')
    ax.plot(df_predictions.index, df_predictions, label=f'Predicted')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.set_title('Predicted vs Actual Values')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
