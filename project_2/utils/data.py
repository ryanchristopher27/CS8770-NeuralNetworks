"""
Purpose: Data Tools
"""

# Imports
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import scale


class Dataset:
    """
    Purpose: Machine learning dataset
    """
    
    def __init__(self, samples, labels, shuffle=False):
        """
        Purpose: 
        - Define instance parameters, optionally shuffle

        Arguments:
        - samples (list): sequence of dataset observations
        - labels (list): dataset labels formatted as scalar or sequence
        """
        
        self.samples = samples
        self.labels = labels

        if shuffle:
            
            indices = np.arange(self.samples.shape[0])
            np.random.shuffle(indices)
            
            self.samples = self.samples[indices]
            self.labels = self.labels[indices]

    def __getitem__(self, index):
        """
        Purpose: 
        - Get dataset information for given index iteration

        Arguments:
        - index (int): current interation counter

        Returns:
        - (tuple[any]): sample and label that corresponds to iteration index
        """
        
        sample, label = self.samples[index], self.labels[index]

        return (sample.astype(np.float32), label.astype(np.float32))

    def __len__(self):
        """
        Purpose: 
        - Get number of dataset observations

        Returns:
        - (int): number of dataset observations
        """

        return self.samples.shape[0]


def get_partition(data, is_train, train_amount=0.8):
    """
    Purpose:
    - Format machine learning dataset into training and validation partitions
    
    Arguments:
    - data (Dataset): machine learning dataset
    - is_train (int): flag to designate training partition
    - train_amount (float): ratio of data for training partition
    
    Returns:
    - (Dataset): updated machine learning dataset
    """

    all_samples, all_labels = data.samples, data.labels

    # Scale: Dataset
    
    all_samples = np.asarray([scale(ele) for ele in all_samples])
    all_labels = scale(all_labels)

    num_train = int(len(all_samples) * train_amount)

    # Partition: Dataset
    
    if is_train:
        samples, labels = all_samples[:num_train], all_labels[:num_train]
    else:
        samples, labels = all_samples[num_train:], all_labels[num_train:]

    return Dataset(samples, labels)


def format_dataset(params, data):
    """
    Purpose:
    - Format machine learning dataset into pytorch dataloaders
    
    Arguments:
    - params (dict[str, any]): user defined parameters
    - data (Dataset): machine learning dataset
    
    Returns:
    - (tuple[torch.utils.data.DataLoader]): pytorch dataloaders
    """

    size = params["model"]["batch_size"]
    num_workers = params["system"]["num_workers"]

    # Format: Dataset --> Training & Validation Datasets

    train = get_partition(data, is_train=1)
    valid = get_partition(data, is_train=0)
    
    # Format: Datasets --> DataLoader
    
    train = DataLoader(train, num_workers=num_workers, shuffle=True,
                       persistent_workers=True, batch_size=size)
    
    valid = DataLoader(valid, num_workers=num_workers, shuffle=False,
                       persistent_workers=True, batch_size=size)

    return (train, valid)



def generate_dataset(params):
    """
    Purpose: 
    - Generate dataset for sequence analysis (many to one, many to many)

    Arguments:
    - params (dict[str, any]): user defined parameters

    Returns:
    - (Function): data generation function
    """
    
    task = params["experiment"]

    if task == 0 or task == 1:
        loader = generate_signals
    else:
        raise NotImplementedError

    return loader(params)