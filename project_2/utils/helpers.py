# Imports
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from utils.plots import *

def cuda_setup() -> tuple:
    if torch.cuda.is_available():
        print(torch.cuda.current_device())     # The ID of the current GPU.
        print(torch.cuda.get_device_name(id))  # The name of the specified GPU, where id is an integer.
        print(torch.cuda.device(id))           # The memory address of the specified GPU, where id is an integer.
        print(torch.cuda.device_count())
        
    on_gpu = torch.cuda.is_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    return device, on_gpu


# Imports
import seaborn as sn  # yes, I had to "conda install seaborn"
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch                                                                                # Library: Pytorch 
import torch.utils.data as utils  

def confusion_matrix(y_true, y_pred, num_classes, num_samples, class_names=None):
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    ConfusionMatrix = torch.zeros((num_classes ,num_classes))
    for i in range(num_samples):
        ConfusionMatrix[int(y_true[i]),int(y_pred[i])] = ConfusionMatrix[int(y_true[i]),int(y_pred[i])] + 1

    df_cm = pd.DataFrame(np.asarray(ConfusionMatrix), index = [i for i in class_names],
                    columns = [i for i in class_names])
    
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt='.0f')
    plt.show()   


class Format_Dataset(utils.Dataset):

    def __init__(self, data_params, choice):
        
        self.choice = choice 
        self.samples = torch.Tensor(data_params['samples']).to(torch.float64)             # Gather: Data Samples
        if(self.choice.lower() == 'train'): 
            self.labels = torch.Tensor(data_params['labels']).to(torch.float64)           # Gather: Data Labels
        
    def __getitem__(self, index):                                                           
        
        if(self.choice.lower() == 'train'): 
            return self.samples[index], self.labels[index]                                  # Return: Next (Sample, Label) 
        else:
            return self.samples[index]                                                     

    def __len__(self):                                                                      
        
        return len(self.samples)
    
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.samples = torch.tensor(dataframe['samples'], dtype=torch.float64)
        self.labels = torch.tensor(dataframe['labels'], dtype=torch.float64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]
         
        # image = torch.Tensor(self.data.iloc[idx]['samples'], dtype=torch.float32)
        # label = torch.Tensor(self.data.iloc[idx]['labels'], dtype=torch.long)
        # return image, label
    
def dataset_to_dataframe(dataset):
    data = []
    for i in range(len(dataset)):
        img, label = dataset[i]
        data.append((img.view(-1).numpy(), label))

    df = pd.DataFrame(data, columns=['samples', 'labels'])
    return df

def get_even_class_indices(dataset, count :int):
    counts = {}
    index_dict = {}
    for i in range(len(dataset)):
        _, label = dataset[i]

        if label not in counts:
            counts[label] = 1
            index_dict[label] = [i]
        else:
            # if counts[label] < count:
            counts[label] += 1
            index_dict[label].append(i)

    all_indices = []
    for key, indices in index_dict.items():
        if len(indices) > count:
            all_indices.extend(np.random.choice(indices, size=count, replace=False))
        else:
            print(f'Class {key} had only {len(indices)} samples or less')
            all_indices.extend(indices)

    return all_indices

    
def get_conf_matrix_stats(y_pred, y_true):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f'Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)')
    print(f'Precision: {precision:.4f} ({precision * 100:.2f}%)')
    print(f'Recall: {recall:.4f} ({recall * 100:.2f}%)')
    print(f'F1-Score: {f1:.4f} ({f1 * 100:.2f}%)')
             

def get_latest_version(path):
    """
    Purpose:
    - Gather most recent version folder
    
    Arguments:
    - path (str): path to version folders
    """

    all_folders = [int(ele.replace("version_", "")) for ele in os.listdir(path)]

    return all_folders[np.argmax(all_folders)]

def get_training_results(path, path_plots, target_names, show_plots, save_plots):
    """
    Purpose:
    - Gather and show training analytics
    
    Arguments:
    - path (str): path to analytics file
    - target_names (list[str]): columns of analytics file to display
    """

    # Gather: All Analytics
    
    print("Loading path: %s" % path)

    data = pd.read_csv(path)

    # Display: Target Analytics
    
    for name in target_names:
    
        df = data.dropna(subset=[name])
        
        if "lr" in name:
            tag = "epoch"
            x_vals = list(range(df.shape[0]))
        else:
            tag = name.split("_")[-1]
            x_vals = df[tag]
        
        y_vals = df[name]
    
        title = "Plotting %s vs %s" % (name, tag)
        y_label = "%s" % name
        x_label = "%s" % tag
    
        plot_training(x_vals, y_vals, title, x_label, y_label, path_plots, show_plots, save_plots)