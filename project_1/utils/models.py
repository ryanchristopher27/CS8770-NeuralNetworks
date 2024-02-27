# Imports
import torch.nn as nn


def get_model(
        model_name: str = 'Balanced_MLP',
        output_size: int = 10,
        input_size: int = int(28 * 28)
):

    if model_name == "Balanced_MLP":
        class Main_MLP(nn.Module):
            def __init__(self):
                super(Main_MLP, self).__init__()
                self.layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(input_size, 2000),
                    nn.ReLU(),
                    nn.Linear(2000, 2000),
                    nn.ReLU(),
                    nn.Linear(2000, 256),
                    nn.ReLU(),
                    nn.Linear(256, output_size),
                )
            def forward(self, x):
                y_pred = self.layers(x)
                return y_pred
            
        model = Main_MLP()

    elif model_name == "Wide_MLP":
        class Wide_MLP(nn.Module):
            def __init__(self):
                super(Wide_MLP, self).__init__()
                self.layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(28*28, 28*28*4),
                    nn.ReLU(),
                    nn.Linear(28*28*4, 28*28*4),
                    nn.ReLU(),
                    nn.Linear(28*28*4, output_size),
                    nn.Softmax(),
                )
            
            def forward(self, x):
                y_pred = self.layers(x)
                return y_pred
            
        model = Wide_MLP()

    elif model_name == "Deep_MLP":
        class Deep_MLP(nn.Module):
            def __init__(self):
                super(Deep_MLP, self).__init__()
                self.layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(28*28, 28*28),
                    nn.ReLU(),
                    nn.Linear(28*28, 28*28),
                    nn.ReLU(),
                    nn.Linear(28*28, 28*28),
                    nn.ReLU(),
                    nn.Linear(28*28, 392),
                    nn.ReLU(),
                    nn.Linear(392, 392),
                    nn.ReLU(),
                    nn.Linear(392, output_size),
                    nn.Softmax(),
                )
            
            def forward(self, x):
                y_pred = self.layers(x)
                return y_pred
            
        model = Deep_MLP()

    elif model_name == 'Balanced_CNN':
        class Balanced_CNN(nn.Module):
            def __init__(self):
                super(Balanced_CNN, self).__init__()
                self.layers = nn.Sequential(
                    nn.Conv2d(1, 10, kernel_size = 5, padding = 2),
                    nn.MaxPool2d(2),
                    nn.ReLU(),
                    nn.Conv2d(10, 20, kernel_size = 5, padding = 2),
                    nn.Dropout2d(),
                    nn.MaxPool2d(2),
                    nn.ReLU(),
                    nn.Conv2d(20, 40, kernel_size = 5, padding = 2),
                    nn.Dropout2d(),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(40 * 7 * 7, 128),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, output_size),
                    nn.LogSoftmax(dim=1),
                )

            def forward(self, x):
                y_pred = self.layers(x)
                return y_pred
            
        model = Balanced_CNN()
    
    return model