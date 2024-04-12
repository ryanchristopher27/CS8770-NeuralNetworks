# Imports
import torch.nn as nn
import torch.nn.init as init


def get_model(
        model_name: str = 'Balanced_MLP',
        output_size: int = 10,
        input_size: int = int(28 * 28)
):

    if model_name == "Balanced_MLP":
        class Balanced_MLP(nn.Module):
            def __init__(self):
                super(Balanced_MLP, self).__init__()
                self.layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(input_size, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, output_size),
                    nn.Softmax(),
                )
            def forward(self, x):
                y_pred = self.layers(x)
                return y_pred
            
            def weight_init(self, m):
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=0, std=0.01)
                    init.constant_(m.bias, 0)            
            
        model = Balanced_MLP()
