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

    elif model_name == "Wide_MLP":
        class Wide_MLP(nn.Module):
            def __init__(self):
                super(Wide_MLP, self).__init__()
                self.layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(input_size, 28*28*4),
                    nn.ReLU(),
                    nn.Linear(28*28*4, 28*28*4),
                    nn.ReLU(),
                    nn.Linear(28*28*4, output_size),
                    nn.Softmax(),
                )
            
            def forward(self, x):
                y_pred = self.layers(x)
                return y_pred
            
            def weight_init(self, m):
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=0, std=0.01)
                    init.constant_(m.bias, 0)   
            
        model = Wide_MLP()

    elif model_name == "One_Layer_MLP":
        class One_Layer_MLP(nn.Module):
            def __init__(self):
                super(One_Layer_MLP, self).__init__()
                self.layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(input_size, 28*28*8),
                    nn.ReLU(),
                    nn.Linear(28*28*8, output_size),
                    nn.Softmax(),
                )
            
            def forward(self, x):
                y_pred = self.layers(x)
                return y_pred
            
            def weight_init(self, m):
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=0, std=0.01)
                    init.constant_(m.bias, 0)   
            
        model = One_Layer_MLP()

    elif model_name == "Deep_MLP":
        class Deep_MLP(nn.Module):
            def __init__(self):
                super(Deep_MLP, self).__init__()
                self.layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(input_size, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, output_size),
                    nn.Softmax(),
                )
            
            def forward(self, x):
                y_pred = self.layers(x)
                return y_pred
            
            def weight_init(self, m):
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=0, std=0.01)
                    init.constant_(m.bias, 0)   
            
        model = Deep_MLP()
        
    elif model_name == 'Deep_Skinny_MLP':
        class Deep_Skinny_MLP(nn.Module):
            def __init__(self):
                super(Deep_Skinny_MLP, self).__init__()
                self.layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, output_size),
                    nn.Softmax(),
                )
            
            def forward(self, x):
                y_pred = self.layers(x)
                return y_pred
            
            def weight_init(self, m):
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=0, std=0.01)
                    init.constant_(m.bias, 0)   
            
        model = Deep_Skinny_MLP()

    elif model_name == "Large_MLP":
        class Large_MLP(nn.Module):
            def __init__(self):
                super(Large_MLP, self).__init__()
                self.layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(input_size, 28*28*10),
                    nn.ReLU(),
                    nn.Linear(28*28*10, 28*28*5),
                    nn.ReLU(),
                    nn.Linear(28*28*5, 28*28*2),
                    nn.ReLU(),
                    nn.Linear(28*28*2, 28*28),
                    nn.ReLU(),
                    nn.Linear(28*28, 128),
                    nn.ReLU(),
                    nn.Linear(128, output_size),
                    nn.Softmax(),
                )
            
            def forward(self, x):
                y_pred = self.layers(x)
                return y_pred
            
            def weight_init(self, m):
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=0, std=0.01)
                    init.constant_(m.bias, 0)   
            
        model = Large_MLP()

    elif model_name == "Square_MLP":
        class Square_MLP(nn.Module):
            def __init__(self):
                super(Square_MLP, self).__init__()
                self.layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(input_size, 28*28),
                    nn.ReLU(),
                    nn.Linear(28*28, 28*28),
                    nn.ReLU(),
                    nn.Linear(28*28, 28*28),
                    nn.ReLU(),
                    nn.Linear(28*28, 28*28),
                    nn.ReLU(),
                    nn.Linear(28*28, output_size),
                    nn.Softmax(),
                )
            
            def forward(self, x):
                y_pred = self.layers(x)
                return y_pred
            
            def weight_init(self, m):
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=0, std=0.01)
                    init.constant_(m.bias, 0)   
            
        model = Square_MLP()

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
            
            def weight_init(self, m):
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=0, std=0.01)
                    init.constant_(m.bias, 0)   
            
        model = Balanced_CNN()

    elif model_name == 'No_Max_Pooling_CNN':
        class No_Max_Pooling_CNN(nn.Module):
            def __init__(self):
                super(No_Max_Pooling_CNN, self).__init__()
                self.layers = nn.Sequential(
                    nn.Conv2d(1, 10, kernel_size = 5, padding = 2, stride=1),
                    nn.ReLU(),
                    nn.Conv2d(10, 20, kernel_size = 5, padding = 2),
                    nn.Dropout2d(),
                    nn.ReLU(),
                    nn.Conv2d(20, 40, kernel_size = 5, padding = 2),
                    nn.Dropout2d(),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(40 * 28 * 28, 128),
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
            
        model = No_Max_Pooling_CNN()

    elif model_name == 'Small_Kernel_CNN':
        class Small_Kernel_CNN(nn.Module):
            def __init__(self):
                super(Small_Kernel_CNN, self).__init__()
                self.layers = nn.Sequential(
                    nn.Conv2d(1, 10, kernel_size = 3, padding = 1, stride=1),
                    nn.MaxPool2d(2),
                    nn.ReLU(),
                    nn.Conv2d(10, 20, kernel_size = 3, padding = 1),
                    nn.Dropout2d(),
                    nn.MaxPool2d(2),
                    nn.ReLU(),
                    nn.Conv2d(20, 40, kernel_size = 3, padding = 1),
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
            
        model = Small_Kernel_CNN()

    elif model_name == 'High_Stride_CNN':
        class High_Stride_CNN(nn.Module):
            def __init__(self):
                super(High_Stride_CNN, self).__init__()
                self.layers = nn.Sequential(
                    nn.Conv2d(1, 10, kernel_size = 5, padding = 2, stride = 3),
                    nn.ReLU(),
                    nn.Conv2d(10, 20, kernel_size = 5, padding = 2, stride = 3),
                    nn.Dropout2d(),
                    nn.ReLU(),
                    nn.Conv2d(20, 40, kernel_size = 5, padding = 2, stride = 1),
                    nn.Dropout2d(),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(40 * 4 * 4, 128),
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
            
        model = High_Stride_CNN()

    elif model_name == 'Large_CNN':
        class Large_CNN(nn.Module):
            def __init__(self):
                super(Large_CNN, self).__init__()
                self.layers = nn.Sequential(
                    nn.Conv2d(1, 15, kernel_size = 5, padding = 2),
                    nn.ReLU(),
                    nn.Conv2d(15, 30, kernel_size = 5, padding = 2),
                    nn.Dropout2d(),
                    nn.ReLU(),
                    nn.Conv2d(30, 60, kernel_size = 5, padding = 2),
                    nn.Dropout2d(),
                    nn.MaxPool2d(2),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(60 * 14 * 14, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, output_size),
                    nn.LogSoftmax(dim=1),
                )

            def forward(self, x):
                y_pred = self.layers(x)
                return y_pred
            
        model = Large_CNN()

    elif model_name == 'Small_CNN':
        class Small_CNN(nn.Module):
            def __init__(self):
                super(Small_CNN, self).__init__()
                self.layers = nn.Sequential(
                    nn.Conv2d(1, 5, kernel_size = 5, padding = 2),
                    nn.MaxPool2d(2),
                    nn.ReLU(),
                    nn.Conv2d(5, 10, kernel_size = 5, padding = 2),
                    nn.Dropout2d(),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(10 * 14 * 14, 512),
                    nn.ReLU(),
                    nn.Linear(512, 64),
                    nn.ReLU(),
                    nn.Linear(64, output_size),
                    nn.LogSoftmax(dim=1),
                )

            def forward(self, x):
                y_pred = self.layers(x)
                return y_pred
            
        model = Small_CNN()
    
    return model