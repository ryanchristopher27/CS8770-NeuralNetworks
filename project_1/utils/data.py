# Imports
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import MNIST

# Local Imports
from utils.helpers import *

def get_dataloader(
    data_name: str = 'MNIST',
    model_name: str = 'Balanced_MLP',
    train_batch_size: int = 50,
    test_batch_size: int = 50,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    if data_name == 'MNIST':
        train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)

        train_selected_indices = get_even_class_indices(train_dataset, count=5000)
        train_subset_sampler = SubsetRandomSampler(train_selected_indices)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=train_batch_size, 
            sampler=train_subset_sampler,
            worker_init_fn=torch.manual_seed(42),
        )


        test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
        test_selected_indices = get_even_class_indices(test_dataset, count=850)
        test_subset_sampler = SubsetRandomSampler(test_selected_indices)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=test_batch_size, 
            sampler=test_subset_sampler,
            worker_init_fn=torch.manual_seed(42),
        )

        output_size = 10
        input_size = 28 * 28

    return train_loader, test_loader, output_size, input_size