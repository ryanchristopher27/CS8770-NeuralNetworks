# Imports
import torch.optim as optim
import torch.nn as nn
import time
from tqdm import tqdm

# Local Imports
from utils.helpers import *
from utils.models import *
from utils.data import *
from utils.plots import *
from utils.hyperparameters import *


def main():

    dataset = 'MNIST'
    model_name = 'Balanced_MLP'
    train_batch_size = 200
    test_batch_size = 50
    device, on_gpu = cuda_setup()
    epochs = 2
    learning_rate = 0.01
    momentum = 0.9
    optimizer_name = "SGD"
    criterion_name = "CrossEntropy"
    display_plots = True
    
    single_model_evaluation(
        model_name=model_name,
        dataset=dataset,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        momentum=momentum,
        optimizer_name=optimizer_name,
        criterion_name=criterion_name,
        display_plots=display_plots,
        device=device,
        on_gpu=on_gpu,
    )


def single_model_evaluation(
    model_name: str,
    dataset: str,
    train_batch_size: int,
    test_batch_size: int,
    epochs: int,
    learning_rate: float,
    momentum: float,
    optimizer_name: str,
    criterion_name: str,
    display_plots: bool,
    device: str,
    on_gpu: bool,
):
    
    train_loader, test_loader, output_size, input_size = get_dataloader(data_name=dataset, model_name=model_name, train_batch_size=train_batch_size, test_batch_size=test_batch_size)

    num_classes = output_size 

    model = get_model(model_name=model_name, output_size=output_size, input_size=input_size)

    model.to(device) if on_gpu else None


    optimizer = get_optimizer(
        optimizer_name=optimizer_name,
        model=model,
        learning_rate=learning_rate,
        momentum=momentum,
    )
    criterion = get_criterion(criterion_name=criterion_name)

    train_loss_epoch = np.zeros(epochs)
    val_loss_epoch = np.zeros(epochs)

    train_acc_epoch = np.zeros(epochs)
    val_acc_epoch = np.zeros(epochs)

    train_correct_epoch = np.zeros(epochs)
    val_correct_epoch = np.zeros(epochs)

    # Train Model
    train_start_time = time.time()
    for epoch in range(0, epochs):
        print('==='*30)
        train(
            model=model,
            epoch=epoch, 
            dataloader=train_loader, 
            train_loss_epoch=train_loss_epoch,
            train_acc_epoch=train_acc_epoch,
            train_correct_epoch=train_correct_epoch,
            optimizer=optimizer,
            criterion=criterion,
            on_gpu=on_gpu,
            device=device
        )
        validation(
            model=model,
            epoch=epoch, 
            dataloader=test_loader,
            val_loss_epoch=val_loss_epoch,
            val_acc_epoch=val_acc_epoch,
            val_correct_epoch=val_correct_epoch,
            criterion=criterion,
            on_gpu=on_gpu,
            device=device
        )
    train_end_time = time.time()

    train_duration = train_end_time - train_start_time
    hours, remainder = divmod(train_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f'\nTime to Train: {int(hours)}:{int(minutes)}:{int(seconds)} seconds')


    # Test Model
    y_pred, y_true = test(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        on_gpu=on_gpu,
        device=device
    )

    # Plot Curves
    if display_plots:
        plot_loss_curve(train_loss_epoch, val_loss_epoch)
        plot_accuracy_curve(train_acc_epoch, val_acc_epoch)
        confusion_matrix(y_true, y_pred, num_classes=num_classes, num_samples=len(y_true), class_names=[str(i) for i in range(num_classes)])


def train(
    model, 
    epoch, 
    dataloader, 
    train_loss_epoch,
    train_acc_epoch,
    train_correct_epoch,
    optimizer, 
    criterion, 
    on_gpu, 
    device
):
    model.train()

    batch_count = len(dataloader)

    loss_accumulator = 0
    correct_accumulator = 0

    print(f'Train Epoch: {epoch+1}')

    for batch, (data, label) in tqdm(enumerate(dataloader), total=batch_count):
        if on_gpu:
            data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, label)
        loss.backward()

        optimizer.step()

        loss_accumulator += loss.item()

        pred = output.data.max(1, keepdim=True)[1]
        correct_accumulator += pred.eq(label.data.view_as(pred)).sum()


    train_loss_epoch[epoch] = loss_accumulator / batch_count
    train_acc_epoch[epoch] = correct_accumulator / len(dataloader.dataset)
    train_correct_epoch[epoch] = correct_accumulator

    print(f'Train Set: Average Batch Loss: {loss_accumulator/batch_count:.6f}, Accuracy: {correct_accumulator}/{len(dataloader.dataset)} ({100 * correct_accumulator / len(dataloader.dataset):.0f}%)')


def validation(
    model,
    epoch, 
    dataloader,
    val_loss_epoch,
    val_acc_epoch,
    val_correct_epoch,
    criterion,
    on_gpu,
    device
):
    model.eval()
    loss_accumulator = 0
    correct_accumulator = 0

    with torch.no_grad():
        for data, label in dataloader:
            if on_gpu:
                data, label = data.to(device), label.to(device)
            output = model(data)

            loss_accumulator += criterion(output, label)

            pred = output.data.max(1, keepdim=True)[1]
            correct_accumulator += pred.eq(label.data.view_as(pred)).sum()

    val_loss_epoch[epoch] = loss_accumulator / len(dataloader)
    val_acc_epoch[epoch] = correct_accumulator / len(dataloader.dataset)
    val_correct_epoch[epoch] = correct_accumulator

    print(f'\nVal Set: Loss: {loss_accumulator/len(dataloader.dataset):.6f}, Accuracy: {correct_accumulator}/{len(dataloader.dataset)} ({100 * correct_accumulator / len(dataloader.dataset):.0f}%)')


def test(
    model,
    dataloader,
    criterion,
    on_gpu,
    device
):
    model.eval()
    loss_accumulator = 0
    correct_accumulator = 0

    y_pred = np.zeros((len(dataloader) * dataloader.batch_size, 1))
    y_true = np.zeros(len(dataloader) * dataloader.batch_size)

    with torch.no_grad():
        for i, (data, label) in enumerate(dataloader):
            if on_gpu:
                data, label = data.to(device), label.to(device)
            output = model(data)

            loss_accumulator += criterion(output, label)

            pred = output.data.max(1, keepdim=True)[1]
            correct_accumulator += pred.eq(label.data.view_as(pred)).sum()

            y_pred[i * dataloader.batch_size : (i + 1) * dataloader.batch_size] = pred.cpu()
            y_true[i * dataloader.batch_size : (i + 1) * dataloader.batch_size] = label.cpu()

    test_loss = loss_accumulator/len(dataloader.dataset)
    test_accuracy = correct_accumulator/len(dataloader.dataset)
    test_correct = correct_accumulator

    print(f'\nTest Set: Loss: {test_loss:.6f}, Accuracy: {test_correct}/{len(dataloader.dataset)} ({100 * test_accuracy:.0f}%)')

    y_pred = y_pred.flatten()

    return y_pred, y_true




if __name__ == '__main__':
    main()