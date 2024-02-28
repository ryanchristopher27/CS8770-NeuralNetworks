# Imports
import matplotlib.pyplot as plt


def plot_images_with_ground_truth_labels(test_loader):
    examples = enumerate(test_loader)
    _, (example_data, example_targets) = next(examples)

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def plot_loss_curve(train_loss, val_loss):
    fig = plt.figure()
    plt.plot(train_loss, color='blue')
    plt.plot(val_loss, color='red')
    plt.legend(['Train Loss', 'Validation Loss'], loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Per Epoch')
    plt.show()
    
def plot_accuracy_curve(train_acc, val_acc):
    fig = plt.figure()
    plt.plot(train_acc, color='blue')
    plt.plot(val_acc, color='red')
    plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Per Epoch')
    plt.show()

def plot_accuracy_curves_multiple_models(curves, models, x_label, y_label, title):
    fig = plt.figure()
    for curve in curves:
        plt.plot(curve)

    plt.legend(models, loc='upper right')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()