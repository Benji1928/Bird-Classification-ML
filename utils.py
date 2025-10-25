import numpy as np
import matplotlib.pyplot as plt
import nntools as nt
import torch
from torch import nn



def show_img(image, ax=plt):
    # ax = axies object, plot area. ax=plt means default value for ax parameter

    image = image.to('cpu').numpy()
    # moves the tensor to the CPU because Matplotlib can only work with NumPy arrays that are on the CPU, not PyTorch tensors on the GPU.
    
    image = np.moveaxis(image, [0,1,2], [2,0,1])
    # Matplotlib expects different axes (H,W,C) compared to PyTorch (C,W,H)

    image = (image + 1) / 2
    # Unnormalize image

    image[image < 0] = 0
    image[image > 1] = 1

    i = ax.imshow(image)
    # matplotlib function that displays image

    ax.axis("off")

    return i
    


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def criterion(self, y, d):
        return self.cross_entropy(y, d)



class ClassificationStatsManager(nt.StatsManager):

    def __init__(self):
        super(ClassificationStatsManager, self).__init__()

    def init(self):
        super(ClassificationStatsManager, self).init()
        self.running_accuracy = 0

    def accumulate(self, loss, x, y, d):
        super(ClassificationStatsManager, self).accumulate(loss, x, y, d)
        _, l = torch.max(y, 1)
        self.running_accuracy += torch.mean((l == d).float())

    def summarize(self):
        loss = super(ClassificationStatsManager, self).summarize()
        accuracy = 100 * self.running_accuracy / self.number_update
        return {'loss': loss, 'accuracy': accuracy}




class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience: number of epochs to wait after last improvement
            min_delta: minimum change to consider as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # reset counter if improvement
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

            

def plot(exp, fig, axes):
    axes[0].clear()
    axes[1].clear()
    
    def to_float(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().item()
        return x

    # Training and Validation loss
    axes[0].plot([to_float(exp.history[k][0]['loss']) for k in range(exp.epoch)],
                 label="training loss")
    axes[0].plot([to_float(exp.history[k][1]['loss']) for k in range(exp.epoch)],
                 label="Validation loss")
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # Training and Validation accuracy
    axes[1].plot([to_float(exp.history[k][0]['accuracy']) for k in range(exp.epoch)],
                 label="training accuracy")
    axes[1].plot([to_float(exp.history[k][1]['accuracy']) for k in range(exp.epoch)],
                 label="Validation accuracy")
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(loc='lower right')

    # Add text for current epoch
    axes[0].text(0.95, 0.95, f"Epoch {exp.epoch}", 
                 transform=axes[0].transAxes, ha='right', va='top',
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.6))
    
    plt.tight_layout()
    fig.canvas.draw()