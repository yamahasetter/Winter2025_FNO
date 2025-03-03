import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
# import torch
from .make_movie import make_movie

path_outputs = '/media/volume/alyns/Research/Findings/1/'

def plot_regressive_loss(loss, dir=path_outputs, title='Autoregressive FNO 2D Loss'):
    os.chdir(dir)

    time = np.arange(0,len(loss), 1)

    plt.plot(time, loss, label='loss', color='tab:blue', linestyle='--', marker='o', markersize=6, fillstyle='none')

    # Add x, y gridlines
    plt.grid(visible = True, color ='grey', 
            linestyle ='-.', linewidth = 0.5, 
            alpha = 0.6) 
    
    plt.xlabel("Time")
    plt.ylabel("MSE Loss")

    plt.yscale('log')

    plt.ylim((1e-10,1))
    
    size_x_ticks = len(loss) // 6
    plt.xticks(np.arange(0, len(loss) + size_x_ticks, size_x_ticks))

    plt.title(title)

    plt.savefig(title+ '.png', bbox_inches = "tight")


def plot_losses(dir=path_outputs):
    os.chdir(dir)

    losses = np.load('losses.npy')
    validation_losses = np.load('validation_losses.npy')

    epochs = np.arange(1,len(losses)+1, 1)

    if len(losses)<500:
        plt.plot(epochs, losses, label='loss', color='orange', linestyle='--', marker='o', markersize=6, fillstyle='none')
        plt.plot(epochs, validation_losses, label='validation\nloss', color='tab:blue', linestyle='--', marker='o', markersize=6, fillstyle='none')
    else:
        plt.plot(epochs, losses, label='loss', color='orange', linewidth=.5)
        plt.plot(epochs, validation_losses, label='validation\nloss', color='tab:blue', linewidth=.5)

    # Add x, y gridlines
    plt.grid(visible = True, color ='grey', 
            linestyle ='-.', linewidth = 0.5, 
            alpha = 0.6) 
    
    plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")

    plt.yscale('log')
    if len(losses)<500:
        plt.xticks(np.arange(0, len(losses)+50, 50))
    else:
        plt.xscale('log')
        # plt.xticks(np.arange(0, len(losses)+50, 50))

    plt.title('FNO 2D Loss')

    plt.savefig('losses.png', bbox_inches = "tight")


def compare(pred, true, name = 'compare'):
    fig, axs = plt.subplots(nrows=1, ncols=2,subplot_kw=dict(box_aspect=1))

    axs[0].pcolormesh(pred, norm=colors.SymLogNorm(linthresh=0.09, linscale=1, vmin=-1, vmax=1),cmap='RdBu_r')
    axs[0].set_title('Prediction')

    axs[1].pcolormesh(true, norm=colors.SymLogNorm(linthresh=0.09, linscale=1, vmin=-1, vmax=1),cmap='RdBu_r')
    axs[1].set_title('Ground Truth')

    fig.savefig(name + '.png')


def make_loss_movie(arr1, arr2, T, dt, title, dir=os.getcwd()):
    """
        Given 2 arrays with axes Time x X x Y
        compute the MSE error at each time step
        and make it into a movie
    """
    loss = torch.empty((T,32,32))
    for i in range(1, T):
        loss[i] = torch.mean((arr1[i]-arr2[i])**2).item()

    make_movie(loss.detach().numpy(), T, dt, dir+title)