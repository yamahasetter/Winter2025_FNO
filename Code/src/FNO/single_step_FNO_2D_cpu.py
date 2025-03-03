import torch.nn.functional as F

# from utilities3 import * --------> I don't think I need this

from timeit import default_timer
from tqdm import tqdm
import time
import logging
import numpy as np
import torch
logging.info(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#from torchinfo import summary
import sys
import os
import netCDF4 as nc
# import hdf5storage

# loss plotting library
from plotting.make_movie import make_movie
from plotting.loss_analysis import *

# set up log file
LOG_FILENAME = "logfile.log"
for handler in logging.root.handlers[:]:
	logging.root.removeHandler(handler)

# logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)    
logging.basicConfig(
	filename=LOG_FILENAME,
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Force logging to use local time
logging.Formatter.converter = time.localtime

logging.info('Generating Simulations...')

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cpu")


################################################################
# fourier layer
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic

        self.p = nn.Linear(3, self.width) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        # x = x.permute(0, 2, 3, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)


################################################################
# loss and multistep methods
################################################################


def regular_loss(output, target):
 loss = torch.sum((output-target)**2)
 return loss


def ocean_loss(output, target, ocean_grid):

    loss = (torch.sum((output-target)**2))/ocean_grid
    return loss


def spectral_loss(output, target,wavenum_init,wavenum_init_ydir,lamda_reg,ocean_grid):

    loss1 = torch.sum((output-target)**2) #/ocean_grid
    # loss1 = torch.abs((output-target))/ocean_grid

    out_fft = torch.mean(torch.abs(torch.fft.rfft(output[:,:,:,0],dim=2)),dim=1)
    target_fft = torch.mean(torch.abs(torch.fft.rfft(target[:,:,:,0],dim=2)),dim=1)

    out_fft_ydir = torch.mean(torch.abs(torch.fft.rfft(output[:,:,:,0],dim=1)),dim=2)
    target_fft_ydir = torch.mean(torch.abs(torch.fft.rfft(target[:,:,:,0],dim=1)),dim=2)

    # loss2 = torch.mean(torch.abs(out_fft[:,0:wavenum_init]-target_fft[:,0:wavenum_init]))
    # loss2_ydir = torch.mean(torch.abs(out_fft_ydir[:,0:wavenum_init_ydir]-target_fft_ydir[:,0:wavenum_init_ydir]))

    loss2 = torch.mean(torch.abs(out_fft[:,wavenum_init:]-target_fft[:,wavenum_init:]))
    loss2_ydir = torch.mean(torch.abs(out_fft_ydir[:,wavenum_init_ydir:]-target_fft_ydir[:,wavenum_init_ydir:]))

    loss = (1-lamda_reg)*loss1 + 0.5*lamda_reg*loss2 + 0.5*lamda_reg*loss2_ydir

    return loss


def RK4step(net,input_batch):
 output_1 = net(input_batch)
 output_2= net(input_batch+0.5*output_1)
 output_3 = net(input_batch+0.5*output_2)
 output_4 = net(input_batch+output_3)

 return input_batch + (output_1+2*output_2+2*output_3+output_4)/6


def Eulerstep(net,input_batch):
 output_1 = net(input_batch)
 return input_batch + output_1


def directstep(net,input_batch):
  output_1 = net(input_batch)
  return output_1


################################################################
# data loader / handler
################################################################

def normalize(arr, yield_stats=False):
    """
        make arr 0-mean variance-1
    """
    if not yield_stats:
        return (arr-arr.mean())/arr.std()
    else:
        return (arr-arr.mean())/arr.std(), arr.mean(), arr.std()


def load_training_data(data_dir, trainN, lead, selection=None, exclude=None, lim=199):
    """
        arguments:
            `trainN`, `lead`, `data_dir`

        go to `data_dir` and choose a random simulation.
        choose `trainN` random indices.
        starting at the random indices, select the next `lead`
        elements of the simulation and save to input, then
        save the `lead`+1 element to label.
    """
    # choose a simulation at random
    if selection==None:
        if exclude==None:
            simulation = data_dir + str(np.random.randint(0,lim)) + '_filtered.npy'
            simulation = np.load(simulation)
        else:
            simulation = data_dir + str(np.random.choice([i for i in range(0,lim) if i not in exclude])) + '_filtered.npy'
            simulation = np.load(simulation)
    else:
        simulation = data_dir + str(selection) + '_filtered.npy'
        simulation = np.load(simulation)

    # for input and label data
    input_ = np.ndarray((trainN, lead, 32,32))
    label = np.ndarray((trainN, lead, 32,32))
    sim_length = simulation.shape[0]

    for idx in range(trainN):
        # choose a random position between 0 and len(simulation-lead-1) in simulation array
        start = np.random.randint(0, sim_length-lead-1)
        input_[idx] = simulation[start:start+lead]
        # label[idx, 0] = simulation[start+lead+1]
        label[idx] = simulation[start+1:start+lead+1]

    # normalize
    input_, label = normalize(input_), normalize(label)

    # torch tensors
    return torch.from_numpy(input_).float(), torch.from_numpy(label).float()


def load_this_run(data_dir, lead, sim_name):
    """
        go to data_dir, choose sim_name and gather the data
        there, return first lead elements.
    """
    simulation = np.load(data_dir + sim_name)

    return torch.from_numpy(simulation[:lead]).float()

################################################################
# configs + hyperparameters
################################################################

# path_outputs = '/media/volume/alyns/Research/Findings/1/'
# data_path = '../../data/run2/'
# model_path = '../../Models/1/'

# lamda_reg =0.2
# wavenum_init=50 
# wavenum_init_ydir=50

# modes = 16
# width = 32

# learning_rate = 0.001

# trainN=20 # how many pieces of training data
# lead = 10 # how many timesteps the model needs to make an inference
# batch_size = 5 # the size of the mini-batch
# epochs = 100 # number of training rounds to do
# num_validation_samples = 2 # number of validation samples to get error on

# # validation data
# validation_input, validation_label = load_training_data(data_path, num_validation_samples, lead, selection=199)
# # don't let validation into training data
# excluding = [199]

# model_name = f'FNO2D_Eulerstep_MSE_Loss_Randomized_Cahn_Hilliard_modes_{modes}_wavenum_{wavenum_init}_lead_{lead}.pt'


################################################################
# training and evaluation
################################################################

# net = FNO2d(modes, modes, width)
# optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)

# # making a csv file to save losses
# losses = []
# validation_losses = []

# logging.info('Model Created...\n Beginning Training...')

# for epoch in tqdm(range(0, epochs)):  # loop over the dataset multiple times

#     # running_loss = 0.0
#     input_batch, label_batch = load_training_data(data_path, trainN, lead, exclude=excluding)

#     for step in range(0, trainN-batch_size, batch_size):
#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # load mini batches
#         mini_input_batch, mini_label_batch = input_batch[step:step+batch_size], label_batch[step:step+batch_size]
        
#         # forward + backward + optimize
#         output = Eulerstep(net, mini_input_batch)
#         loss = regular_loss(output, mini_label_batch)

#         loss.backward()
#         optimizer.step()

#         validation_output = Eulerstep(net, validation_input)
#         validation_loss = regular_loss(validation_output, validation_label)

#         # running_loss = 0.0
#         if step % 100 == 0:    # print every 2000 mini-batches
#             logging.info('[%d, %5d] loss: %.3f' %
#                 (epoch + 1, step + 1, loss.item()))
#             logging.info('[%d, %5d] val_loss: %.3f' %
#                 (epoch + 1, step + 1, validation_loss.item()))
            
#     losses.append(loss.item())
#     validation_losses.append(validation_loss.item())

# logging.info('Finished Training')

# # save the losses
# # losses = losses.detach().cpu().numpy()
# # validation_losses = validation_losses.detach().cpu().numpy()

# np.save(path_outputs+'losses.npy',np.array(losses))
# np.save(path_outputs+'validation_losses.npy', np.array(validation_losses))
# logging.info('Losses Saved')

# torch.save(net.state_dict(), model_path+model_name)
# logging.info('FNO Model Saved')


# from loss_analysis import plot_losses

# plot_losses()

############# Auto-regressive prediction #####################

# # load the parameters for the model
# net.load_state_dict(torch.load(model_path + 'best.pt'))

# # choose a run to compare regression against
# simulation = load_this_run(data_path, lead=-1, sim_name='0.npy')

# # normalize it
# simulation, mean, var = normalize(simulation, yield_stats=True)

# # get the number of time points
# T = len(simulation)//2

# # cut simulation in half...
# simulation = simulation[:T]

# # select the initial condition to give to the model
# IC = simulation[:10].unsqueeze(0)

# output = Eulerstep(net, IC)
# prediction = torch.empty((T, 256, 256))
# prediction[0] = simulation[0]

# # run it forward
# for i in tqdm(range(1, T)):
#     output = Eulerstep(net, output)
#     prediction[i] = output[0,-1]


# # compute loss at each timestep
# print("making movire...")
# # simulation = (simulation+mean)*var
# prediction = (prediction+mean)*var

# make_movie(prediction.detach().numpy(), T, 1e-5, f'Resolution Boosted Prediction')


# loss = torch.empty((T))
# for i in tqdm(range(1, T)):
#     loss[i] = torch.mean((simulation[i]-prediction[i])**2).item()

# print('saving loss...')
# np.save('long_time_loss_hi_res',loss.detach().numpy())

# make a movie with the loss
# make_movie(loss.detach().numpy(), T, 1e-5, f'MSE Loss on {T} timesteps')
# make_loss_movie(simulation, prediction, T, 1e-5, f'MSE Loss on {T} timesteps')