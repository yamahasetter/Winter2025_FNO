import numpy as np
from plotting.make_movie import make_movie
from plotting.loss_analysis import *

data_dir = '../../Results/'
# data_dir = '../../data/run4'
os.chdir(data_dir)

# plot_losses(data_dir)

prediction = np.load('lo_res_dedalus_solution.npy')
# loss = np.abs(np.load('regressive_loss_lo_res.npy'))

# print(prediction.shape)

T = len(prediction)

make_movie(prediction, T, 1e-8, f'dedalus_lo_res')
# make_movie(loss, T, 1e-8, f'Spectral Loss on {T} timesteps', 'jet')

# sim = np.load('0_filtered.npy')

# T = len(sim)

# make_movie(sim, T, 1e-8, f'Simulation 0')
