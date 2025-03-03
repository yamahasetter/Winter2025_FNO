from single_step_FNO_2D_cpu import *
import os 

############# Auto-regressive prediction #####################

print('cwd: ', os.getcwd())

load_model_path = '../../Models/9/'
data_path = '../../data/run4/'
path_outputs = '/media/volume/alyns/Research/Models/9/Analysis/'

# load the parameters for the model
modes = 10
width = 32
net = FNO2d(modes, modes, width)

# load the parameters for the model
net.load_state_dict(torch.load(load_model_path + 'FNO2D_EulerStep_Spectral_Loss_batch_sizes_20_lr_0.0001_modes_10_wavenum_50_lead_1.pt'))

# choose a run to compare regression against
simulation = load_this_run(data_path, lead=-1, sim_name='0_filtered.npy')

# normalize it - DONT DO THIS
# simulation, mean, var = normalize(simulation, yield_stats=True)

# get the number of time points and cut simulation in half... cause mem
T = len(simulation)
# simulation = simulation[:T]

# select the initial condition to give to the model
IC = simulation[:1].unsqueeze(0)

output = Eulerstep(net, IC)
prediction = torch.empty((T, 32, 32))
prediction[0] = simulation[0]

# run it forward
for i in tqdm(range(1, T)):
    output = Eulerstep(net, output)
    prediction[i] = output[0,-1]

# saving results
print('saving...')
os.chdir(path_outputs)

np.save('FNO_regressive_lo_res', prediction.detach().numpy())

# print('getting loss...')
# # compute loss at each timestep
# loss = torch.empty((T, 32, 32))
# for i in tqdm(range(1, T)):
#     loss[i] = simulation[i]-prediction[i]

# print('saving loss...')
# np.save('regressive_loss_lo_res',loss.detach().numpy())

print("making movies...")
# simulation = (simulation+mean)*var DONT DO THESE
# prediction = (prediction+mean)*var

make_movie(prediction.detach().numpy(), T, 1e-8, f'FNO_regressive_lo_res', dir=path_outputs)

# make a movie with the loss
# make_movie(loss.detach().numpy(), T, 1e-8, f'MSE Loss on {T} timesteps')
# make_loss_movie(simulation, prediction, T, 1e-8, f'MSE Loss on {T} timesteps', path_outputs)