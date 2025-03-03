import os
import sys
sys.path.append('/media/volume/alyns/Research/Code/src')

from plotting.loss_analysis import plot_losses
from FNO.variable_lead_FNO_2D_cpu import *

############# Auto-regressive prediction #####################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('cwd: ', os.getcwd())

load_lead_10_model_path = '../../Models/singletons/14/'
load_lead_1_model_path = '../../Models/coupled/12/'
data_path = '../../data/run4/'
path_outputs = '../../Models/coupled/12/Analysis'

# load in pre-trained lead=10 model
logging.info('Loading 10-step FNO...')
FNO1 = FNO2d(17, 17, 32, lead=10).to(device)
FNO1.load_state_dict(torch.load(load_lead_10_model_path + 'FNO2D_Eulerstep_MSE_Loss_Training_Ball_modes_17_wavenum_48_lead_10_lr_1e-06.pt', map_location=device))
logging.info('Successfull.')

# and the short models
print('loading corrective network...')
logging.info(f'Builing {4} single-step FNO models...')
FNO2 = FNO2d(12, 12, 32, lead=1).to(device)
FNO3 = FNO2d(12, 12, 32, lead=1).to(device)
FNO4 = FNO2d(12, 12, 32, lead=1).to(device)
FNO5 = FNO2d(12, 12, 32, lead=1).to(device)

FNO2.load_state_dict(torch.load(load_lead_1_model_path + 'FNO2D_EulerStep_Regular_Loss_batch_sizes_10_lr_0.01_modes_12_wavenum_50_lead_14_lr_0.01_model_2.pt', map_location=device))
FNO3.load_state_dict(torch.load(load_lead_1_model_path + 'FNO2D_EulerStep_Regular_Loss_batch_sizes_10_lr_0.01_modes_12_wavenum_50_lead_14_lr_0.01_model_3.pt', map_location=device))
FNO4.load_state_dict(torch.load(load_lead_1_model_path + 'FNO2D_EulerStep_Regular_Loss_batch_sizes_10_lr_0.01_modes_12_wavenum_50_lead_14_lr_0.01_model_4.pt', map_location=device))
FNO5.load_state_dict(torch.load(load_lead_1_model_path + 'FNO2D_EulerStep_Regular_Loss_batch_sizes_10_lr_0.01_modes_12_wavenum_50_lead_14_lr_0.01_model_5.pt', map_location=device))

logging.info('Successfull.')

# choose a run to compare regression against
simulation = load_this_run(data_path, lead=-1, sim_name='0_filtered.npy')

# get the number of time points
T = len(simulation)

print("Shape of FNO1_input: ", simulation[:10].unsqueeze(0).shape)
print("Shape of FNO2_input: ", simulation[10:11].unsqueeze(0).shape)

# start the model
FNO1_output = Eulerstep(FNO1, simulation[:10].unsqueeze(0).to(device))
FNO2_output = Eulerstep(FNO2, simulation[10:11].unsqueeze(0).to(device))
FNO3_output = Eulerstep(FNO3, simulation[11:12].unsqueeze(0).to(device))
FNO4_output = Eulerstep(FNO4, simulation[12:13].unsqueeze(0).to(device))
FNO5_output = Eulerstep(FNO5, simulation[13:14].unsqueeze(0).to(device))

output = FNO1_output + FNO2_output + FNO3_output + FNO4_output + FNO5_output
print("Output shape: ", output.shape)

prediction = torch.empty((T, 32, 32))
prediction[0] = simulation[0]

# run it forward
with torch.no_grad():
    for i in tqdm(range(1, T)):
        # regression = feed the model its own output

        FNO1_output = Eulerstep(FNO1, output)
        FNO2_output = Eulerstep(FNO2, output[0, -4:-3].unsqueeze(1))
        FNO3_output = Eulerstep(FNO3, output[0, -3:-2].unsqueeze(1))
        FNO4_output = Eulerstep(FNO4, output[0, -2:-1].unsqueeze(1))
        FNO5_output = Eulerstep(FNO5, output[0, -1:].unsqueeze(1))

        output = FNO1_output + FNO2_output + FNO3_output + FNO4_output + FNO5_output

        prediction[i] = output[0, -1]

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

# print("making movies...")
# make_movie(prediction.detach().numpy(), T, 1e-8, f'FNO_regressive_lo_res', dir=path_outputs)

# make a movie with the loss
# make_movie(loss.detach().numpy(), T, 1e-8, f'MSE Loss on {T} timesteps')
# make_loss_movie(simulation, prediction, T, 1e-8, f'MSE Loss on {T} timesteps', path_outputs)