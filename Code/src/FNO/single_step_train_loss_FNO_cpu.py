"""
    Train FNO w/ lead=1 to recover loss from low-res numerical result and lead=10 model
"""

# import
import os
print('cwd:', os.getcwd())

import sys
sys.path.append('/media/volume/alyns/Research/Code/src')

from plotting.loss_analysis import plot_losses
from FNO.variable_lead_FNO_2D_cpu import *
from torch.optim.lr_scheduler import ReduceLROnPlateau





################################################################
# configs + hyperparameters
################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

run_number = 14

data_path = '../../data/run4/'
load_lead_10_model_path = '../../Models/singletons/14/'
load_lead_1_model_path = '../../Models/singletons/10/'
save_model_path = f'../../Models/coupled/{run_number}/'
path_outputs = f'../../Models/coupled/{run_number}/Analysis/'


# if dirs dont already exits, make them
if not os.path.isdir(save_model_path):
    os.mkdir(save_model_path)
if not os.path.isdir(path_outputs):
    os.mkdir(path_outputs)


# define the number of single-step models to train
num_correction_models = 4

# load in pre-trained lead=10 model
logging.info('Loading 10-step FNO...')
FNO1 = FNO2d(17, 17, 32, lead=10).to(device)
FNO1.load_state_dict(torch.load(load_lead_10_model_path + 'FNO2D_Eulerstep_MSE_Loss_Training_Ball_modes_17_wavenum_48_lead_10_lr_1e-06.pt', map_location=device))
logging.info('Successfull.')


# spectral loss params
lamda_reg = 0.2
wavenum_init = 50
wavenum_init_ydir = 10
ocean_grid = 32*32

# FNO params
modes = 12
width = 32
learning_rate = .01

batch_size= 10 # how many pieces of training data
FNO1_lead = 10 # how many timesteps FNO1 needs to make an inference
FNO_forcing = num_correction_models # how many future timesteps to force with
lead = FNO1_lead + FNO_forcing
mini_batch_size = 5 # the size of the mini-batch
epochs = 50000 # number of training rounds to do
num_validation_samples = 2 # number of validation samples to get error on

# validation data
validation_input, validation_label = load_training_data(data_path, num_validation_samples, lead, selection=193)

# FNO1_validation data
FNO1_validation_input, FNO1_validation_label = validation_input[:, :FNO1_lead, :, :].to(device), validation_label[:, FNO1_lead:FNO1_lead+1, :, :].to(device)

# FNO1 validation prediction
FNO1_validation_output = Eulerstep(FNO1, FNO1_validation_input)[:, -1:, :, :]

# make validation data for all num_correction_models
FNO2_validation_input, FNO2_validation_label = validation_input[:, FNO1_lead:FNO1_lead+1, :, :].to(device), validation_label[:, FNO1_lead+1:FNO1_lead+2, :, :]
FNO3_validation_input, FNO3_validation_label = validation_input[:, FNO1_lead+1:FNO1_lead+2, :, :].to(device), validation_label[:, FNO1_lead+2:FNO1_lead+3, :, :]
FNO4_validation_input, FNO4_validation_label = validation_input[:, FNO1_lead+2::FNO1_lead+3, :, :].to(device), validation_label[:, FNO1_lead+3:FNO1_lead+4, :, :]
FNO5_validation_input, FNO5_validation_label = validation_input[:, FNO1_lead+3:FNO1_lead+4, :, :].to(device), validation_label[:, FNO1_lead+4:FNO1_lead+5, :, :]

# don't let validation into training data
excluding = [193]

logging.info(f'Builing {num_correction_models} single-step FNO models...')
FNO2 = FNO2d(modes, modes, width, lead=1).to(device)
FNO3 = FNO2d(modes, modes, width, lead=1).to(device)
FNO4 = FNO2d(modes, modes, width, lead=1).to(device)
FNO5 = FNO2d(modes, modes, width, lead=1).to(device)
logging.info('Successfull.')


FNO2_optimizer = torch.optim.Adam(FNO2.parameters(), lr=learning_rate, weight_decay=1e-4)
FNO3_optimizer = torch.optim.Adam(FNO3.parameters(), lr=learning_rate, weight_decay=1e-4)
FNO4_optimizer = torch.optim.Adam(FNO4.parameters(), lr=learning_rate, weight_decay=1e-4)
FNO5_optimizer = torch.optim.Adam(FNO5.parameters(), lr=learning_rate, weight_decay=1e-4)


FNO2_scheduler = ReduceLROnPlateau(FNO2_optimizer, mode='min',min_lr=1e-6, factor=0.9, threshold=.2)
FNO3_scheduler = ReduceLROnPlateau(FNO3_optimizer, mode='min',min_lr=1e-6, factor=0.9, threshold=.2)
FNO4_scheduler = ReduceLROnPlateau(FNO4_optimizer, mode='min',min_lr=1e-6, factor=0.9, threshold=.2)
FNO5_scheduler = ReduceLROnPlateau(FNO5_optimizer, mode='min',min_lr=1e-6, factor=0.9, threshold=.2)


# making a csv file to save losses
losses = []
validation_losses = []

logging.info('Model Created...\n Beginning Training...')

best_epoch_loss = float('inf')
for epoch in tqdm(range(0, epochs)):  # loop over the dataset multiple times

    # load batch + label data for all num_correction_models
    input_batch, label_batch = load_training_data(data_path, batch_size, lead, exclude=excluding, lim=193)

    # FNO1 input data
    FNO1_input, FNO1_label = input_batch[:, :FNO1_lead, :, :].to(device), label_batch[:, FNO1_lead:FNO1_lead+1, :, :].to(device)

    # FNO1 prediction
    FNO1_output = Eulerstep(FNO1, FNO1_input)[:, -1:, :, :]

    # data for all num_correction_models
    FNO2_input_batch = input_batch[:, FNO1_lead:FNO1_lead+1, :, :]
    FNO3_input_batch = input_batch[:, FNO1_lead+1:FNO1_lead+2, :, :]
    FNO4_input_batch = input_batch[:, FNO1_lead+2::FNO1_lead+3, :, :]
    FNO5_input_batch = input_batch[:, FNO1_lead+3:FNO1_lead+4, :, :]

    # moving average losses
    moving_ave_loss = 0
    moving_ave_val_loss = 0

    for step in range(0, batch_size-mini_batch_size, mini_batch_size):
        # zero the parameter gradients
        FNO2_optimizer.zero_grad()
        FNO3_optimizer.zero_grad()
        FNO4_optimizer.zero_grad()
        FNO5_optimizer.zero_grad()

        # load mini batches
        FNO2_mini_input_batch = FNO2_input_batch[step:step+mini_batch_size].to(device)
        FNO3_mini_input_batch = FNO3_input_batch[step:step+mini_batch_size].to(device)
        FNO4_mini_input_batch = FNO4_input_batch[step:step+mini_batch_size].to(device)
        FNO5_mini_input_batch = FNO5_input_batch[step:step+mini_batch_size].to(device)
        
        # forward 
        FNO2_output = Eulerstep(FNO2, FNO2_mini_input_batch) 
        FNO3_output = Eulerstep(FNO3, FNO3_mini_input_batch) 
        FNO4_output = Eulerstep(FNO4, FNO4_mini_input_batch) 
        FNO5_output = Eulerstep(FNO5, FNO5_mini_input_batch) 
        
        # concat to get final out
        output = FNO1_output[step:step+mini_batch_size] + FNO2_output + FNO3_output + FNO4_output + FNO5_output
        
        # the label is just the last element of the label batch for FNO1
        mini_label_batch = FNO1_label[step:step+mini_batch_size].to(device)

        # compute the loss
        loss = regular_loss(output, mini_label_batch)
        # loss = spectral_loss(output, mini_label_batch,wavenum_init,wavenum_init_ydir,lamda_reg,ocean_grid)

        # backprop
        loss.backward(retain_graph=True)
        FNO2_optimizer.step()
        FNO3_optimizer.step()
        FNO4_optimizer.step()
        FNO5_optimizer.step()

        # do validation
        validation_output = (
                Eulerstep(FNO2, FNO2_validation_input) 
                + Eulerstep(FNO3, FNO3_validation_input) 
                + Eulerstep(FNO4, FNO4_validation_input) 
                + Eulerstep(FNO5, FNO5_validation_input) 
                + FNO1_validation_output
                                )
        
        validation_loss = regular_loss(validation_output, FNO1_validation_label)
        # validation_loss = spectral_loss(validation_output, validation_label,wavenum_init,wavenum_init_ydir,lamda_reg,ocean_grid)

        # moving average losses
        moving_ave_loss += loss.item()
        moving_ave_val_loss += validation_loss.item()

        # running_loss = 0.0
        if step % 100 == 0:    # print every 2000 mini-batches
            logging.info(f'[{epoch + 1}, {step + 1}, {loss.item()}]')
            logging.info(f'[{epoch + 1}, {step + 1}, {validation_loss.item()}]')

    # moving average loss
    losses.append(moving_ave_loss/batch_size)
    validation_losses.append(moving_ave_val_loss/batch_size)

    # update scheduler
    FNO2_scheduler.step(validation_losses[-1])
    FNO3_scheduler.step(validation_losses[-1])
    FNO4_scheduler.step(validation_losses[-1])
    FNO5_scheduler.step(validation_losses[-1])
    logging.info(f"Update learning rate: {FNO2_scheduler.get_last_lr()[0]}")

    # save best model
    epoch_loss = validation_losses[-1]
    if epoch_loss < best_epoch_loss:
            # update best loss
            best_epoch_loss = epoch_loss

            # save the models that gave the best result
            model_name = f'FNO2D_EulerStep_Regular_Loss_batch_sizes_{batch_size}_lr_{learning_rate}_modes_{modes}_wavenum_{wavenum_init}_lead_{lead}_lr_{learning_rate}_model_{2}.pt'
            torch.save(FNO2.state_dict(), save_model_path+model_name)

            model_name = f'FNO2D_EulerStep_Regular_Loss_batch_sizes_{batch_size}_lr_{learning_rate}_modes_{modes}_wavenum_{wavenum_init}_lead_{lead}_lr_{learning_rate}_model_{3}.pt'
            torch.save(FNO3.state_dict(), save_model_path+model_name)

            model_name = f'FNO2D_EulerStep_Regular_Loss_batch_sizes_{batch_size}_lr_{learning_rate}_modes_{modes}_wavenum_{wavenum_init}_lead_{lead}_lr_{learning_rate}_model_{4}.pt'
            torch.save(FNO4.state_dict(), save_model_path+model_name)

            model_name = f'FNO2D_EulerStep_Regular_Loss_batch_sizes_{batch_size}_lr_{learning_rate}_modes_{modes}_wavenum_{wavenum_init}_lead_{lead}_lr_{learning_rate}_model_{5}.pt'
            torch.save(FNO5.state_dict(), save_model_path+model_name)

            logging.info(f"Best model saved at epoch {epoch + 1} with loss {best_epoch_loss:.4f}")




logging.info('Finished Training')

# save the losses
# losses = losses.detach().cpu().numpy()
# validation_losses = validation_losses.detach().cpu().numpy()

np.save(path_outputs+'losses.npy',np.array(losses))
np.save(path_outputs+'validation_losses.npy', np.array(validation_losses))
logging.info('Losses Saved')


logging.info('FNO Models Saved')

plot_losses(path_outputs)