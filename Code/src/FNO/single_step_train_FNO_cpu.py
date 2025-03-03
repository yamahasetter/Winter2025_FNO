from FNO.single_step_FNO_2D_cpu import *
from plotting.loss_analysis import plot_losses

################################################################
# configs + hyperparameters
################################################################

import os
print('cwd:', os.getcwd())

path_outputs = '../../Models/9/Analysis/'
data_path = '../../data/run4/'
load_model_path = '../../Models/9/'
save_model_path = '../../Models/9/'

# spectral loss params
lamda_reg = 0.2
wavenum_init = 50
wavenum_init_ydir = 10
ocean_grid = 32*32

# FNO params
modes = 10
width = 32
learning_rate = 0.0001

batch_size= 10 # how many pieces of training data
lead = 1 # how many timesteps the model needs to make an inference
mini_batch_size = 5 # the size of the mini-batch
epochs = 400 # number of training rounds to do
num_validation_samples = 2 # number of validation samples to get error on

# validation data
validation_input, validation_label = load_training_data(data_path, num_validation_samples, lead, selection=193)
# don't let validation into training data
excluding = [193]

model_name = f'FNO2D_EulerStep_Spectral_Loss_batch_sizes_{batch_size}_lr_{learning_rate}_modes_{modes}_wavenum_{wavenum_init}_lead_{lead}.pt'

net = FNO2d(modes, modes, width)

# load the parameters for the model
# net.load_state_dict(torch.load(load_model_path + 'FNO2D_Eulerstep_MSE_Loss_Randomized_Cahn_Hilliard_modes_12_wavenum_50_lead_10.pt'))

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)

# making a csv file to save losses
losses = []
validation_losses = []

logging.info('Model Created...\n Beginning Training...')

for epoch in tqdm(range(0, epochs)):  # loop over the dataset multiple times

    # running_loss = 0.0
    input_batch, label_batch = load_training_data(data_path, batch_size, lead, exclude=excluding, lim=193)

    # moving average losses
    moving_ave_loss = 0
    moving_ave_val_loss = 0

    for step in range(0, batch_size-mini_batch_size, mini_batch_size):
        # zero the parameter gradients
        optimizer.zero_grad()

        # load mini batches
        mini_input_batch, mini_label_batch = input_batch[step:step+mini_batch_size], label_batch[step:step+mini_batch_size]
        
        # forward + backward + optimize
        output = Eulerstep(net, mini_input_batch)
        loss = regular_loss(output, mini_label_batch)
        # loss = spectral_loss(output, mini_label_batch,wavenum_init,wavenum_init_ydir,lamda_reg,ocean_grid)

        loss.backward()
        optimizer.step()

        validation_output = Eulerstep(net, validation_input)
        validation_loss = regular_loss(validation_output, validation_label)
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

    if epoch % 100 == 0:
        learning_rate *= .9


logging.info('Finished Training')

# save the losses
# losses = losses.detach().cpu().numpy()
# validation_losses = validation_losses.detach().cpu().numpy()

np.save(path_outputs+'losses.npy',np.array(losses))
np.save(path_outputs+'validation_losses.npy', np.array(validation_losses))
logging.info('Losses Saved')

torch.save(net.state_dict(), save_model_path+model_name)
logging.info('FNO Model Saved')

plot_losses(path_outputs)