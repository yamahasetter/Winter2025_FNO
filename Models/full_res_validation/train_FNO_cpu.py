import os
import sys
sys.path.append('/media/volume/alyns/Research/Code/src')

from plotting.loss_analysis import plot_losses

from FNO.variable_lead_FNO_2D_cpu import *
from plotting.loss_analysis import plot_losses
from torch.optim.lr_scheduler import ReduceLROnPlateau


# set up log file
# delete if already there:
if "logfile.log" in os.listdir():
     os.remove("logfile.log")

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

################################################################
# configs + hyperparameters
################################################################
print('cwd:', os.getcwd())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

run_number = 22
path_outputs = f'../../Models/singletons/{run_number}/Analysis/'
data_path = '../../data/run4/'
save_model_path = f'../../Models/singletons/{run_number}/'

# if dirs dont already exits, make them
if not os.path.isdir(save_model_path):
    os.mkdir(save_model_path)
if not os.path.isdir(path_outputs):
    os.mkdir(path_outputs)

# spectral loss params
lamda_reg = 0.4
wavenum_init = 48
wavenum_init_ydir = 48
ocean_grid = 32*32

# FNO params
modes = 14
width = 32
learning_rate = 1e-2

batch_size= 20 # how many pieces of training data
skip = 10 # give every `skip` timesteps from data
lead = 10 # how many timesteps the model needs to make an inference
mini_batch_size = 5 # the size of the mini-batch
epochs = 5000 # number of training rounds to do
num_validation_samples = 2 # number of validation samples to get error on

# validation data
validation_input, validation_label = load_training_data(data_path, num_validation_samples, lead*skip, selection=193)
validation_input, validation_label = validation_input[:,::skip,:,:].to(device), validation_label[:,::skip,:,:].to(device)
# validation_input, validation_label = validation_input.to(device), validation_label.to(device)

# don't let validation into training data
excluding = [193]

model_name = f'FNO2D_Eulerstep_MSE_Loss_Training_Ball_modes_{modes}_wavenum_{wavenum_init}_lead_{lead}_lr_{learning_rate}.pt'

net = FNO2d(modes, modes, width, lead=10).to(device)

# load the parameters for the model
# net.load_state_dict(torch.load(load_model_path + 'FNO2D_Eulerstep_MSE_Loss_Randomized_Cahn_Hilliard_modes_12_wavenum_50_lead_10.pt'))

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', min_lr=1e-6, factor=0.95, threshold=.8)

# making a csv file to save losses
losses = []
validation_losses = []

logging.info('Model Created...\n Beginning Training...')

best_epoch_loss = float('inf')
for epoch in tqdm(range(0, epochs)):  # loop over the dataset multiple times

    # running_loss = 0.0
    input_batch, label_batch = load_training_data_ball(data_path, batch_size, lead, var=1e-5, exclude=excluding, lim=193, skip=skip)
    # input_batch, label_batch =input_batch[:,::skip,:,:], label_batch[:,::skip,:,:]

    # moving average losses
    moving_ave_loss = 0
    moving_ave_val_loss = 0

    for step in range(0, batch_size-mini_batch_size, mini_batch_size):
        # zero the parameter gradients
        optimizer.zero_grad()

        # load mini batches
        mini_input_batch, mini_label_batch = input_batch[step:step+mini_batch_size].to(device), label_batch[step:step+mini_batch_size].to(device)
        
        # print("Mini batch size: ", mini_input_batch.shape)
        # forward + backward + optimize
        output = Eulerstep(net, mini_input_batch)
        
        # MSE + pointwise = loss
        loss = regular_loss(output[-1], mini_label_batch[-1]) + torch.max(torch.abs(output[-1]-mini_label_batch[-1]))
        # loss = spectral_loss(output, mini_label_batch,wavenum_init,wavenum_init_ydir,lamda_reg,ocean_grid)

        loss.backward()
        optimizer.step()

        validation_output = Eulerstep(net, validation_input)
        validation_loss = regular_loss(validation_output, validation_label) + torch.max(torch.abs(validation_output[-1]-validation_label))
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
    scheduler.step(validation_losses[-1])
    logging.info(f"Update learning rate: {scheduler.get_last_lr()[0]}")

    # save best model
    epoch_loss = validation_losses[-1]
    if epoch_loss < best_epoch_loss:
        # update best loss
        best_epoch_loss = epoch_loss

        # save the models that gave the best result
        torch.save(net.state_dict(), save_model_path+model_name)

        logging.info(f"Best model saved at epoch {epoch + 1} with loss {best_epoch_loss:.4f}")



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