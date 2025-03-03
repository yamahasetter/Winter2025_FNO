"""
	This script will iterate thru the contents of /data/raw/
	applying a block-mean filter to each simulation
	and store the results in /data/filtered
"""

# import required libraries
import os
import numpy as np
import shutil
import pandas as pd
import logging

# ---------------------------------------------
# set up log file
LOG_FILENAME = "filtering_logfile.log"
for handler in logging.root.handlers[:]:
	logging.root.removeHandler(handler)

# logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)    
logging.basicConfig(
	filename=LOG_FILENAME,
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logging.info('BEGINNING FILTERING PROCESS...')
# ---------------------------------------------

# get root dir and make prefix for directories
root = '../../data/filtered'

# go to filtered dir
os.chdir(root)

# this code was copied from https://github.com/keflavich/image_registration/blob/master/image_registration/fft_tools/downsample.py#L11
try:
    try:
        from numpy import nanmean
    except ImportError:
        from scipy.stats import nanmean
except ImportError as ex:
    print("Image-registration requires either numpy >= 1.8 or scipy.")
    raise ex

def downsample(myarr,factor,estimator=nanmean):
    """
    Downsample a 2D array by averaging over *factor* pixels in each axis.
    Crops upper edge if the shape is not a multiple of factor.

    This code is pure np and should be fast.

    keywords:
        estimator - default to mean.  You can downsample by summing or
            something else if you want a different estimator
            (e.g., downsampling error: you want to sum & divide by sqrt(n))
    """
    ys,xs = myarr.shape
    crarr = myarr[:ys-(ys % int(factor)),:xs-(xs % int(factor))]
    dsarr = estimator( np.concatenate([[crarr[i::factor,j::factor] 
        for i in range(factor)] 
        for j in range(factor)]), axis=0)
    return dsarr



# get names of all simulation in data/raw dir
simulations = os.listdir('../raw')

# ---------------------------------------------


"""
	Run the script:
"""

logging.info(f"Filtering starting_grid.csv ...")

# apply filter and save it in filtered dir
raw_data = pd.read_csv('../raw/starting_grid.csv', sep=',', header=None).to_numpy()
filtered_data = downsample(raw_data, 8) # 256/8 = 32
np.savetxt('starting_grid.csv', filtered_data, delimiter=",")
simulations.remove('starting_grid.csv')

try:
	# iterate over the sims in data/raw/
	for sim in simulations:

		logging.info('-'*60)
		logging.info(f"FILTERING: {sim}")

		# make new filtered dir for this sim
		os.mkdir(sim)

		# remember the path for the raw sim is data/raw/sim_#/
		raw_sim_path = '../raw/' + sim + '/'
		filtered_sim_path = sim + '/'

		# iterate over the timesteps
		for timeslice in os.listdir(raw_sim_path):
			# we want to copy the alpha values to this dir just in case we need them
			if timeslice == 'alpha.csv':
				logging.info("Copying alpha.csv ...")
				shutil.copy(raw_sim_path + 'alpha.csv', filtered_sim_path)
			else:
				logging.info(f"{timeslice}...")
				# apply filter to this sim and save it in filtered dir
				raw_data = pd.read_csv(raw_sim_path + timeslice, sep=',', header=None).to_numpy()

				filtered_data = downsample(raw_data, 8) # 256/8 = 32

				np.savetxt(filtered_sim_path + timeslice, filtered_data, delimiter=",")


	logging.info('-'*60)
	logging.info("SUCCESSFULLY COMPLETED DATA FILTERING")

except Exception as e:
	logging.error(e)













