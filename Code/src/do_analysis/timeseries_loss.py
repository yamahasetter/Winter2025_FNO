import numpy as np
import matplotlib.pyplot as plt

"""
    This script compares predictive timeseries to ground-truth timeseries
"""

# load the ground truth timeseries
ground_truth = np.load('../../data/run4/0_filtered.npy')[:-1]

# load the approximations
lo_res_dedalus = np.load('../../Results/lo_res_dedalus_solution.npy')[:-1]
FNO_lead_10 = np.load('../../Models/singletons/7/Analysis/FNO_regressive_lo_res.npy')
FNO_lead_1 = np.load('../../Models/singletons/9/Analysis/FNO_regressive_lo_res.npy')
coupled_FNO = np.load('../../Models/coupled/12/Analysis/FNO_regressive_lo_res.npy')

# for each, compute the MSE at each time step
dedalus_error = np.mean((ground_truth-lo_res_dedalus)**2, axis=(1, 2))
FNO_error = np.mean((ground_truth-FNO_lead_10)**2, axis=(1, 2))
FNO1_error = np.mean((ground_truth-FNO_lead_1)**2, axis=(1, 2))
coupled_FNO_error = np.mean((ground_truth-coupled_FNO)**2, axis=(1, 2))

# plot results
time = np.arange(0,len(ground_truth), 1)

# plt.plot(time, dedalus_error, label='Numerical', color='tab:blue', linestyle='--', marker='o', markersize=6, fillstyle='none')
plt.plot(time[::50], dedalus_error[::50], label='Numerical', color='tab:blue', linestyle='--', marker='o', markersize=6, fillstyle='none')
# plt.plot(time, FNO_error, label='FNO', color='tab:orange', linestyle='--', marker='o', markersize=6, fillstyle='none')
plt.plot(time[::50], FNO_error[::50], label='10-step FNO', color='tab:orange', linestyle='--', marker='o', markersize=6, fillstyle='none')
plt.plot(time[::50], FNO1_error[::50], label='1-step FNO', color='tab:red', linestyle='--', marker='o', markersize=6, fillstyle='none')
plt.plot(time[::50], coupled_FNO_error[::50], label='Coupled FNO', color='tab:green', linestyle='--', marker='o', markersize=6, fillstyle='none')

# Add x, y gridlines
plt.grid(visible = True, color ='grey', 
        linestyle ='-.', linewidth = 0.5, 
        alpha = 0.6) 

plt.xlabel("Time")
plt.ylabel("MSE Loss")

plt.yscale('log')
plt.ylim(10e-8,100000)

size_x_ticks = len(ground_truth) // 10
plt.xticks(np.arange(0, len(ground_truth) + size_x_ticks, size_x_ticks))

plt.title("MSE of Filtered Solutions")
plt.legend()

plt.savefig("MSE of Filtered Solutions.png", bbox_inches = "tight")