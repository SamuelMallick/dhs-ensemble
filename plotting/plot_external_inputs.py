import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())
from tikz import save2tikz

loads_folder = "simulations/loads_4"
sim_len = 288 * 3

# load data
with open(f"{loads_folder}/loads_min.pkl", "rb") as f:
    P_loads = pickle.load(f)[:, :sim_len]
# elec price data for MPC
with open(f"{loads_folder}/elec_price.pkl", "rb") as f:
    elec_price = pickle.load(f)[:sim_len]
# temp limit data for MPC
with open(f"{loads_folder}/limits.pkl", "rb") as f:
    data = pickle.load(f)
    Ts_min = data["Ts_min"][:sim_len]
    Tr_min = data["Tr_min"][:sim_len]

fig, ax = plt.subplots(3, 1, sharex=True)
# get time in hours
time = np.arange(sim_len) * 5 / 60
ax[0].plot(time, -1e-3 * P_loads.T)
ax[1].plot(time, elec_price)
ax[2].plot(time, Ts_min, color="black")
ax[2].plot(time, Tr_min, color="black", linestyle="--")
ax[2].axhline(85, color="red")
ax[2].axhline(75, color="red", linestyle="--")

# labels
ax[2].set_xlabel("Time (hours)")
ax[0].set_ylabel("P_l (kW)")
ax[1].set_ylabel("c ($/kWh)")
ax[2].set_ylabel("T (°C)")

save2tikz(plt.gcf())
plt.show()
