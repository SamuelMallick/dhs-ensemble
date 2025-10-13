import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())
from tikz import save2tikz

files = [
    "results/ECC/weights/new/3_day/sim_data_mpc_rule_str.pkl",
    "results/ECC/weights/new/3_day/sim_data_mpc_50_50_mhe.pkl",
    "results/ECC/weights/new/3_day/sim_data_mpc_opt_mhe.pkl",
    "results/ECC/weights/new/3_day/sim_data_mpc_mal_indietro_mhe.pkl",
    "results/ECC/weights/new/3_day/sim_data_mpc_mal_opt_mhe.pkl",
]
names = ["rule str", "50_50", "qp", "mal indietro", "mal opt"]

y = []
for name in files:
    try:
        with open(name, "rb") as f:
            data = pickle.load(f)
            y.append(data["y"])
    except FileNotFoundError:
        raise FileNotFoundError(f"File {name} not found. Run sim first.")
Ts_min = data["Ts_min"][: y[0].shape[1]]

fix, ax = plt.subplots(1, 5, sharey=True)
for i, y_ in enumerate(y):
    ax[i].plot(y_[[0, 3, 6, 9, 12], -288:].T)
    ax[i].axhline(85, color="black", linestyle="--")
    ax[i].plot(Ts_min[-288:], color="black", linestyle="--")

# labels
ax[0].set_ylabel("T_s (°C)")

save2tikz(plt.gcf())

plt.show()
