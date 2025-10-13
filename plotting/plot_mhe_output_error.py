import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())
from tikz import save2tikz

file = "results/ECC/weights/new/3_day/sim_data_mpc_mal_opt_mhe.pkl"

try:
    with open(file, "rb") as f:
        data = pickle.load(f)
        y = data["y"]
        y_ol = data["y_ol"]
        y_mhe = data["y_mhe"]
except FileNotFoundError:
    raise FileNotFoundError(f"File {file} not found. Run sim first.")

fig, ax = plt.subplots(2, 1, sharex=True)
y_ol_ = np.split(y_ol, int(y_ol.shape[0] / 17), axis=0)
y_mhe_ = np.split(y_mhe, int(y_mhe.shape[0] / 17), axis=0)
for j in range(len(y_ol_)):
    err = np.array(
        [np.linalg.norm(y[:17, k] - y_ol_[j][:, k]) for k in range(y.shape[1])]
    )
    ax[j].plot(err, label=f"ol_{j+1}")
    err = np.array(
        [np.linalg.norm(y[:17, k] - y_mhe_[j][:, k]) for k in range(y.shape[1])]
    )
    ax[j].plot(err, label=f"mhe_{j+1}", linestyle="--")

# labels
ax[1].set_xlabel("Time (5 min steps)")
ax[0].set_ylabel("y err")
ax[1].set_ylabel("y err")

save2tikz(plt.gcf())

plt.show()
