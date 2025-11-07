import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())
# from tikz import save2tikz

files = [
    "results/rule_loads_4.pkl",
    "results/w_av_loads_4.pkl",
    "results/opt_loads_4.pkl",
    "results/mal_indietro_loads_4.pkl",
    "results/mal_avanti_loads_4.pkl",
]
names = ["rule", "w_av", "opt", "mal indietro", "mal avanti"]

y = []
r = []
times = []
for name in files:
    try:
        with open(name, "rb") as f:
            data = pickle.load(f)
            y.append(data["y"])
            r.append(data["r"])
            times.append(data["solver_time_mpc"])
    except FileNotFoundError:
        raise FileNotFoundError(f"File {name} not found. Run sim first.")

Ts_min = data["Ts_min"][: y[0].shape[1]]
Tr_min = data["Tr_min"][: y[0].shape[1]]
elec_price = data["elec_price"][: y[0].shape[1]]
P_loads = data["P_loads"][:, : y[0].shape[1]]

fig, ax = plt.subplots(2, 1, sharex=True)
rs = [np.sum(r_) for r_ in r]
ax[0].bar(names, rs)
ax[0].set_ylim(np.min(rs) * 0.95, np.max(rs) * 1.05)
vols = []
for y_ in y:
    o = y_[[0, 3, 6, 9, 12]] - Ts_min
    o[o > 0] = 0
    o = np.linalg.norm(o, axis=1)
    vols.append(np.sum(o))
ax[1].bar(names, vols)
# ax[1].set_yscale("log")

# labels
ax[0].set_ylabel("Euro ($)")
ax[1].set_ylabel("Viol symbol")

time_means = [t[t != 0].mean() for t in times]
time_stds = [t[t != 0].std() for t in times]

# save2tikz(plt.gcf())

plt.show()
