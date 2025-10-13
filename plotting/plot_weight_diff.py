import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())
from tikz import save2tikz

files = [
    "results/ECC/weights/new/3_day/sim_data_mpc_mal_indietro_mhe.pkl",
    "results/ECC/weights/new/3_day/sim_data_mpc_mal_opt_mhe.pkl",
]
names = ["mal indietro", "mal opt"]

lam = []
for name in files:
    try:
        with open(name, "rb") as f:
            data = pickle.load(f)
            lam.append(data["lam"])
    except FileNotFoundError:
        raise FileNotFoundError(f"File {name} not found. Run sim first.")

fig, ax = plt.subplots(1, 1, sharex=True)
ax.plot(
    np.array(
        [
            np.linalg.norm(lam[0][:, [k]] * np.ones((2, 72)) - lam[1][:, :, k])
            for k in range(lam[0].shape[1])
        ]
    ).squeeze()
)
ax.set_ylabel("lam sym")
ax.set_xlabel("Time (5 min steps)")

save2tikz(plt.gcf())

plt.show()
