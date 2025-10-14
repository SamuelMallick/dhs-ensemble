import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())
# from tikz import save2tikz

files = [
    "results/mal_indietro_loads_4.pkl",
    "results/mal_avanti_loads_4.pkl",
]
names = ["mal indietro", "mal avanti"]

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
            np.linalg.norm(lam[0][:, :, k] - lam[1][:, :, k])
            for k in range(lam[0].shape[2])
        ]
    ).squeeze()
)
ax.set_ylabel("lam sym")
ax.set_xlabel("Time (5 min steps)")

# save2tikz(plt.gcf())

plt.show()
