import os
import pickle
from time import time

import casadi as cs
import numpy as np
from scipy.io import loadmat
from typing_extensions import Literal

from env import DHSSystem
from mpc.mpc import (
    DhsStorageMpcEnsemble,
)
from observer.mhe import MHE
from observer.trivial_observer import TrivialEnsembleObserver
from prediction_model.weights import MahalanobisWeighting, OptWeighting

ts = 5.0 * 60  # sampling time in seconds
N = 72  # prediction horizon
mhe_horizon = 72
num_inputs = 3
input_block = 24
storage_mass = 200000.0
if input_block > N / num_inputs:
    raise ValueError("NOOOOOO")
sim_len = 280 * 3  # simulation length in steps
USE_MHE = True

sim_type: Literal["rule", "w_av", "opt", "mal_indietro", "mal_avanti"] = "mal_indietro"
USE_MPC = sim_type != "rule"

loads_folder = "loads_4"
if os.name == "nt":
    fmu_filename = "simulation_model/dhs_win.fmu"
else:
    fmu_filename = "simulation_model/dhs_linux.fmu"

# loads input data for MPC
with open(f"sim_data/{loads_folder}/loads_min.pkl", "rb") as f:
    P_loads = pickle.load(f)
    P_loads = np.hstack(
        [P_loads, P_loads[:, -1][:, None].repeat(N + 10, axis=1)]
    )  # pad for N steps
# elec price data for MPC
with open(f"sim_data/{loads_folder}/elec_price.pkl", "rb") as f:
    elec_price = pickle.load(f)
    elec_price = np.hstack(
        [elec_price, elec_price[-1] * np.ones((N + 10,))]
    )  # pad for N steps
# temp limit data for MPC
with open(f"sim_data/{loads_folder}/limits.pkl", "rb") as f:
    data = pickle.load(f)
    Ts_min = data["Ts_min"]
    Tr_min = data["Tr_min"]
    Ts_min = np.hstack([Ts_min, Ts_min[-1] * np.ones((N + 10,))])  # pad for N steps
    Tr_min = np.hstack([Tr_min, Tr_min[-1] * np.ones((N + 10,))])  # pad for N steps
# load profiles for env, changing every second
with open(f"sim_data/{loads_folder}/loads_sec.pkl", "rb") as f:
    data = pickle.load(f)
    if isinstance(data, dict):
        P_loads_env = np.array([data[f"P_inp{i+1}"][:, 1] for i in range(5)])
    else:
        P_loads_env = data

if (
    P_loads.shape[1] < sim_len
    or elec_price.shape[0] < sim_len
    or Ts_min.shape[0] < sim_len
    or Tr_min.shape[0] < sim_len
):
    raise ValueError("Load profiles are shorter than simulation length")

env = DHSSystem(
    fmu_filename=fmu_filename,
    step_size=ts,
    P_loads=P_loads_env,
    elec_price=elec_price,
    storage_mass=storage_mass,
    T_s_min=Ts_min,
)

# prepare Malahanobis weighting
lam = cs.repmat([1, 0], 1, N)
if sim_type in ["mal_indietro", "mal_avanti"]:
    mat = loadmat("prediction_model/training_data/Power_lowconsumption.mat")
    data_ref_1 = np.vstack([mat["T"].T] + [mat[f"P{i}"].T for i in range(1, 6)]).T
    input_scaler_mat_1 = loadmat(
        "prediction_model/input_scaler_low.mat", squeeze_me=True
    )["input_scaler_file"].item()[0]
    mat = loadmat("prediction_model/training_data/Power_highconsumption.mat")
    data_ref_2 = np.vstack([mat["T"].T] + [mat[f"P{i}"].T for i in range(1, 6)]).T
    input_scaler_mat_2 = loadmat(
        "prediction_model/input_scaler_high.mat", squeeze_me=True
    )["input_scaler_file"].item()[0]
    mal_weighting = MahalanobisWeighting(
        data_ref=[data_ref_1, data_ref_2],
        scaling=[
            (input_scaler_mat_1["scale"].item(), input_scaler_mat_1["bias"].item()),
            (input_scaler_mat_2["scale"].item(), input_scaler_mat_2["bias"].item()),
        ],
    )

if sim_type == "opt":
    opt_weighting = OptWeighting(buffer_size=10, num_models=2)

# model weights
layers_path = [
    "prediction_model/layers_low.mat",
    "prediction_model/layers_high.mat",
]
input_scaler_path = [
    "prediction_model/input_scaler_low.mat",
    "prediction_model/input_scaler_high.mat",
]
output_scaler_path = [
    "prediction_model/output_scaler_low.mat",
    "prediction_model/output_scaler_high.mat",
]
mhe_class = MHE
mpc_class = DhsStorageMpcEnsemble
observer_class = TrivialEnsembleObserver

pars_init = {
    "T_ref": 75 * np.ones((5,)),
    "w": 1e2 / 5 * np.ones((1,)),
    "c_t": 1e1 * np.ones((1,)),
    "c_s": 1e1 * np.ones((1,)),
    "T_E_ref": 75 * np.ones((1,)),
    "V0": np.zeros((1,)),
    "f": np.zeros((3 + 7 + 5,)),
    "Q": np.zeros((3 + 7 + 5, 3 + 7 + 5)),
    "T_lim_off": np.zeros((5,)),
}
mpc_kwargs = {
    "dt": ts,
    "prediction_horizon": N,
    "num_inputs": num_inputs,
    "layers_path": layers_path,
    "input_scaler_path": input_scaler_path,
    "output_scaler_path": output_scaler_path,
    "pars_init": pars_init,
    "storage_mass": storage_mass,
}

mpc = mpc_class(
    **mpc_kwargs,
    mal_weighter=mal_weighting if sim_type in ["mal_indietro", "mal_avanti"] else None,
)
mhes = [
    mhe_class(
        prediction_horizon=mhe_horizon,
        layers_path=layers_path[i],
        input_scaler_path=input_scaler_path[i],
        output_scaler_path=output_scaler_path[i],
    )
    for i in range(len(layers_path))
]
ol_obs = observer_class(
    layers_path=layers_path,
    input_scaler_path=input_scaler_path,
    output_scaler_path=output_scaler_path,
)

y, _ = env.reset()
x_mhe = [mhe.x_est for mhe in mhes]

# simulation data
data = {
    "u": np.zeros((2, sim_len)),
    "y": np.zeros((len(env.outputs), sim_len)),
    "nn_input": np.zeros((1, sim_len)),
    "y_ol": (
        np.zeros((17 * len(layers_path), sim_len))
        if isinstance(layers_path, list)
        else np.zeros((17, sim_len))
    ),
    "y_mhe": (
        np.zeros((17 * len(layers_path), sim_len))
        if isinstance(layers_path, list)
        else np.zeros((17, sim_len))
    ),
    "r": np.zeros((1, sim_len)),
    "td": np.zeros((1, sim_len)),
    "P_loads": P_loads,
    "elec_price": elec_price,
    "Ts_min": Ts_min,
    "Tr_min": Tr_min,
    "infeasible_steps": np.zeros((1, sim_len)),
    "solver_time_mpc": np.zeros((1, sim_len)),
    "solver_time_mhe": np.zeros((1, sim_len)),
    "lam": np.zeros((2, N, sim_len)),
}

if num_inputs == 0:
    num_inputs = N
vals_guess = {
    "T_b_s": np.ones((1, num_inputs)) * 75,
    "q_E_h": np.ones((1, num_inputs)) * 0,
    "q_E_c": np.ones((1, num_inputs)) * 0,
    "T_i_s": np.ones((5, N)) * 75,
    "T_r": np.ones((1, N)) * 65,
    "q_r": np.ones((1, N)) * 2.0,
    "T_E": np.ones((1, N)) * 65,
    "s": np.zeros((5, 1)),
    "q_b": np.ones((1, N)) * 2.0,
    "T_s": np.ones((1, N)) * 75,
    "lam": np.vstack((np.ones((1, N)), np.zeros((1, N)))),
    "u": np.tile(np.array([[0], [0], [75.0]]), (1, num_inputs)),
}

solve_mpc = True
for k in range(sim_len):
    print(f"Time step {k+1}/{sim_len}")
    if k % input_block != 0 and k > 0:
        pass
    else:
        solve_mpc = True

    if k > mhe_horizon - 1:
        args = {
            "y": data["y"][:17, k - mhe_horizon : k],
            "T_s": data["nn_input"][:, k - mhe_horizon : k],
            "P_loads": P_loads[:, k - mhe_horizon : k],
        }
        for i, mhe in enumerate(mhes):
            sol = mhe.update_state(args)
            x_mhe[i] = mhe.x_est
            data["solver_time_mhe"][:, k] += sol.stats["t_wall_total"]

    if k > 0 and sim_type == "mal_indietro":
        d = np.vstack((NN_input, P_loads[:, k - 1].reshape(-1, 1)))
        lam = mal_weighting.compute_weights(d.squeeze())
        lam = cs.repmat(lam, 1, N)
    elif k > 0 and sim_type == "opt":
        lam = opt_weighting.compute_weights()
        lam = cs.repmat(lam, 1, N)

    if solve_mpc and USE_MPC:
        sol = mpc.solve(
            pars={
                "x": np.hstack(x_mhe),
                "P_loads": P_loads[:, k : k + N],
                "T_s_min": cs.repmat(Ts_min[k : k + N], 1, 5).T,
                "T_r_min": Tr_min[k : k + N],
                "elec_price": elec_price[k : k + N],
                "T_E_0": y[-3],
                "lam": lam,  # not used as a parameter if mpc optimizes over lam (i.e., mal_avanti)
            },
            vals0=vals_guess,
        )
        if not sol.success:
            print("MPC infeasible!")
            data["infeasible_steps"][0, k] = 1
            sol.vals = vals_guess
        else:
            data["solver_time_mpc"][:, k] = sol.stats["t_wall_total"]
            if "lam" in sol.vals.keys():
                lam = sol.vals["lam"].full().squeeze()
            solve_mpc = False
            print(f"Solver time mpc: {sol.stats['t_wall_total']} s")

        u = cs.vertcat(
            sol.vals["q_E_c"] - sol.vals["q_E_h"],
            sol.vals["T_b_s"],
        )

        vals_guess = dict(
            (
                name,
                (
                    (cs.horzcat(val[:, 1:], val[:, -1:]))
                    if name in ["q_E_h", "q_E_c", "T_b_s", "s"]
                    else cs.horzcat(
                        val[:, input_block:], cs.repmat(val[:, -1:], 1, input_block)
                    )
                ),
            )
            for (name, val) in sol.vals.items()
        )

    if not USE_MPC:
        if elec_price[k] < 0.125:
            q_b = 7.5
        elif elec_price[k] > 0.175:
            q_b = -7.5
        else:
            q_b = 0.0
        u = np.array([[q_b], [75.0]])

    print(f"Storage temp: {y[-3]} C")
    print(f"Control input: {u[:, 0].T}")

    start_time = time()
    y, r, _, _, _ = env.step(u[:, 0].full() if isinstance(u, cs.DM) else u[:, [0]])
    print(f"Step time: {time()-start_time} s")
    NN_input = y[18]
    x_ol, y_ol = ol_obs.step([NN_input] + P_loads[:, k].tolist())
    y_mhe = [mhe.step([NN_input] + P_loads[:, k].tolist())[1] for mhe in mhes]

    if sim_type == "opt":
        opt_weighting.add_observation(np.array(y[:17]), [y_ol[0], y_ol[1]])

    data["u"][:, [k]] = u[:, [0]]
    data["y"][:, k] = np.array(y)
    data["nn_input"][:, k] = NN_input
    data["y_ol"][:, k] = np.array(y_ol).flatten()
    data["y_mhe"][:, k] = np.array(y_mhe).flatten()
    data["r"][:, k] = r
    data["lam"][:, :, k] = lam

print(f"Total cost: {np.sum(data['r'])}")

with open(f"{sim_type}_{loads_folder}.pkl", "wb") as f:
    pickle.dump(data, f)
