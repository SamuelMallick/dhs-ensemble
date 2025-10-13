import os

import casadi as cs
import numpy as np
from csnlp import Nlp
from csnlp.multistart.multistart_nlp import ParallelMultistartNlp
from csnlp.wrappers import Mpc

from prediction_model.dynamic_model import load_data, model
from prediction_model.weights import MahalanobisWeighting


class DhsStorageMpcEnsemble(Mpc):
    delta_u_lim = 5
    tot_mfr_min = 2
    tot_mfr_max = 25
    P_b_min = 1e3
    P_b_max = 10e6
    T_s_max = 85
    T_r_max = 75
    T_b_min = 65

    cp = 4186  # specific heat capacity of water J/(kg K)
    eta = 0.84  # efficiency of boiler from Lorenzo paper
    q_storage_max = 15

    def __init__(
        self,
        prediction_horizon,
        dt,
        layers_path,
        input_scaler_path,
        output_scaler_path,
        storage_mass: float,
        pars_init: dict,
        num_inputs=0,
        gamma=1,
        substitute_opt_vars: bool = True,
        mal_weighter: MahalanobisWeighting | None = None,
        is_Q_function: bool = False,
        max_wall_time: float | None = None,
    ):
        nlp = ParallelMultistartNlp[cs.SX](sym_type="SX", starts=1)
        input_spacing = int(cs.ceil(prediction_horizon / num_inputs))
        Mpc.__init__(
            self,
            prediction_horizon=prediction_horizon,
            nlp=nlp,
            input_spacing=input_spacing,
        )

        self.pars_init = pars_init
        self.substitute_opt_vars = substitute_opt_vars

        if num_inputs > prediction_horizon:
            raise ValueError("num_inputs must be <= prediction_horizon")
        if num_inputs == 0:
            num_inputs = prediction_horizon

        # parameters that define MPC problem
        c_s = self.parameter("c_s", (1,))  # weight for storage terminal term in cost
        T_E_ref = self.parameter("T_E_ref", (1,))  # reference for storage temp
        T_ref = self.parameter("T_ref", (5,))  # reference for load temps
        w = self.parameter("w", (1,))  # weight for slacks in cost
        c_t = self.parameter(
            "c_t", (1,)
        )  # weight for load temp penalty tracking in cost
        V0 = self.parameter("V0", (1,))  # cost offset term
        f = self.parameter(
            "f", (3 + 7 + 5,)
        )  # linear cost term (3 inputs, 7 NN outputs, T_s, T_b_r, T_e, q_b, P_b)
        Q = self.parameter("Q", (3 + 7 + 5, 3 + 7 + 5))  # quadratic cost term
        T_lim_off = self.parameter("T_lim_off", (5,))  # temperature limit offset

        # inputs variables
        q_E_h_, q_E_h = self.action("q_E_h", 1, lb=0, ub=self.q_storage_max)
        q_E_c_, q_E_c = self.action("q_E_c", 1, lb=0, ub=self.q_storage_max)
        _, T_b_s = self.action("T_b_s", 1, lb=self.T_b_min, ub=self.T_s_max)
        self.constraint(
            "directional", q_E_h_[0, :] * q_E_c_[0, :], "==", 0
        )  # bilinear constraint such that one of q_E_h or q_E_c is zero, using u_ to avoid duplicate constraints due to input spacing
        if is_Q_function:
            a0 = self.parameter("a_init", (3, 1))
            q_E_h = cs.horzcat(
                cs.repmat(a0[0, 0], 1, input_spacing), q_E_h[input_spacing:]
            )
            q_E_c = cs.horzcat(
                cs.repmat(a0[1, 0], 1, input_spacing), q_E_c[input_spacing:]
            )
            T_b_s = cs.horzcat(
                cs.repmat(a0[2, 0], 1, input_spacing), T_b_s[input_spacing:]
            )

        # NN variables and parameters
        P_loads = self.parameter("P_loads", (5, prediction_horizon))

        T_i_s, _, _ = self.variable("T_i_s", (5, prediction_horizon))
        T_r, _, _ = self.variable("T_r", (1, prediction_horizon), ub=self.T_r_max)
        q_r, _, _ = self.variable(
            "q_r", (1, prediction_horizon), lb=self.tot_mfr_min, ub=self.tot_mfr_max
        )
        y = cs.vertcat(T_i_s, T_r, q_r)

        # energy storage dynamics
        M = storage_mass
        T_E_0 = self.parameter("T_E_0", (1, 1))
        if substitute_opt_vars:
            T_E_ = [T_E_0]
            for k in range(prediction_horizon):
                T_E_.append(
                    T_E_[-1]
                    + ((dt * q_E_h[0, k]) / (M)) * (T_r[0, k] - T_E_[-1])
                    + ((dt * q_E_c[0, k]) / (M)) * (T_b_s[0, k] - T_E_[-1])
                )
            T_E_ = cs.horzcat(*T_E_)
            self.constraint("storage_temp_bounds", T_E_[1:], ">=", 0)
            self.constraint("storage_temp_bounds_ub", T_E_[1:], "<=", self.T_s_max)
            T_E = T_E_[:-1]
            T_E_N = T_E_[-1]
        else:
            T_E_, _, _ = self.variable(
                "T_E", (1, prediction_horizon), lb=0, ub=self.T_s_max
            )
            T_E = cs.horzcat(T_E_0, T_E_[:, :-1])
            T_E_N = T_E_[:, -1]
            self.constraint(
                "storage_dynamics",
                T_E_,
                "==",
                T_E
                + ((dt * q_E_h) / (M)) * (T_r - T_E)
                + ((dt * q_E_c) / (M)) * (T_b_s - T_E),
            )

        # mass flow
        if substitute_opt_vars:
            q_b = q_r - q_E_h + q_E_c
        else:
            q_b, _, _ = self.variable("q_b", (1, prediction_horizon))
            self.constraint("q_b_def", q_b, "==", q_r - q_E_h + q_E_c)
        self.constraint("q_b_min", q_b, ">=", self.tot_mfr_min)

        # mixing
        if substitute_opt_vars:
            T_s = (1 / (q_r)) * (q_b * T_b_s + q_E_h * T_E - q_E_c * T_b_s)
            T_b_r = (1 / (q_b)) * (q_r * T_r + q_E_c * T_E - q_E_h * T_r)
        else:
            T_s, _, _ = self.variable("T_s", (1, prediction_horizon))
            self.constraint(
                "T_s_def",
                T_s,
                "==",
                (1 / (q_r)) * (q_b * T_b_s + q_E_h * T_E - q_E_c * T_b_s),
            )
            T_b_r, _, _ = self.variable("T_b_r", (1, prediction_horizon))
            self.constraint(
                "T_b_r_def",
                T_b_r,
                "==",
                (1 / (q_b)) * (q_r * T_r + q_E_c * T_E - q_E_h * T_r),
            )

        # NN DHS dynamics
        x = (
            self.parameter("x", (30, 1))
            if type(layers_path) == str
            else self.parameter("x", (30, len(layers_path)))
        )  # NN internal state
        o = self.create_model(
            x,
            cs.vertcat(T_s, P_loads),
            prediction_horizon,
            layers_path,
            input_scaler_path,
            output_scaler_path,
            mal_weighter=mal_weighter,
        )
        self.constraint(
            "dynamics",
            y,
            "==",
            o[0],
        )

        # boiler power
        if substitute_opt_vars:
            P_b = self.cp * q_b * (T_b_s - T_b_r)
        else:
            P_b, _, _ = self.variable("P_b", (1, prediction_horizon))
            self.constraint("P_b_def", P_b, "==", self.cp * q_b * (T_b_s - T_b_r))

        # limits
        s, _, _ = self.variable(
            "s", (5, 1), lb=0
        )  # slacks for soft constraints on supply temp
        T_s_min = self.parameter("T_s_min", (5, prediction_horizon))
        T_r_min = self.parameter("T_r_min", (1, prediction_horizon))

        # constrain load temperatures
        self.constraint("T_s_min", T_i_s + s + T_lim_off, ">=", T_s_min)
        self.constraint("T_s_max", T_i_s, "<=", self.T_s_max + s)

        self.constraint("T_r_min", T_r, ">=", T_r_min)

        self.constraint("P_b_min", P_b, ">=", self.P_b_min)
        self.constraint("P_b_max", P_b, "<=", self.P_b_max)

        # cost
        elec_price = self.parameter("elec_price", (1, prediction_horizon))
        gammapowers = cs.DM(gamma ** np.arange(prediction_horizon)).T

        vars = cs.vertcat(
            q_E_h,
            q_E_c,
            T_b_s,
            y,
            T_s,
            T_b_r,
            T_E,
            q_b,
            P_b,
        )
        self.minimize(
            V0
            + ((gammapowers * elec_price * (dt / 3600.0)) @ (P_b.T / 1000.0))
            / self.eta  # /1000 for kW price and dt/3600 for hours
            + gammapowers[-1] * c_t * cs.sum1((T_i_s[:, -1] - T_ref) ** 2)
            + gammapowers[-1] * c_s * (T_E_N - T_E_ref) ** 2
            + w * cs.sum1(cs.sum2(s))
            + cs.sum2(f.T @ vars)
            + cs.trace(vars.T @ Q @ vars)
        )

        self.initialize_solver(max_wall_time)

    def initialize_solver(self, max_wall_time: float | None = None):
        # solver
        linear_solver = "mumps"  #  if os.name == "nt" else "ma57"
        opts = {
            "expand": True,
            "show_eval_warnings": False,
            "warn_initial_bounds": True,
            "print_time": False,
            "record_time": True,
            "bound_consistency": True,
            "calc_lam_x": True,
            "calc_lam_p": False,
            "ipopt": {
                "sb": "yes",
                "print_level": 1,
                "max_iter": 20000,
                "print_user_options": "yes",
                "print_options_documentation": "no",
                "linear_solver": linear_solver,  # spral
                "nlp_scaling_method": "gradient-based",
                "nlp_scaling_max_gradient": 10,  # TODO should this be tuned?
            },
        }
        if max_wall_time:
            opts["ipopt"]["max_wall_time"] = max_wall_time
        self.init_solver(opts, solver="ipopt")

    def create_model(
        self,
        x,
        u,
        N,
        layers_path,
        input_scaler_path,
        output_scaler_path,
        mal_weighter=None,
    ):
        if (
            not isinstance(layers_path, list)
            or not isinstance(input_scaler_path, list)
            or not isinstance(output_scaler_path, list)
        ):
            raise ValueError(
                "For EnsembleDhsMpc, layers_path, input_scaler_path, and output_scaler_path must be lists of paths"
            )

        num_models = len(layers_path)
        layers = []
        input_scalers = []
        output_scalers = []
        if mal_weighter:
            lam, _, _ = self.variable("lam", (num_models, N))
            for k in range(N):
                self.constraint(
                    f"mal_def_{k}",
                    lam[:, k],
                    "==",
                    cs.vertcat(*mal_weighter.compute_weights(u[:, k])),
                )
        else:
            lam = self.parameter("lam", (num_models, N))
        for lp, isp, osp in zip(layers_path, input_scaler_path, output_scaler_path):
            layers_dicts, input_scaler_dict, output_scaler_dict = load_data(
                lp, isp, osp
            )
            layers.append(layers_dicts)
            input_scalers.append(input_scaler_dict)
            output_scalers.append(output_scaler_dict)
        o = [
            model(
                x[:, i],
                u,
                N,
                layers[i],
                input_scalers[i],
                output_scalers[i],
                which_outputs=[0, 3, 6, 9, 12, 15, 16],
            )
            for i in range(num_models)
        ]
        return sum(
            cs.repmat(lam[i, :], o[i][0].shape[0]) * o[i][0] for i in range(num_models)
        ), [o[i][1] for i in range(num_models)]

    def solve(
        self,
        pars,
        vals0,
    ):
        pars = {
            **pars,
            **{k: v for k, v in self.pars_init.items() if k not in pars},
        }
        return self.nlp.solve(pars, vals0)
