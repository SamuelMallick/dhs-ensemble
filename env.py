import os
import re
from typing import Any

import casadi as cs
import gymnasium as gym
import numpy as np
from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Slave


class DHSSystem(gym.Env[np.ndarray, np.ndarray]):
    cp = 4186  # specific heat capacity of water J/(kg K)

    internal_step_size = 0.01  # seconds

    eta_gen = 0.84
    eta_pump = 0.8

    H = 20  # pump head in m
    g = 9.81  # gravity m/s^2

    q_b_min = 2.0  # minimum boiler flow kg/s

    w = 20  # penalty on constraint violation

    def __init__(
        self,
        fmu_filename: str,
        step_size: float,
        P_loads: np.ndarray,
        elec_price: np.ndarray,
        T_s_min: np.ndarray,
        storage_mass: float,
    ):
        super().__init__()
        self.fmu_filename = fmu_filename
        self.step_size = step_size
        self.num_internal_steps = int(self.step_size / self.internal_step_size)
        self.time = 0.0

        self.P_loads = P_loads
        self.elec_price = elec_price
        self.T_s_min = T_s_min
        self.storage_mass = storage_mass

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Resets the state of the network. An x0 can be passed in the options dict."""
        super().reset(seed=seed, options=options)
        start_time = 0.0
        self.time = 0.0

        # read the model description
        model_description = read_model_description(self.fmu_filename)

        # collect the value references
        with_storage = False
        vrs = {}
        for variable in model_description.modelVariables:
            vrs[variable.name] = variable.valueReference
            if "mfr_stes" in variable.name:
                with_storage = True
        input_names = [
            "T_boiler_ref",
            "P_load1",
            "P_load2",
            "P_load3",
            "P_load4",
            "P_load5",
        ]
        output_names = [
            "Ts_load[1]",  # 0
            "Tr_load[1]",  # 1
            "mfr_load[1]",  # 2
            "Ts_load[2]",  # 3
            "Tr_load[2]",  # 4
            "mfr_load[2]",  # 5
            "Ts_load[3]",  # 6
            "Tr_load[3]",  # 7
            "mfr_load[3]",  # 8
            "Ts_load[4]",  # 9
            "Tr_load[4]",  # 10
            "mfr_load[4]",  # 11
            "Ts_load[5]",  # 12
            "Tr_load[5]",  # 13
            "mfr_load[5]",  # 14
            "T_ret",  # 15
            "mfr_ret",  # 16
            "T_boiler_out",  # 17
            "T_supply",  # 18
            "mfr_supply",  # 19
            "P_boiler_out",  # 20
        ]
        if with_storage:
            input_names = ["mfr_stes"] + input_names
            output_names = output_names + [
                "T_tes",
                "mfr_boiler",
                "T_boiler_in",
            ]

        pattern_L = re.compile(r"^[sr]\d{2}\.(L)$")
        self.pipe_lengths = [vrs[name] for name in vrs if pattern_L.match(name)]
        pattern_D = re.compile(r"^[sr]\d{2}\.(D)$")
        self.pipe_diameters = [vrs[name] for name in vrs if pattern_D.match(name)]
        self.inputs = [vrs[name] for name in input_names]
        self.outputs = [vrs[name] for name in output_names]

        # extract the FMU
        unzipdir = extract(self.fmu_filename)

        fmu = FMU2Slave(
            guid=model_description.guid,
            unzipDirectory=unzipdir,
            modelIdentifier=model_description.coSimulation.modelIdentifier,
            instanceName="instance1",
        )

        # initialize
        fmu.instantiate()
        fmu.setupExperiment(startTime=start_time)
        fmu.enterInitializationMode()
        fmu.exitInitializationMode()

        self.y = fmu.getReal(self.outputs)
        self.fmu = fmu
        self.time = start_time
        self.with_storage = with_storage

        if with_storage:
            rad = self.fmu.getReal([vrs["tes.D"]])[0] / 2.0  # half diameter
            self.fmu.setReal(
                [vrs["tes.h"]], [(self.storage_mass) / (rad**2 * np.pi * 1000.0)]
            )  # set height based on mass, density=1000kg/m3

        return self.y, {}

    def get_stage_cost(self, output: np.ndarray, action: np.ndarray) -> float:
        """Computes the stage cost `L(s,a)`. A per second cost."""
        if len(output) < 23:
            raise NotImplementedError(
                "get_stage_cost not implemented for systems without storage."
            )
        P = output[20]
        T_s = [output[i] for i in [0, 3, 6, 9, 12]]
        q_b = output[-2]
        index = int(
            np.floor(self.time / self.step_size)
        )  # for indexing the values that are passed with 5 min intervals
        gen_cost = (self.elec_price[index] * (1 / 3600.0) * (P / 1000.0)) / self.eta_gen
        pump_cost = (
            self.elec_price[index] * (1 / (3600.0 * 1000.0)) * (q_b * self.g * self.H)
        ) / self.eta_pump
        viol_vost = np.sum(np.maximum(0, self.T_s_min[index] - T_s)) * self.w
        return gen_cost  #  + viol_vost  #  + pump_cost

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Steps the system."""
        r = 0
        if isinstance(action, cs.DM):
            action = action.full()
        if self.with_storage and action.shape[0] == 1:
            action = np.vstack((0, action))
        elif action.shape[0] == 3:
            action = np.array([action[1] - action[0], action[2]])
        step_counter = 0
        while step_counter < self.num_internal_steps:
            if step_counter % int(1 / self.internal_step_size) == 0:
                u = np.vstack(
                    [action, self.P_loads[:, [int(self.time % 1)]]]
                )  # %1 because loads change every second
                self.fmu.setReal(self.inputs, list(u))
            r += self.get_stage_cost(self.y, action) * (
                self.step_size / self.num_internal_steps
            )
            self.fmu.doStep(
                currentCommunicationPoint=self.time,
                communicationStepSize=self.internal_step_size,
            )
            self.time += self.internal_step_size
            if step_counter == 0:
                y_new = self.fmu.getReal(self.outputs)
                if self.with_storage and y_new[-2] < self.q_b_min:
                    action[0] = (
                        self.q_b_min - y_new[16]
                    )  # enforce storage flow to respect min boiler flow
                    print(f"Storage flow modified to {action[0]}")
                    u = np.vstack(
                        [action, self.P_loads[:, [int(self.time % 1)]]]
                    )  # %1 because loads change every second
                    self.fmu.setReal(self.inputs, list(u))
            step_counter += 1
        y_new = self.fmu.getReal(self.outputs)
        self.y = y_new
        return (
            y_new,
            r,
            False,
            False,
            {"P_loads": self.P_loads[:, int(self.time % 1)]},
        )

    def endogenous_model_change(self):
        pipe_lengths = self.fmu.getReal(self.pipe_lengths)
        pipe_diameters = self.fmu.getReal(self.pipe_diameters)
        self.fmu.setReal(self.pipe_lengths, [l * 1.05 for l in pipe_lengths])
        self.fmu.setReal(self.pipe_diameters, [d * 0.95 for d in pipe_diameters])

    def get_T_E(self) -> float:
        if not self.with_storage:
            raise NotImplementedError(
                "get_T_E not implemented for systems without storage."
            )
        return self.y[-3]

    def get_num_outputs(self) -> int:
        return len(self.outputs)
