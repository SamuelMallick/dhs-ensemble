import casadi as cs
import numpy as np

from prediction_model.dynamic_model import load_data, model


class TrivialObserver:
    def __init__(
        self, layers_path: str, input_scaler_path: str, output_scaler_path: str
    ):
        self.x = np.zeros((30, 1))  # initial state for the neural network
        self.layers_dicts, self.input_scaler_dict, self.output_scaler_dict = load_data(
            layers_path, input_scaler_path, output_scaler_path
        )

    def step(self, u):
        o = model(
            self.x,
            cs.vertcat(*u),
            1,
            self.layers_dicts,
            self.input_scaler_dict,
            self.output_scaler_dict,
        )
        self.x = o[1]
        y = o[0]
        return self.x, y

    def reset(self):
        self.x = np.zeros((30, 1))
        return self.x


class TrivialEnsembleObserver:
    def __init__(
        self, layers_path: list, input_scaler_path: list, output_scaler_path: list
    ):
        if (
            not isinstance(layers_path, list)
            or not isinstance(input_scaler_path, list)
            or not isinstance(output_scaler_path, list)
        ):
            raise ValueError(
                "For TrivialEnsembleObserver, layers_path, input_scaler_path, and output_scaler_path must be lists of paths"
            )

        self.x = [
            np.zeros((30, 1)) for _ in range(len(layers_path))
        ]  # initial states or the neural networks
        self.layers = []
        self.input_scalers = []
        self.output_scalers = []
        for lp, isp, osp in zip(layers_path, input_scaler_path, output_scaler_path):
            layers_dicts, input_scaler_dict, output_scaler_dict = load_data(
                lp, isp, osp
            )
            self.layers.append(layers_dicts)
            self.input_scalers.append(input_scaler_dict)
            self.output_scalers.append(output_scaler_dict)

    def step(self, u):
        num_models = len(self.layers)
        o = [
            model(
                self.x[i],
                cs.vertcat(*u),
                1,
                self.layers[i],
                self.input_scalers[i],
                self.output_scalers[i],
            )
            for i in range(num_models)
        ]
        self.x = [o[i][1] for i in range(num_models)]
        y = [o[i][0] for i in range(num_models)]
        return self.x, y

    def reset(self):
        self.x = [np.zeros((30, 1)) for _ in range(len(self.layers))]
        return self.x
