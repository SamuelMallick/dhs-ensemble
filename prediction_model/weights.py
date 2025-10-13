import casadi as cs
import numpy as np
from scipy.io import loadmat


class MahalanobisWeighting:

    def __init__(self, data_ref: list[np.ndarray], scaling: list[tuple] = []):
        """
        Initialize the Mahalanobis distance calculator with reference data.

        Parameters
        ----------
        data_ref : np.ndarray
            Each row is a data point in the reference dataset.
        scaling : list of tuples, optional
            Each tuple contains (scale, bias) for input normalization.
            If provided, data_ref will be normalized using these values.
            Also input data will be normalized before distance calculation.
        """
        # TODO Check is scaling correct because usually NN files have different order
        if scaling:
            if len(scaling) != len(data_ref):
                raise ValueError("Length of scaling must match number of features.")
            data_ref = [
                (data_ref[i] - scaling[i][1]) / (scaling[i][0])
                for i in range(len(data_ref))
            ]
        self.mu = [np.mean(data_ref[i], axis=0) for i in range(len(data_ref))]
        self.S = [np.cov(data_ref[i], rowvar=False) for i in range(len(data_ref))]
        self.S_inv = [np.linalg.inv(s) for s in self.S]
        self.scaling = scaling

    def add_data(self, new_data: np.ndarray):
        """
        Add new data points to the reference dataset and update statistics.

        Parameters
        ----------
        new_data : np.ndarray
            Each row is a new data point to add.
        """
        raise NotImplementedError("Dynamic updating of covariance not implemented.")

    def compute_weights(self, data_new: np.ndarray):
        """
        Compute weights for multiple models based on Mahalanobis distances.

        Parameters
        ----------
        data_new : np.ndarray
            Each row is a data point to compute the weights for.

        Returns
        -------
        np.ndarray
            Weights for each model, shape (num_models,).
        """
        dists = self.mahalanobis_distance_multi(data_new, return_all=False)
        # Convert distances to weights (smaller distance -> higher weight)
        inv_dists = [
            1 / (d + 1e-6) for d in dists
        ]  # add small value to avoid division by zero
        weights = [o / sum(inv_dists) for o in inv_dists]
        return weights

    def mahalanobis_distance_multi(
        self, data_new: np.ndarray | cs.SX, return_all: bool = False
    ):
        """
        Compute Mahalanobis distances of new data points relative
        to a reference dataset.

        Parameters
        ----------
        data_new : np.ndarray
            Each row is a data point to compute the distance for.
        return_all : bool, optional (default=False)
            If True, return all individual distances as a 1D array.
            If False, return only the mean distance.

        Returns
        -------
        float or np.ndarray
            Mean Mahalanobis distance (float) if return_all=False,
            otherwise array of all distances.
        """

        # Step 4: Compute Mahalanobis distances (vectorized)
        # reorder for physics-based data ordering
        # data_new
        if isinstance(data_new, cs.SX):
            data_new = cs.vertcat(
                data_new[0],
                data_new[1],
                data_new[4],
                data_new[2],
                data_new[3],
                data_new[5],
            )
            if self.scaling:
                data_new = [
                    (data_new - self.scaling[i][1]) / (self.scaling[i][0])
                    for i in range(len(self.scaling))
                ]
            dists = [
                (d - mu).T @ s_inv @ (d - mu)
                for d, mu, s_inv in zip(data_new, self.mu, self.S_inv)
            ]
            dists = [cs.fabs(d) for d in dists]
            dists = [cs.sqrt(d) for d in dists]
            return dists if return_all else [cs.sum1(d) / d.shape[0] for d in dists]
        else:  # TODO remove, just use casadi
            data_new = np.array(
                [
                    data_new[0],
                    data_new[1],
                    data_new[4],
                    data_new[2],
                    data_new[3],
                    data_new[5],
                ]
            )
            if self.scaling:
                data_new = [
                    (data_new - self.scaling[i][1]) / (self.scaling[i][0])
                    for i in range(len(self.scaling))
                ]
            dists = [
                (d - mu).reshape(1, -1) @ s_inv @ (d - mu).reshape(-1, 1)
                for d, mu, s_inv in zip(data_new, self.mu, self.S_inv)
            ]
            # if not all(np.all(dists[i] > 0) for i in range(len(dists))):
            dists = [np.abs(d) for d in dists]
            # raise ValueError(
            #     "Mahalanobis distance computation resulted in non-positive values."
            # )
            dists = [np.sqrt(d.flatten()) for d in dists]  # shape (num_data_points,)
            return dists if return_all else [np.mean(d) for d in dists]


class OptWeighting:
    def __init__(self, num_models: int, buffer_size: int):
        self.N = buffer_size
        self.nx = num_models

        H = 2 * cs.DM.ones(num_models, num_models)
        A = cs.DM.ones(1, num_models)
        qp = {}
        qp["h"] = H.sparsity()
        qp["a"] = A.sparsity()
        self.qp = cs.conic("S", "qpoases", qp)
        self.observations = []

    def add_observation(self, y_true: np.ndarray, y_pred: list[np.ndarray]):
        self.observations.append((y_true, y_pred))
        if len(self.observations) > self.N:
            self.observations.pop(0)

    def compute_weights(self):
        if not self.observations:
            raise ValueError("No observations to compute weights from.")

        num_obs = len(self.observations)
        y_true, y_preds = zip(*self.observations)
        Y = [np.hstack(y_preds[i]) for i in range(num_obs)]
        H = 2 * sum(Y[i].T @ Y[i] for i in range(num_obs))
        g = -2 * sum(y_true[i].T @ Y[i] for i in range(num_obs))
        sol = self.qp(h=H, g=g, lbx=0, ubx=1, lba=1, uba=1, a=np.ones((1, self.nx)))
        if not self.qp.stats()["success"]:
            raise RuntimeError("QP solver failed to find a solution.")

        lam = sol["x"].full().flatten()
        # error = sum(
        #     np.linalg.norm(y_true[i] - sum(lam[j] * y_preds[i][j] for j in range(self.nx)), ord=2)**2
        #     for i in range(num_obs)
        # )
        # lam_test = [1, 0]
        # error_test = sum(
        #     np.linalg.norm(y_true[i] - sum(lam_test[j] * y_preds[i][j] for j in range(self.nx)), ord=2)**2
        #     for i in range(num_obs)
        # )
        # cost = sol["cost"].full().flatten() + np.sum(np.linalg.norm(y_true[i], ord=2)**2 for i in range(num_obs))
        return sol["x"].full().flatten()
