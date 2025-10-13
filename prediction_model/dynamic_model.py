import casadi as cs
import numpy as np
import scipy.io


def sigmoid(x):
    """
    Sigmoid activation function.

    Args:
        x (casadi.SX or casadi.DM): The input tensor.

    Returns:
        casadi.SX or casadi.DM: The output tensor after applying the sigmoid function.
    """
    return 1 / (1 + cs.exp(-x))


def gru_cell(uk, xk, weights, n_states):
    """
    Implements a single GRU cell step using CasADi.

    Args:
        uk (casadi.SX or casadi.DM): The input to the GRU cell.
        xk (casadi.SX or casadi.DM): The current state of the GRU cell.
        weights (dict): A dictionary containing the GRU weights and biases.
        n_states (int): The number of states in the GRU cell.

    Returns:
        tuple: A tuple containing the next state (xkp1) of the GRU cell.
    """
    # Extract weights and biases
    Wzf = weights["Wzf"]
    Uzf = weights["Uzf"]
    bzf = weights["bzf"]
    Wr = weights["Wr"]
    Ur = weights["Ur"]
    br = weights["br"]

    # Update and Reset gates
    zk = sigmoid(
        cs.mtimes(Wzf[:, :n_states].T, uk)
        + cs.mtimes(Uzf[:, :n_states].T, xk)
        + bzf[:n_states]
    )
    fk = sigmoid(
        cs.mtimes(Wzf[:, n_states:].T, uk)
        + cs.mtimes(Uzf[:, n_states:].T, xk)
        + bzf[n_states:]
    )

    # Candidate hidden state
    phi = cs.tanh(cs.mtimes(Wr.T, uk) + cs.mtimes(Ur.T, (fk * xk)) + br)

    # Next hidden state
    xkp1 = zk * xk + (1 - zk) * phi

    return xkp1


def model(
    x_k, u, N, layers_dicts, input_scaler_dict, output_scaler_dict, which_outputs=[]
):
    """
    Propagates the dynamic model based on GRU RNNs for N steps.

    This function is designed to be used within a CasADi optimization framework.

    Args:
        x_k (casadi.SX or casadi.DM): Initial state of the combined GRU model.
        u (casadi.SX or casadi.DM): Input sequence for the entire horizon.
        N (int): The number of steps to propagate the model.
        layers_dicts (list): List of dictionaries, each containing GRU layer weights.
        input_scaler_dict (dict): Dictionary containing input scaling parameters.
        output_scaler_dict (dict): Dictionary containing output scaling parameters.
        which_outputs (list, optional): List of indices specifying which outputs to return.
                                         If empty, all outputs are returned. Defaults to [].

    Returns:
        tuple: A tuple containing the predicted outputs over the horizon and the final state.
    """
    # re-ordering because network was trained with inputs differently
    order = [0, 1, 4, 2, 3, 5]
    u = u[order, :]

    n_states = [3, 3, 3, 8, 4, 9]
    n_states_total = sum(n_states)

    xk = x_k
    y_N = []

    for j in range(N):
        uk = u[:, j]

        # Scale input
        input_bias = input_scaler_dict["bias"]
        input_scale = input_scaler_dict["scale"]
        uk_scaled = (uk - input_bias) / input_scale

        # Define custom sums for inputs to the first 5 GRU cells
        sum1 = uk_scaled[2, :] + uk_scaled[3, :] + uk_scaled[4, :] + uk_scaled[5, :]
        sum2 = uk_scaled[1, :] + uk_scaled[3, :] + uk_scaled[4, :] + uk_scaled[5, :]
        sum3 = uk_scaled[1, :] + uk_scaled[2, :] + uk_scaled[4, :] + uk_scaled[5, :]
        sum4 = uk_scaled[1, :] + uk_scaled[2, :] + uk_scaled[3, :] + uk_scaled[5, :]
        sum5 = uk_scaled[1, :] + uk_scaled[2, :] + uk_scaled[3, :] + uk_scaled[4, :]

        # Inputs for each GRU
        uk1 = cs.vertcat(uk_scaled[0, :], uk_scaled[1, :], sum1)
        uk2 = cs.vertcat(uk_scaled[0, :], uk_scaled[2, :], sum2)

        # GRU 1
        xk1 = xk[0 : n_states[0], :]
        xkp1 = gru_cell(uk1, xk1, layers_dicts[0], n_states[0])
        yk1 = cs.mtimes(layers_dicts[1]["weight"], xk1) + layers_dicts[1]["bias"]

        # GRU 2
        xk2 = xk[n_states[0] : n_states[0] + n_states[1], :]
        xkp2 = gru_cell(uk2, xk2, layers_dicts[2], n_states[1])
        yk2 = cs.mtimes(layers_dicts[3]["weight"], xk2) + layers_dicts[3]["bias"]

        # Inputs for GRU 3 (now that yk1 and yk2 are defined)
        uk3 = cs.vertcat(uk_scaled[3, :], yk1[0, :], sum3)

        # GRU 3
        xk3 = xk[sum(n_states[:2]) : sum(n_states[:3]), :]
        xkp3 = gru_cell(uk3, xk3, layers_dicts[4], n_states[2])
        yk3 = cs.mtimes(layers_dicts[5]["weight"], xk3) + layers_dicts[5]["bias"]

        # Inputs for GRU 4 and 5 (now that yk3 is defined)
        uk4 = cs.vertcat(uk_scaled[4, :], yk2[0, :], yk3[0, :], sum4)
        uk5 = cs.vertcat(uk_scaled[5, :], yk2[0, :], yk3[0, :], sum5)

        # GRU 4
        xk4 = xk[sum(n_states[:3]) : sum(n_states[:4]), :]
        xkp4 = gru_cell(uk4, xk4, layers_dicts[6], n_states[3])
        yk4 = cs.mtimes(layers_dicts[7]["weight"], xk4) + layers_dicts[7]["bias"]

        # GRU 5
        xk5 = xk[sum(n_states[:4]) : sum(n_states[:5]), :]
        xkp5 = gru_cell(uk5, xk5, layers_dicts[8], n_states[4])
        yk5 = cs.mtimes(layers_dicts[9]["weight"], xk5) + layers_dicts[9]["bias"]

        # GRU 6 (input depends on outputs of previous GRUs)
        uk6 = cs.vertcat(yk1[1:, :], yk2[1:, :], yk3[1:, :], yk4[1:, :], yk5[1:, :])
        xk6 = xk[sum(n_states[:5]) : sum(n_states[:6]), :]
        xkp6 = gru_cell(uk6, xk6, layers_dicts[10], n_states[5])
        yk6 = cs.mtimes(layers_dicts[11]["weight"], xk6) + layers_dicts[11]["bias"]

        # Update the overall state
        xk = cs.vertcat(xkp1, xkp2, xkp3, xkp4, xkp5, xkp6)

        # Combine GRU outputs and scale
        y_raw = cs.vertcat(yk1, yk2, yk3, yk4, yk5, yk6)

        output_bias = output_scaler_dict["bias"]
        output_scale = output_scaler_dict["scale"]

        y_scaled = y_raw * output_scale + output_bias

        # reorder as NN is trained with loads in order 1, 4, 2, 3, 5
        blocks = cs.reshape(
            y_scaled[:15, :], (3, 5)
        ).T  # (3, 5) with transpose as casadi uses col-major
        order = [0, 2, 3, 1, 4]
        y_scaled = cs.vertcat(
            cs.vertcat(*[blocks[i, :].T for i in order]), y_scaled[15:, :]
        )

        # Select relevant outputs
        y_k = (
            cs.vertcat(*[y_scaled[i, :] for i in which_outputs])
            if which_outputs
            else y_scaled
        )

        y_N.append(y_k)

    return cs.horzcat(*y_N), xk


def load_data(layers_file_path, input_scaler_file_path, output_scaler_file_path):
    """
    Loads the model weight and scaler data from MATLAB .mat files.

    Args:
        layers_file_path (str): Path to the layers .mat file.
        input_scaler_file_path (str): Path to the input scaler .mat file.
        output_scaler_file_path (str): Path to the output scaler .mat file.

    Returns:
        tuple: A tuple containing the loaded dictionaries for layers,
               input scaler, and output scaler.
    """
    # Load MATLAB .mat files
    layers_mat = scipy.io.loadmat(layers_file_path, squeeze_me=True)
    input_scaler_mat = scipy.io.loadmat(input_scaler_file_path, squeeze_me=True)
    output_scaler_mat = scipy.io.loadmat(output_scaler_file_path, squeeze_me=True)

    # restructure based off strange way scipy loads them
    layers_list = [
        l["weights"].item() for l in layers_mat["layers_file"].item()[0]
    ]  # this is now a list of layers
    input_scaler = input_scaler_mat["input_scaler_file"].item()[0]
    output_scaler = output_scaler_mat["output_scaler_file"].item()[0]

    layers_dicts = []
    for l in layers_list:
        layer_dict = {}
        for name in l.dtype.names:
            layer_dict[name] = l[name].item()
        layers_dicts.append(layer_dict)

    input_scaler_dict = {}
    for name in input_scaler.dtype.names:
        input_scaler_dict[name] = input_scaler[name].item()

    output_scaler_dict = {}
    for name in output_scaler.dtype.names:
        output_scaler_dict[name] = output_scaler[name].item()

    return layers_dicts, input_scaler_dict, output_scaler_dict
