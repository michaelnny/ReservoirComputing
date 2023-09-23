from typing import Any, Callable, Tuple
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


class InputLayer:
    def __init__(self, in_features: int, out_features: int, scale: float = 1.0, bias: float = 1.0) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.W = None
        self.b = bias

        self.init_weights()

    def init_weights(self) -> None:
        # uniform distribution [-1, 1]
        self.W = np.random.uniform(-1, 1, size=(self.in_features, self.out_features)) * self.scale

    def reset(self) -> None:
        self.init_weights()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.W) + self.b


class OutputLayer:
    def __init__(self, in_features: int, out_features: int, bias: float = 1.0) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.W = None
        self.b = bias

        self.init_weights()

    def init_weights(self) -> None:
        # uniform distribution [-1, 1]
        self.W = np.random.uniform(-1, 1, size=(self.in_features, self.out_features))

    def reset(self) -> None:
        self.init_weights()

    def update_weights(self, weights: np.ndarray) -> None:
        # update weights of output layer
        assert weights.shape == (self.in_features, self.out_features)
        self.W = weights

    def __call__(self, r_t: np.ndarray) -> np.ndarray:
        return np.dot(r_t, self.W) + self.b


class ReservoirLayer:
    def __init__(self, num_nodes: int, leak_rate: float, spectral_radius: float = 0.0, activation=np.tanh) -> None:
        self.num_nodes = num_nodes

        # leakage rate [0, 1]
        self.leak_rate = leak_rate
        # if 0, compute it dynamically
        self.spectral_radius = spectral_radius
        self.activation = activation
        self.W = None

        self.init_weights()

    def init_weights(self):
        weights = np.random.normal(0, 1, self.num_nodes * self.num_nodes).reshape([self.num_nodes, self.num_nodes])
        # spectral_radius, if self.spectral_radius is 0, compute it dynamically
        if self.spectral_radius <= 0:
            self.spectral_radius = max(abs(linalg.eigvals(weights)))
        self.W = weights / self.spectral_radius

    def reset(self) -> None:
        self.init_weights()

    def __call__(self, r_t: np.ndarray, u_t: np.ndarray) -> np.ndarray:
        return (1 - self.leak_rate) * r_t + self.leak_rate * self.activation(np.dot(r_t, self.W) + u_t)


class ESN:
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_resv_nodes: int,
        leak_rate: float = 0.5,
        spectral_radius: float = 0.0,
        activation: Callable = np.tanh,
        input_bias: float = 1.0,
        output_bias: float = 0.0,
    ) -> None:
        self.num_inputs = num_inputs
        self.num_resv_nodes = num_resv_nodes
        self.num_outputs = num_outputs
        self.leak_rate = leak_rate

        self.input_layer = InputLayer(num_inputs, num_resv_nodes, input_bias)
        self.resv_layer = ReservoirLayer(num_resv_nodes, leak_rate, spectral_radius, activation)
        self.output_layer = OutputLayer(num_resv_nodes, num_outputs, output_bias)

        self.reservoir_states = []

    def reset(self):
        self.input_layer.reset()
        self.resv_layer.reset()
        self.output_layer.reset()

        self.reservoir_states = []

    def train(
        self,
        train_input: np.ndarray,
        train_target: np.ndarray,
        _lambda: float = 0.1,
    ) -> None:
        assert len(train_input) == len(train_target)
        assert len(train_input.shape) == len(train_target.shape) == 2

        T = len(train_input)
        self.reservoir_states = []

        # initialize dummy reservoir state for first timestep
        r_t = np.zeros(self.num_resv_nodes)
        # collect reservoir states
        for t in range(T):
            u_t = self.input_layer(train_input[t])
            r_tp1 = self.resv_layer(r_t, u_t)
            self.reservoir_states.append(r_tp1)
            r_t = r_tp1

        # update output layer weights
        reservoir_states = np.vstack(self.reservoir_states)
        self.update_output_weights(reservoir_states, train_target, _lambda)

    def update_output_weights(self, reservoir_states: np.ndarray, train_target: np.ndarray, _lambda: float) -> None:
        # Compute the output weights analytically
        # Ridge Regression
        E_lambda = np.identity(self.num_resv_nodes) * _lambda
        inv_x = np.linalg.inv(np.dot(reservoir_states.T, reservoir_states) + E_lambda)
        # update weights of output layer
        out_weights = np.dot(np.dot(inv_x, reservoir_states.T), train_target)
        # make sure have the dimension
        out_weights = out_weights.reshape(self.output_layer.W.shape)

        # TODO: what about bias of the output layer????
        self.output_layer.update_weights(out_weights)

    def predict(self, input_data: np.ndarray, true_target: np.ndarray) -> Tuple[np.ndarray, float]:
        assert len(input_data) == len(true_target)
        assert len(input_data.shape) == len(true_target.shape) == 2

        T = len(input_data)
        pred_target = []

        # initialize dummy reservoir state for first timestep
        r_t = np.zeros(self.num_resv_nodes)
        for t in range(T):
            u_t = self.input_layer(input_data[t])
            r_tp1 = self.resv_layer(r_t, u_t)
            x_tp1 = self.output_layer(r_tp1)

            pred_target.append(x_tp1)
            r_t = r_tp1

        pred_target = np.vstack(pred_target)
        assert pred_target.shape == true_target.shape

        # compute MSE
        squared_diff = (pred_target - true_target) ** 2
        mse = np.mean(squared_diff, axis=0)

        return pred_target, mse

    def predict_autonomous(self, input_data: np.ndarray, burnin=10) -> Tuple[np.ndarray, float]:
        assert len(input_data.shape) == 2

        T = len(input_data)
        pred_target = []

        # initialize dummy reservoir state for first timestep
        r_t = np.zeros(self.num_resv_nodes)

        x_t = input_data[0]
        for t in range(T):
            u_t = self.input_layer(x_t)
            r_tp1 = self.resv_layer(r_t, u_t)
            x_tp1 = self.output_layer(r_tp1)

            pred_target.append(x_tp1)

            r_t = r_tp1

            if t >= burnin:
                x_t = x_tp1
            else:
                x_t = input_data[t]

        pred_target = np.vstack(pred_target)
        assert pred_target.shape == input_data.shape

        # compute MSE
        squared_diff = (pred_target[burnin:, :] - input_data[burnin:, :]) ** 2
        mse = np.mean(squared_diff, axis=0)

        return pred_target, mse
