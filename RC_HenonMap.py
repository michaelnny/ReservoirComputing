import numpy as np
import matplotlib.pyplot as plt

from esn import ESN
from data_generator import generate_henon_map_data


def main():
    """Train on Henon Map predict X, Y components"""
    data = generate_henon_map_data()  # [10000, 2]

    esn = ESN(num_inputs=2, num_outputs=2, num_resv_nodes=400, leak_rate=1)

    # train the model, given x, y components at time step t, predict the x, y components at time step t+1.
    train_size = 5000
    train_input = data[:train_size, :]
    train_target = data[1 : train_size + 1, :]
    _lambda = 0.2
    esn.train(train_input, train_target, _lambda)

    # evaluate the model, given x, y components at time step t, predict the x, y components at time step t+1.
    eval_size = 500
    eval_input = data[train_size : train_size + eval_size, :]
    eval_target = data[train_size + 1 : train_size + 1 + eval_size, :]
    pred_target, mse = esn.predict(eval_input, eval_target)

    # plot evaluation results
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    fig.tight_layout(pad=4)
    fig.subplots_adjust(top=0.9)
    plt.suptitle(f"Henon Map Prediction (Reservoir nodes={esn.num_resv_nodes}, Leak rate={esn.leak_rate}, Lambda={_lambda})")
    T = range(len(pred_target))
    labels = ['X', 'Y']
    for i, label in enumerate(labels):
        ax = axes[i]
        ax.plot(T, eval_target[:, i], label="Actual", color='blue')
        ax.plot(T, pred_target[:, i], label="Predicted", color='orange', linestyle='--')
        ax.set_xlabel("t")
        ax.set_ylabel(f"{label}")
        ax.legend(loc="upper right")
        ax.set_title(f"Henon {label}, MSE={mse[i]:.6f}")

    plt.show()

    # evaluate the model on autonomous prediction
    # given x, y components at time step t, predict x, y components at time step t+1 t+1, t+2, ..., t+n autonomously.
    burnin = 400
    auto_pred_target, mse = esn.predict_autonomous(eval_input, eval_target, burnin)

    # plot evaluation results
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    fig.tight_layout(pad=4)
    fig.subplots_adjust(top=0.9)
    plt.suptitle(f"Henon Map Prediction (Reservoir nodes={esn.num_resv_nodes}, Leak rate={esn.leak_rate}, Lambda={_lambda})")
    T = range(len(auto_pred_target))
    labels = ['X', 'Y']
    for i, label in enumerate(labels):
        ax = axes[i]
        ax.plot(T, eval_target[:, i], label="Actual", color='blue')
        ax.plot(T, auto_pred_target[:, i], label="Predicted", color='orange', linestyle='--')
        ax.axvline(
            x=burnin,
            color="red",
            label="Start of Autonomous Prediction",
        )
        ax.set_xlabel("t")
        ax.set_ylabel(f"{label}")
        ax.legend(loc="upper right")
        ax.set_title(f"Henon {label}, MSE={mse[i]:.6f}")

    plt.show()


if __name__ == '__main__':
    np.random.seed(31)

    main()
