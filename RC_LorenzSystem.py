import numpy as np
import matplotlib.pyplot as plt

from esn import ESN
from data_generator import generate_lorenz_data


def main():
    """Train on Lorenz System to predict X, Y, Z components"""
    data = generate_lorenz_data()  # [10000, 3]

    esn = ESN(num_inputs=3, num_outputs=3, num_resv_nodes=600, leak_rate=0.75)

    # train the model, give x_t, y_t, z_t to predict x_t+1, y_t+1, z_t+1
    train_size = 6000
    train_input = data[:train_size, :]
    train_target = data[1 : train_size + 1, :]
    _lambda = 0.25
    esn.train(train_input, train_target, _lambda)

    # evaluate the model, give x_t, y_t, z_t to predict x_t+1, y_t+1, z_t+1
    eval_size = 2000
    eval_input = data[train_size : train_size + eval_size, :]
    eval_target = data[train_size + 1 : train_size + 1 + eval_size, :]
    pred_target, mse = esn.predict(eval_input, eval_target)

    # plot evaluation results
    fig, axes = plt.subplots(3, 1, figsize=(16, 9))
    fig.tight_layout(pad=4)
    fig.subplots_adjust(top=0.9)
    plt.suptitle(f"Lorenz Prediction (Reservoir nodes={esn.num_resv_nodes}, Leak rate={esn.leak_rate}, Lambda={_lambda})")
    T = range(len(pred_target))
    labels = ['X', 'Y', 'Z']
    for i, label in enumerate(labels):
        ax = axes[i]
        ax.plot(T, eval_target[:, i], label="Actual", color='blue')
        ax.plot(T, pred_target[:, i], label="Predicted", color='orange', linestyle='--')
        ax.set_xlabel("t")
        ax.set_ylabel(f"{label}")
        ax.legend(loc="upper right")
        ax.set_title(f"Lorenz {label}, MSE={mse[i]:.6f}")

    plt.show()

    # Make a 3D plot for comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    plt.suptitle(
        f"Lorenz  Prediction - 3D Comparison (Reservoir nodes={esn.num_resv_nodes}, Leak rate={esn.leak_rate}, Lambda={_lambda})"
    )
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(eval_target[:-1, 0:1], eval_target[:-1, 1:2], eval_target[:-1, 2:3], color='blue')
    ax1.set_title('Lorenz System - Actual')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(pred_target[1:, 0:1], pred_target[:-1, 1:2], pred_target[:-1, 2:3], color='orange')
    ax2.set_title('Lorenz System - Predicted')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    plt.show()

    # evaluate the model on autonomous prediction
    burnin = 1500
    auto_pred_target, mse = esn.predict_autonomous(eval_input, eval_target, burnin)

    # plot evaluation results
    fig, axes = plt.subplots(3, 1, figsize=(16, 9))
    fig.tight_layout(pad=4)
    fig.subplots_adjust(top=0.9)
    plt.suptitle(
        f"Lorenz Autonomous Prediction (Reservoir nodes={esn.num_resv_nodes}, Leak rate={esn.leak_rate}, Lambda={_lambda})"
    )
    T = range(len(auto_pred_target))
    labels = ['X', 'Y', 'Z']
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
        ax.set_title(f"Lorenz {label}, MSE={mse[i]:.6f}")

    plt.show()


if __name__ == '__main__':
    np.random.seed(31)

    main()
