import numpy as np
import matplotlib.pyplot as plt

from esn import ESN
from data_generator import generate_lorenz_data


def predict_yztp1_on_xt():
    """Train on Lorenz System component X to predict components Y, Z"""
    data = generate_lorenz_data()  # [10000, 3]

    esn = ESN(num_inputs=1, num_outputs=2, num_resv_nodes=400, leak_rate=0.75)

    # train the model, give x_t to predict y_tp1, z_tp1
    train_size = 6000
    train_input = data[:train_size, 0:1]  # keep dimension
    train_target = data[1 : train_size + 1, 1:]
    _lambda = 0.2
    esn.train(train_input, train_target, _lambda)

    # evaluate the model, give x_t to predict y_tp1, z_tp1
    eval_size = 2000
    eval_input = data[train_size : train_size + eval_size, 0:1]  # keep dimension
    eval_target = data[train_size + 1 : train_size + 1 + eval_size, 1:]

    pred_target, mse = esn.predict(eval_input, eval_target)

    # plot evaluation results
    fig, axes = plt.subplots(1, 3, figsize=(16, 9))
    fig.tight_layout(pad=4)
    fig.subplots_adjust(top=0.9)
    plt.suptitle(
        f"Lorenz Prediction of Y, Z on X (Reservoir nodes={esn.num_resv_nodes}, Leak rate={esn.leak_rate}, Lambda={_lambda})"
    )
    T = range(len(pred_target))
    labels = ['Y', 'Z']
    for i, label in enumerate(labels):
        ax = axes[i]
        ax.plot(T, eval_target[:, i], label="Actual")
        ax.plot(T, pred_target[:, i], label="Predicted", linestyle='--')
        ax.set_xlabel("t")
        ax.set_ylabel(f"{label}")
        ax.legend(loc="upper right")
        ax.set_title(f"Lorenz {label}, MSE={mse[i]:.6f}")

    # Make the third subplot a 3D plot
    ax_3d = fig.add_subplot(133, projection='3d')
    ax_3d.plot(eval_input[1:], eval_target[:-1, 0:1], eval_target[:-1, 1:2], label='Actual')
    ax_3d.plot(eval_input[1:], pred_target[:-1, 0:1], pred_target[:-1, 1:2], label='Predicted', linestyle='--')
    ax_3d.set_title('Lorenz System - 3D Plot')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.legend(loc="upper right")

    plt.show()


def predict_xtp1_on_xt():
    """Train on Lorenz System component X to continuously predict components X autonomously"""
    data = generate_lorenz_data()  # [10000, 3]

    esn = ESN(num_inputs=1, num_outputs=1, num_resv_nodes=670, leak_rate=0.75)

    # train the model, give x_t to predict x_t+1
    train_size = 6000
    train_input = data[:train_size, 0:1]  # keep dimension
    train_target = data[1 : train_size + 1, 0:1]
    _lambda = 0.2
    esn.train(train_input, train_target, _lambda)

    # evaluate the model, give x_t to predict x_t+1, t+2, ..., t+n autonomously
    eval_size = 200
    burnin = 100
    eval_input = data[train_size : train_size + eval_size, 0:1]  # keep dimension
    eval_target = data[train_size + 1 : train_size + eval_size + 1, 0:1]  # keep dimension

    pred_target, mse = esn.predict_autonomous(eval_input, eval_target, burnin)

    # plot evaluation results
    fig = plt.figure(figsize=(16, 9))
    plt.suptitle(
        f"Lorenz Autonomous Prediction of X (Reservoir nodes={esn.num_resv_nodes}, Leak rate={esn.leak_rate}, Lambda={_lambda}, MSE={mse[0]:.6f})"
    )
    T = range(len(pred_target))
    plt.plot(T, eval_target[:, 0], label="Actual")
    plt.plot(T, pred_target[:, 0], label="Predicted", linestyle='--')

    # Plot a vertical line indicate start of Autonomous Prediction
    plt.axvline(x=burnin, color="red")

    # Get the y-limits of the current plot
    ymin, ymax = plt.ylim()
    # Calculate the y-coordinate for the label text
    y_coord = ymin + 0.95 * (ymax - ymin)
    plt.text(burnin + 5, y_coord, "Start of Autonomous Prediction", color="red")

    plt.xlabel("t")
    plt.ylabel('X')
    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    np.random.seed(31)

    predict_yztp1_on_xt()

    predict_xtp1_on_xt()
