import numpy as np
import matplotlib.pyplot as plt

from esn import ESN
from data_generator import generate_lorenz_data


def predict_yz_on_x():
    """Train on Lorenz System component X to predict components Y, Z"""
    data = generate_lorenz_data()  # [10000, 3]

    train_size = 6000
    eval_size = 2000

    # train on x, predict y, z
    train_input = data[:train_size, 0:1]  # keep dimension
    train_target = data[:train_size, 1:]

    eval_input = data[train_size : train_size + eval_size, 0:1]  # keep dimension
    eval_target = data[train_size : train_size + eval_size, 1:]

    esn = ESN(num_inputs=1, num_outputs=2, num_resv_nodes=400, leak_rate=0.5, output_bias=0)

    esn.train(train_input, train_target)

    # give X, predict Y, Z
    pred_target, mse = esn.predict(eval_input, eval_target)

    T = range(len(pred_target))

    fig, axes = plt.subplots(1, 3, figsize=(16, 9))
    fig.tight_layout(pad=4)
    fig.subplots_adjust(top=0.9)

    plt.suptitle(f"Lorenz Prediction of Y, Z on X (Reservoir nodes = {esn.num_resv_nodes}, Leak rate = {esn.leak_rate})")

    labels = ['Y', 'Z']
    for i, label in enumerate(labels):
        ax = axes[i]
        ax.plot(T, eval_target[:, i], label="Actual")
        ax.plot(T, pred_target[:, i], label="Predicted", linestyle='--')
        ax.set_xlabel("t")
        ax.set_ylabel(f"{label}")
        ax.legend(loc="upper right")
        ax.set_title(f"Lorenz {label}, MSE = {mse[i]:.6f}")

    # Make the third subplot a 3D plot
    ax_3d = fig.add_subplot(133, projection='3d')
    ax_3d.plot(eval_input, eval_target[:, 0:1], eval_target[:, 1:2], label='Actual')
    ax_3d.plot(eval_input, pred_target[:, 0:1], pred_target[:, 1:2], label='Predicted', linestyle='--')
    ax_3d.set_title('Lorenz System - 3D Plot')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.legend(loc="upper right")

    plt.show()


def predict_x_on_x():
    data = generate_lorenz_data()  # [10000, 3]

    train_size = 6000
    eval_size = 1000

    # train on x_t, predict x_t+1
    train_input = data[:train_size, 0:1]  # keep dimension
    train_target = data[1 : train_size + 1, 0:1]

    eval_input = data[train_size : train_size + eval_size, 0:1]  # keep dimension

    esn = ESN(
        num_inputs=1,
        num_outputs=1,
        num_resv_nodes=400,
        leak_rate=0.25,
    )

    esn.train(train_input, train_target)

    # give a sequence of x_t, predict x_t+1, t+2, ..., t+n autonomously
    burnin = 500
    pred_target, mse = esn.predict_autonomous(eval_input, burnin)

    T = range(len(pred_target))

    fig = plt.figure(figsize=(16, 9))

    plt.suptitle(f"Lorenz Autonomous Prediction of X (Reservoir nodes = {esn.num_resv_nodes}, Leak rate = {esn.leak_rate})")

    plt.plot(T, eval_input[:, 0], label="Actual")
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
    plt.title(f"Lorenz X, MSE = {mse[0]:.6f}")

    plt.show()


if __name__ == '__main__':
    np.random.seed(31)

    predict_yz_on_x()

    predict_x_on_x()
