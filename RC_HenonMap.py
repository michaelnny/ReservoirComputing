import numpy as np
import matplotlib.pyplot as plt

from esn import ESN
from data_generator import generate_henon_map_data


def predict_y_on_x():
    """Train on Henon Map component X to predict components Y"""
    data = generate_henon_map_data()  # [10000, 2]

    train_size = 5000
    eval_size = 2000

    # train on x, predict y
    train_input = data[:train_size, 0:1]  # keep dimension
    train_target = data[:train_size, 1:2]

    eval_input = data[train_size : train_size + eval_size, 0:1]  # keep dimension
    eval_target = data[train_size : train_size + eval_size, 1:2]

    esn = ESN(num_inputs=1, num_outputs=1, num_resv_nodes=400, leak_rate=0.5)

    esn.train(train_input, train_target)

    # give X, predict Y
    pred_target, mse = esn.predict(eval_input, eval_target)

    fig = plt.figure(figsize=(16, 9))
    plt.suptitle(f"Henon Map Prediction of Y on X (Reservoir nodes = {esn.num_resv_nodes}, Leak rate = {esn.leak_rate}, MSE = {mse[0]:.6f})")
    plt.scatter(eval_input, eval_target, s=1, c='blue', marker='.', label="Actual")
    plt.scatter(eval_input, pred_target, s=1, c='orange', marker='.', label="Predicted")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="upper right")
    plt.show()


def predict_x_on_x():
    data = generate_henon_map_data()  # [10000, 2]

    train_size = 5000
    eval_size = 500

    # train on x_t, predict x_t+1
    train_input = data[:train_size, 0:1]  # keep dimension
    train_target = data[1 : train_size + 1, 0:1]

    eval_input = data[train_size : train_size + eval_size, 0:1]  # keep dimension

    esn = ESN(num_inputs=1, num_outputs=1, num_resv_nodes=400, leak_rate=0.1)

    esn.train(train_input, train_target)

    # give a sequence of x_t, predict x_t+1, t+2, ..., t+n autonomously
    burnin = 300
    pred_target, mse = esn.predict_autonomous(eval_input, burnin)

    T = range(len(pred_target))

    fig = plt.figure(figsize=(16, 9))

    plt.suptitle(f"Henon Map Autonomous Prediction of X (Reservoir nodes = {esn.num_resv_nodes}, Leak rate = {esn.leak_rate}, MSE = {mse[0]:.6f})")

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
    plt.show()


if __name__ == '__main__':
    np.random.seed(31)

    predict_y_on_x()

    predict_x_on_x()
