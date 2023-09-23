import numpy as np
import matplotlib.pyplot as plt

from esn import ESN
from data_generator import generate_henon_map_data


def predict_ytp1_on_xt():
    """Train on Henon Map component X to predict components Y"""
    data = generate_henon_map_data()  # [10000, 2]

    esn = ESN(num_inputs=1, num_outputs=1, num_resv_nodes=400, leak_rate=1)

    # train the model, give x_t to predict y_tp1
    train_size = 5000
    train_input = data[:train_size, 0:1]  # keep dimension
    train_target = data[1 : train_size + 1, 1:2]
    _lambda = 0.2
    esn.train(train_input, train_target, _lambda)

    # evaluate the model, give x_t to predict y_tp1
    eval_size = 2000
    eval_input = data[train_size : train_size + eval_size, 0:1]  # keep dimension
    eval_target = data[train_size + 1 : train_size + 1 + eval_size, 1:2]
    pred_target, mse = esn.predict(eval_input, eval_target)

    # plot evaluation results
    fig = plt.figure(figsize=(16, 9))
    plt.suptitle(
        f"Henon Map Prediction of Y on X (Reservoir nodes={esn.num_resv_nodes}, Leak rate={esn.leak_rate}, Lambda={_lambda}, MSE={mse[0]:.6f})"
    )
    plt.scatter(eval_input[1:], eval_target[:-1], s=1, c='blue', marker='.', label="Actual")
    plt.scatter(eval_input[1:], pred_target[:-1], s=1, c='orange', marker='.', label="Predicted")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="upper right")
    plt.show()


def predict_xtp1_on_xt():
    """Train on Henon Map component X to continuously predict components X autonomously"""
    data = generate_henon_map_data()  # [10000, 2]

    esn = ESN(num_inputs=1, num_outputs=1, num_resv_nodes=450, leak_rate=0.75)

    # train the model, give x_t to predict x_t+1
    train_size = 5000
    train_input = data[:train_size, 0:1]  # keep dimension
    train_target = data[1 : train_size + 1, 0:1]
    _lambda = 0.25
    esn.train(train_input, train_target, _lambda)

    # evaluate the model, give x_t to predict x_t+1, t+2, ..., t+n autonomously
    eval_size = 100
    burnin = 50
    eval_input = data[train_size : train_size + eval_size, 0:1]  # keep dimension
    eval_target = data[train_size + 1 : train_size + eval_size + 1, 0:1]  # keep dimension

    pred_target, mse = esn.predict_autonomous(eval_input, eval_target, burnin)

    # plot evaluation results
    fig = plt.figure(figsize=(16, 9))
    plt.suptitle(
        f"Henon Map Autonomous Prediction of X (Reservoir nodes = {esn.num_resv_nodes}, Leak rate = {esn.leak_rate}, MSE = {mse[0]:.6f})"
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

    predict_ytp1_on_xt()

    predict_xtp1_on_xt()
