import numpy as np
import matplotlib.pyplot as plt

from esn import ESN
from data_generator import generate_mackey_glass_data


def main():
    """Train on Mackey-Glass to predict X component"""
    data = generate_mackey_glass_data()  # [10000, 1]

    esn = ESN(num_inputs=1, num_outputs=1, num_resv_nodes=200, leak_rate=1)

    # train the model, given x at time step t, predict the x at time step t+1.
    train_size = 5000
    train_input = data[:train_size, :]
    train_target = data[1 : train_size + 1, :]
    _lambda = 0.25
    esn.train(train_input, train_target, _lambda)

    # evaluate the model, given x at time step t, predict the x at time step t+1.
    eval_size = 2000
    eval_input = data[train_size : train_size + eval_size, :]
    eval_target = data[train_size + 1 : train_size + 1 + eval_size, :]
    pred_target, mse = esn.predict(eval_input, eval_target)

    # plot evaluation results
    fig = plt.figure(figsize=(16, 9))

    plt.suptitle(
        f"Mackey-Glass Prediction (Reservoir nodes={esn.num_resv_nodes}, Leak rate={esn.leak_rate}, Lambda={_lambda}, MSE={mse[0]:.6f})"
    )
    T = range(len(pred_target))

    plt.plot(T, eval_target[:, 0], label="Actual", color='blue')
    plt.plot(T, pred_target[:, 0], label="Predicted", color='orange', linestyle='--')
    plt.xlabel("t")
    plt.ylabel("X")
    plt.legend(loc="upper right")
    plt.show()

    # evaluate the model on autonomous prediction
    # given x at time step t, predict x at time step t+1 t+1, t+2, ..., t+n autonomously.
    burnin = 1000
    auto_pred_target, mse = esn.predict_autonomous(eval_input, eval_target, burnin)

    # plot evaluation results
    fig = plt.figure(figsize=(16, 9))

    plt.suptitle(
        f"Mackey-Glass Autonomous Prediction (Reservoir nodes={esn.num_resv_nodes}, Leak rate={esn.leak_rate}, Lambda={_lambda}, MSE={mse[0]:.6f})"
    )
    T = range(len(pred_target))
    plt.plot(T, eval_target[:, 0], label="Actual", color='blue')
    plt.plot(T, auto_pred_target[:, 0], label="Predicted", color='orange', linestyle='--')
    plt.axvline(x=burnin, color="red", label="Start of Autonomous Prediction")
    plt.xlabel("t")
    plt.ylabel("X")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    np.random.seed(31)

    main()
