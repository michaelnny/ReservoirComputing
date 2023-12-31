import numpy as np
import matplotlib.pyplot as plt

from esn import ESN
from data_generator import generate_stock_price_random_walk, generate_stock_price_gbm


def main():
    """Train on synthetic stock price to predict stock price"""
    # data = generate_stock_price_random_walk(5000)  # [5000, 1]
    data = generate_stock_price_gbm(5000)  # [5000, 1]

    scale = 100

    data /= scale

    esn = ESN(num_inputs=1, num_outputs=1, num_resv_nodes=200, leak_rate=0.3)

    # train the model, given price at time step t, predict price as time step t+1
    train_size = 4000
    train_input = data[:train_size, :]
    train_target = data[1 : train_size + 1, :]
    _lambda = 0.1
    esn.train(train_input, train_target, _lambda)

    # evaluate the model, given price at time step t, predict price as time step t+1
    eval_size = 500
    eval_input = data[train_size : train_size + eval_size, :]
    eval_target = data[train_size + 1 : train_size + 1 + eval_size, :]
    pred_target, mse = esn.predict(eval_input, eval_target)

    # plot evaluation results
    fig = plt.figure(figsize=(16, 9))

    plt.suptitle(
        f"Synthetic Stock Price Prediction (Reservoir nodes={esn.num_resv_nodes}, Leak rate={esn.leak_rate}, Lambda={_lambda}, MSE={mse[0]:.6f})"
    )
    T = range(len(pred_target))

    plt.plot(T, eval_target[:, 0], label="Actual", color='blue')
    plt.plot(T, pred_target[:, 0], label="Predicted", color='orange', linestyle='--')
    plt.xlabel("t")
    plt.ylabel("X")
    plt.legend(loc="upper right")
    plt.show()

    # evaluate the model on autonomous prediction
    # given price at time step t, predict price at time step t+1 t+1, t+2, ..., t+n autonomously.
    burnin = 250
    auto_pred_target, mse = esn.predict_autonomous(eval_input, eval_target, burnin)

    # plot evaluation results
    fig = plt.figure(figsize=(16, 9))

    plt.suptitle(
        f"Synthetic Stock Price Autonomous Prediction (Reservoir nodes={esn.num_resv_nodes}, Leak rate={esn.leak_rate}, Lambda={_lambda}, MSE={mse[0]:.6f})"
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
