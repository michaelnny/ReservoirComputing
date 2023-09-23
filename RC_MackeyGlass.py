import numpy as np
import matplotlib.pyplot as plt

from esn import ESN
from data_generator import generate_mackey_glass_data


def main():
    """Train on Mackey-Glass predict X component"""
    data = generate_mackey_glass_data()  # [10000, ]

    data = np.expand_dims(data, 1)  # [10000, 1]

    esn = ESN(num_inputs=1, num_outputs=1, num_resv_nodes=250, leak_rate=0.75)

    # train the model, give x_t to predict x_t+1
    train_size = 5000
    train_input = data[:train_size, :]
    train_target = data[1 : train_size + 1, :]
    _lambda = 0.5
    esn.train(train_input, train_target, _lambda)

    # evaluate the model, give x_t to predict x_tp1
    eval_size = 2000
    eval_input = data[train_size : train_size + eval_size, :]
    eval_target = data[train_size + 1 : train_size + 1 + eval_size, :]
    pred_target, mse = esn.predict(eval_input, eval_target)

    # plot evaluation results
    fig = plt.figure(figsize=(16, 9))

    plt.suptitle(
        f"Mackey-Glass Prediction (Reservoir nodes={esn.num_resv_nodes}, Leak rate={esn.leak_rate}, Lambda={_lambda})"
    )
    T = range(len(pred_target))

    plt.plot(T, eval_target[:, 0], label="Actual", color='blue')
    plt.plot(T, pred_target[:, 0], label="Predicted", color='orange', linestyle='--')
    plt.xlabel("t")
    plt.ylabel("X")
    plt.legend(loc="upper right")
    plt.title(f"Mackey-Glass X, MSE={mse[0]:.6f}")

    plt.show()

    # evaluate the model on autonomous prediction
    burnin = 1000
    auto_pred_target, mse = esn.predict_autonomous(eval_input, eval_target, burnin)

    # plot evaluation results
    fig = plt.figure(figsize=(16, 9))

    plt.suptitle(
        f"Mackey-Glass Autonomous Prediction (Reservoir nodes={esn.num_resv_nodes}, Leak rate={esn.leak_rate}, Lambda={_lambda})"
    )
    T = range(len(pred_target))
    plt.plot(T, eval_target[:, 0], label="Actual", color='blue')
    plt.plot(T, auto_pred_target[:, 0], label="Predicted", color='orange', linestyle='--')
    plt.axvline(x=burnin, color="red", label="Start of Autonomous Prediction")
    plt.xlabel("t")
    plt.ylabel("X")
    plt.legend(loc="upper right")
    plt.title(f"Mackey-Glass X, MSE={mse[0]:.6f}")

    plt.show()


if __name__ == '__main__':
    np.random.seed(31)

    main()
