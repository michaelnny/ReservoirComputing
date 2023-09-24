from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from esn import ESN


def main(data: np.ndarray, scale: float = 100):
    """
    Train on SPY stock price to predict stock price.

    Parameters:
        data (numpy.ndarray): a 2D numpy.ndarray with shape of [T, 4], which contains historical price data.
        scale (float): scale ratio of the data before use to train the model (default 100).

    """

    assert scale >= 1

    data /= scale

    esn = ESN(num_inputs=4, num_outputs=4, num_resv_nodes=500, leak_rate=0.3)

    # train the model, give open, high, low, and close prices at time step t, predict open, high, low, and close prices at time step t+1
    train_size = int(len(data) * 0.7)
    train_input = data[:train_size, :]
    train_target = data[1 : train_size + 1, :]
    _lambda = 0.1
    esn.train(train_input, train_target, _lambda)

    # evaluate the model, give open, high, low, and close prices at time step t, predict open, high, low, and close prices at time step t+1
    eval_size = len(data) - train_size - 1
    eval_input = data[train_size : train_size + eval_size, :]
    eval_target = data[train_size + 1 : train_size + 1 + eval_size, :]
    pred_target, mse = esn.predict(eval_input, eval_target)

    # plot evaluation results
    fig, axes = plt.subplots(4, 1, figsize=(16, 9))
    fig.tight_layout(pad=4)
    fig.subplots_adjust(top=0.9)
    plt.suptitle(f"SPY Price Prediction (Reservoir nodes={esn.num_resv_nodes}, Leak rate={esn.leak_rate}, Lambda={_lambda})")
    T = range(len(pred_target))
    labels = ['Open', 'High', 'Low', 'Close']
    for i, label in enumerate(labels):
        ax = axes[i]
        ax.plot(T, eval_target[:, i] * scale, label="Actual", color='blue')
        ax.plot(T, pred_target[:, i] * scale, label="Predicted", color='orange', linestyle='--')
        ax.set_xlabel("Time (day)")
        ax.set_ylabel(f"{label}")
        ax.legend(loc="upper right")
        ax.set_title(f"SPY {label}, MSE={mse[i]:.6f}")

    plt.show()

    # evaluate the model on autonomous prediction
    burnin = int(eval_size * 0.8)
    auto_pred_target, mse = esn.predict_autonomous(eval_input, eval_target, burnin)

    # plot evaluation results
    fig, axes = plt.subplots(4, 1, figsize=(16, 9))
    fig.tight_layout(pad=4)
    fig.subplots_adjust(top=0.9)
    plt.suptitle(
        f"SPY Price Autonomous Prediction (Reservoir nodes={esn.num_resv_nodes}, Leak rate={esn.leak_rate}, Lambda={_lambda})"
    )
    T = range(len(pred_target))
    labels = ['Open', 'High', 'Low', 'Close']
    for i, label in enumerate(labels):
        ax = axes[i]
        ax.plot(T, eval_target[:, i] * scale, label="Actual", color='blue')
        ax.plot(T, auto_pred_target[:, i] * scale, label="Predicted", color='orange', linestyle='--')
        ax.axvline(x=burnin, color="red", label="Start of Autonomous Prediction")
        ax.set_xlabel("Time (day)")
        ax.set_ylabel(f"{label}")
        ax.legend(loc="upper right")
        ax.set_title(f"SPY {label}, MSE={mse[i]:.6f}")

    plt.show()


def load_stock_data_from_csv(csv_file: str, most_recent_years: int = 10) -> np.ndarray:
    """
    Load historical price data from csv file, and open keep the most recent N years' data.

    Parameters:
        csv_file (str): the csv file contains historical price data.
        most_recent_years (int): keep most recent N years of data (default 10).

    Returns:
        data (numpy.ndarray): a 2D numpy.ndarray with shape of [T, 4], where T is the number of days.
    """
    df = pd.read_csv(csv_file)

    # Select the desired columns
    selected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df[selected_columns]

    # Sort the DataFrame by the 'Date' column
    df.sort_values(by='Date', inplace=True)

    # Reset the index after sorting
    df.reset_index(drop=True, inplace=True)

    # Convert the 'Date' column to a datetime object
    df['Date'] = pd.to_datetime(df['Date'])

    # Find the most recent date in the dataset
    most_recent_date = df['Date'].max()

    # Calculate a date N years ago from the most recent date
    n_years_ago = most_recent_date - pd.DateOffset(years=most_recent_years)

    # Filter the DataFrame to keep only the rows within the last 5 years
    df_filtered = df[df['Date'] >= n_years_ago]

    # Reset the index after filtering
    df_filtered.reset_index(drop=True, inplace=True)

    # Select the columns you want to include in the NumPy array
    selected_columns = ['Open', 'High', 'Low', 'Close']
    data = df_filtered[selected_columns]

    # Convert the selected data to a NumPy array
    return data.to_numpy()  # [T, 4]


if __name__ == '__main__':
    np.random.seed(31)

    price_file = './data/SPY.csv'
    data = load_stock_data_from_csv(price_file)

    main(data)
