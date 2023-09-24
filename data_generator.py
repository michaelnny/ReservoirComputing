import numpy as np
from scipy.integrate import solve_ivp


def generate_lorenz_data() -> np.ndarray:
    # Lorenz system parameters
    u0 = [1.0, 0.0, 0.0]
    tspan = (0.0, 200.0)
    p = [10.0, 28.0, 8 / 3]

    # Define Lorenz system
    def lorenz(t, u):
        du = np.zeros(3)
        du[0] = p[0] * (u[1] - u[0])
        du[1] = u[0] * (p[1] - u[2]) - u[1]
        du[2] = u[0] * u[1] - p[2] * u[2]
        return du

    # Solve and take data
    sol = solve_ivp(lorenz, tspan, u0, t_eval=np.linspace(tspan[0], tspan[1], int(200 / 0.02)))

    data = sol.y  # [3, 10000]

    return data.T  # [10000, 3]


def generate_henon_map_data(iterations=10000) -> np.ndarray:
    def henon_map(x, y, a=1.4, b=0.3):
        new_x = 1 - a * x**2 + y
        new_y = b * x
        return new_x, new_y

    x_values = [0]
    y_values = [0]

    # Generate Henon map data
    for i in range(iterations):
        x, y = henon_map(x_values[-1], y_values[-1])
        x_values.append(x)
        y_values.append(y)

    data = np.stack([x_values[1:], y_values[1:]]).T  # [10000, 2]
    return data


# code adapted from:
# https://github.com/manu-mannattil/nolitsa/blob/master/nolitsa/data.py
def generate_mackey_glass_data(
    length=10000, x0=None, a=0.2, b=0.1, c=10.0, tau=23.0, n=1000, sample=0.46, discard=250
) -> np.ndarray:
    """Generate time series using the Mackey-Glass equation.

    Generates time series using the discrete approximation of the
    Mackey-Glass delay differential equation described by Grassberger &
    Procaccia (1983).

    Parameters:
    length : int, optional (default = 10000)
        Length of the time series to be generated.
    x0 : array, optional (default = random)
        Initial condition for the discrete map.  Should be of length n.
    a : float, optional (default = 0.2)
        Constant a in the Mackey-Glass equation.
    b : float, optional (default = 0.1)
        Constant b in the Mackey-Glass equation.
    c : float, optional (default = 10.0)
        Constant c in the Mackey-Glass equation.
    tau : float, optional (default = 23.0)
        Time delay in the Mackey-Glass equation.
    n : int, optional (default = 1000)
        The number of discrete steps into which the interval between
        t and t + tau should be divided.  This results in a time
        step of tau/n and an n + 1 dimensional map.
    sample : float, optional (default = 0.46)
        Sampling step of the time series.  It is useful to pick
        something between tau/100 and tau/10, with tau/sample being
        a factor of n.  This will make sure that there are only whole
        number indices.
    discard : int, optional (default = 250)
        Number of n-steps to discard in order to eliminate transients.
        A total of n*discard steps will be discarded.

    Returns
    -------
    x : array
        Array containing the time series.
    """
    sample = int(n * sample / tau)
    grids = n * discard + sample * length
    x = np.empty(grids)

    if not x0:
        x[:n] = 0.5 + 0.05 * (-1 + 2 * np.random.random(n))
    else:
        x[:n] = x0

    A = (2 * n - b * tau) / (2 * n + b * tau)
    B = a * tau / (2 * n + b * tau)

    for i in range(n - 1, grids - 1):
        x[i + 1] = A * x[i] + B * (x[i - n] / (1 + x[i - n] ** c) + x[i - n + 1] / (1 + x[i - n + 1] ** c))
    data = x[n * discard :: sample]

    return np.expand_dims(data, 1)  # [10000, 1]


def generate_stock_price_random_walk(
    num_days: int = 1000, initial_price: float = 100, drift: float = 0.0, volatility: float = 0.2
) -> np.ndarray:
    """
    Generate synthetic stock price data using random walk.

    Parameters:
        num_days (int): number of days (default 1000)
        initial_price (float): initial stock price (default 100)
        drift (float): drift or average daily return (default 0)
        volatility (float): volatility or standard deviation of daily returns (default 0.2)

    Returns:
        price (numpy.ndarray): a 2D numpy.ndarray with shape of [num_days, 1] contains price data over num_days
    """

    # Generate synthetic stock price data
    daily_returns = np.random.normal(drift / num_days, volatility / np.sqrt(num_days), num_days)
    price = initial_price * np.exp(np.cumsum(daily_returns))

    return np.expand_dims(price, 1)  # [num_days, 1]


def generate_stock_price_gbm(num_days=1000, initial_price=100, mu=0.1, sigma=0.2, dt=1 / 252) -> np.ndarray:
    """
    Generate synthetic stock price data using GBM

    Parameters:
        num_days (int): number of days (default 1000)
        initial_price (float): initial stock price (default 100)
        mu (float): annual drift or average return (default 0.1)
        sigma (float): annual volatility or standard deviation of return (default 0.2)
        dt (float): time step (assuming trading 252 days a year) (default 1 / 252)

    Returns:
        price (numpy.ndarray): a 2D numpy.ndarray with shape of [num_days, 1] contains price data over num_days
    """

    price = [initial_price]
    for _ in range(num_days - 1):
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * np.random.normal(0, 1)
        price.append(price[-1] * np.exp(drift + diffusion))

    return np.expand_dims(price, 1)  # [num_days, 1]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    lorenz_data = generate_lorenz_data()

    ax = plt.figure(figsize=(12, 6)).add_subplot(projection='3d')

    ax.plot(lorenz_data[:, 0], lorenz_data[:, 1], lorenz_data[:, 2])
    ax.set_title('Lorenz System - 3D Plot')
    ax.set_xlabel('Component X')
    ax.set_ylabel('Component Y')
    ax.set_zlabel('Component Z')

    plt.tight_layout()
    plt.show()

    henonmap_data = generate_henon_map_data()

    # Plot the Henon map
    plt.figure(figsize=(8, 6))
    plt.scatter(henonmap_data[:, 0], henonmap_data[:, 1], s=1, c='blue', marker='.')
    plt.title('Henon Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    mackey_glass_data = generate_mackey_glass_data()

    # Plot the time series
    plt.figure(figsize=(10, 6))
    plt.plot(mackey_glass_data)
    plt.title("Mackey-Glass Time Series")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()
