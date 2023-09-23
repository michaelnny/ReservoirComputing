import numpy as np
from scipy.integrate import solve_ivp


def generate_lorenz_data():
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


def generate_henon_map_data(iterations=10000):
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
