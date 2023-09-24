# ReservoirComputing
Implementing Reservoir Computing Networks for Predicting Dynamic Systems

**NOTE**: This project is intended for educational and research purposes only. Although we've tested the code on certain cases, we cannot guarantee it is bug-free. So bug reports and pull requests are welcome.


# Environment and Requirements
* Python        3.10.6
* Numpy         1.23.4
* Scipy         1.10.1


# Lorenz System Prediction

Case 1: Given x, y, z components at time step t, predict the x, y, z components at time step t+1.

![Lorenz System - Prediction plot](/images/Lorenz_1.png)

![Lorenz System - Prediction 3D plot](/images/Lorenz_2.png)

Case 2: Given x, y, z components at time step t, predict x, y, z components at time step t+1 t+1, t+2, ..., t+n autonomously.

![Lorenz System - Autonomous prediction](/images/Lorenz_3.png)


# Henon Map Prediction

Case 1: Given x, y components at time step t, predict the x, y components at time step t+1.

![Henon Map - Prediction plot](/images/HenonMap_1.png)


Case 2: Given x, y components at time step t, predict x, y components at time step t+1 t+1, t+2, ..., t+n autonomously.

![Henon Map - Autonomous prediction](/images/HenonMap_2.png)



# Stock Price Prediction

Case 1: Given price at time step t, predict price as time step t+1, all using synthetic stock price data

![Synthetic Stock Price - Prediction plot](/images/RandomStock_1.png)


Case 2: Given open, high, low, and close prices at time step t, predict open, high, low, and close prices at time step t+1, using SPY historical price data

![SPY Stock Price - Prediction plot](/images/SPY_1.png)


# License
This project is licensed under the MIT License, see the LICENSE file for details


# Citing our work
If you reference or use our project in your research, please cite:

```
@software{ReservoirComputing2023github,
  title = {{ReservoirComputing}: Implementing Reservoir Computing Networks for Predicting Dynamic Systems},
  author = {Michael Hu},
  url = {https://github.com/michaelnny/ReservoirComputing},
  version = {1.0.0},
  year = {2023},
}
```