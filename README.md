# ReservoirComputing
Implementing Reservoir Computing Networks for Predicting Dynamic Systems

**NOTE**: This project is intended for educational and research purposes only. Although we've tested the code on certain cases, we cannot guarantee it is bug-free. So bug reports and pull requests are welcome.


# Environment and Requirements
* Python        3.10.6
* Numpy         1.23.4
* Scipy         1.10.1


# Lorenz System Prediction

Case 1: Given x_t, y_t, z_t, predict the corresponding x_t+1, y_t+1, z_t+1 components.

![Lorenz System - Prediction plot](/images/Lorenz_1.png)

![Lorenz System - Prediction 3D plot](/images/Lorenz_2.png)

Case 2: Given x_t, y_t, z_t, continue to predict x, y, z at t+1, t+2, ..., t+n autonomously.

![Lorenz System - Autonomous prediction](/images/Lorenz_3.png)


# Henon Map Prediction

Case 1: Given x_t, y_t, z_t, predict the corresponding x_t+1, y_t+1 components.

![Henon Map - Prediction plot](/images/HenonMap_1.png)


Case 2: Given x_t, y_t, continue to predict x, y at t+1, t+2, ..., t+n autonomously.

![Henon Map - Autonomous prediction](/images/HenonMap_2.png)


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