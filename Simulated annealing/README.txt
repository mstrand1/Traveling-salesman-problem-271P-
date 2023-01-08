This zip file includes the .py file for the Python function Simulated Annealing. To use the functions open them in Visual Studio Code, Juypiter, or any other Python compiler and run to initialize the function.

Required packages:
numpy
pandas
matplotlib
time
random

Inputs:
- dist_mat, numpy array: nxn symmetric numpy array
- max_iter, int: stopping criteria
- X_0, float: initial desired acceptance
- reheat_threshold, int: number of stagnant iterations before reheating. Usually ~20% of runtime (maxiter//5)
- apps, int: number of times to repeat a randomizing action

Returns:
- best_dist: distance of the best found path
- cpu_time: total runtime at termination

Example usage:

dist, cpu_time = SA(distance_matrix, max_iter = 10**6, X_0 = 0.85, reheat_threshold = 50000, apps = 1)
