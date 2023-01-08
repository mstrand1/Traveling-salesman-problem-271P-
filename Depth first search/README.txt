This zip file includes the .py file for the Python function Branch and Bound Depth First Search. To use the functions open them in Visual Studio Code, Juypiter, or any other Python compiler and run to initialize the function.

Required packages:
numpy
pandas
matplotlib
time
random
math

Inputs:
- dist_mat, numpy array: nxn symmetric numpy array
- heuristic, str: "NN" or "2opt"
- cluster, bool: True for clustered subproblems
- n_clusters, int: number of sub problems to split into
- max_bottoms, int: stopping criteria - number of times we've reached the bottom
- opt_iter, int: stopping criterion for 2opt
- mean, float: average distance
- sigma, float: standard deviation of distance

Returns:
- best_dist: distance of the best found path
- cpu_time: total runtime at termination

Example usage:

dist, cpu_time = BNB_DFS(distance_matrix, heuristic = "2opt", cluster = True, n_clusters = 5, h_freq = 10, max_bottoms = 200, opt_iter = 50, mean = 100, sigma = 10)
