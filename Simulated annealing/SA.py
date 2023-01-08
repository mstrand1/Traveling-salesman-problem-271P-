#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from time import process_time


def SA(dist_mat, max_iter, X_0, reheat_threshold, apps):
    
    # dist_mat: nxn symmetric numpy array
    # max_iter: stopping criteria
    # X_0: initial desired acceptance
    # reheat_threshold: number of stagnant iterations before reheating. Usually ~20% of runtime (maxiter//5)
    # apps: number of times to repeat a randomizing action

    t1_start = process_time() # cpu timer
    p_array = []
    distance_array = []
    temp_array = []
    swap_types = ['Swap 2', 'Insert', 'Swap Subroute', 'Invert'] # random actions

    # Total distance for current path including return to start
    def objective_function(state):
        cost = 0
        for i in range(len(state) - 1):
            cost += dist_mat[state[i], state[i + 1]]
        cost += dist_mat[state[-1], state[0]] # states do not carry the return node but we always account for it
        return cost

    # construct initial path
    def init_path():
        # Initial path: 0 -> 1, 1 -> 2, ... N-1 -> N -> 0
        initial = [i for i in range(len(dist_mat))]
        return initial

    # Computing the Initial Temperature of Simulated Annealing, Walid 2003
    # idea: sample SS bad transitions and calculate the acceptance ratio for this sample: iteratively reduce T until desired prob X_0
    def init_temp(X_0):

        s_trials = 0 # sample trials
        E_after = []
        E_before = []

        SS = int(max_iter / 3)  # sampel size: trains such that we get X_0 acceptance for the first 10% of runtime

        # generate SS random positive transitions
        current_sample_path = init_path()
        current_sample_distance = objective_function(current_sample_path)

        # proceed as like in normal SA
        while s_trials < SS:
            new_sample_path = choose_apply_action(current_sample_path)
            new_sample_distance = objective_function(new_sample_path)
            E = current_sample_distance - new_sample_distance

            if E < 0:
                E_before.append(current_sample_distance)  # Emin
                E_after.append(new_sample_distance)  # Emax
                s_trials += 1

                if np.random.binomial(1, X_0) == 1:
                    current_sample_distance = new_sample_distance
                    current_sample_path = new_sample_path
            else:
                current_sample_path = new_sample_path
                current_sample_distance = new_sample_distance

        # Iteratively adjust initial temperature until acceptance ratio (X_Tn) is approximately equal X_0
        start = True
        X_Tn = 0
        Tn = 10 * len(dist_mat)  # initial starting temperature (only need Tn > 1)

        # rescale initial temp until within margin of error of X_0
        while abs(X_Tn - X_0) > 0.01:

            num = 0
            denom = 0
            
            # without this check we terminate on the first iteration
            if start:
                X_Tn = X_0
                start = False

            Tn = abs(Tn * (np.log(X_Tn) / np.log(X_0)))

            for i in range(SS):
                num += np.exp(-E_after[i] / Tn)
                denom += np.exp(-E_before[i] / Tn)

            X_Tn = num / denom

        return Tn

    # path changing actions (4)
    def switch_two(state):
        # ex: a,b = 2,5
        # [0, 1, 2, 3, 4, 5, 6, 0] -> [0, 1, 5, 3, 4, 2, 6,0]
        for i in range(apps):
            a = random.randint(0, len(state) - 1)
            b = random.randint(0, len(state) - 1)
            while b == a:
                b = random.randint(0, len(state) - 1)
            state[a], state[b] = state[b], state[a]
        return state

    def invert_path_between(state):
        # ex: a,b = 2,5
        # [0, 1, 2, 3, 4, 5, 6, 0] -> [0, 1, 4, 3, 2, 5, 6, 0]
        for i in range(apps):
            a = random.randint(0, len(state) - 1)
            b = random.randint(0, len(state) - 1)
            while b == a:
                b = random.randint(0, len(state) - 1)
            if b < a:
                a, b = b, a
            state[a:b] = state[a:b][::-1]  # reverse sublist
        return state

    def insert_random(state):
        # ex: a,b = 2,5
        # [0, 1, 2, 3, 4, 5, 6, 0] -> [0, 1, 3, 4, 5, 2, 6, 0]
        for i in range(apps):
            a = random.randint(0, len(state) - 1)
            b = random.randint(0, len(state) - 1)
            while b == a:
                b = random.randint(0, len(state) - 1)
            city = state.pop(a)
            state.insert(b, city)
        return state

    def swap_subroute(state):
        # move a whole subroute
        for i in range(apps):
            a = random.randint(0, len(state) - 1)
            b = random.randint(0, len(state) - 1)
            while b == a:
                b = random.randint(0, len(state) - 1)
            if b < a:
                a, b = b, a
            subroute = state[a:b]
            state = [state[x] for x in range(len(state)) if state[x] not in subroute]
            c = random.randint(0, max(1, len(state) - 1))
            state[c:c] = subroute
        return state

    # randomly choose and apply an action
    def choose_apply_action(path):

        random_type = random.choice(swap_types)

        if random_type == 'Swap 2':
            new_path = switch_two(path)
        elif random_type == 'Insert':
            new_path = insert_random(path)
        elif random_type == 'Swap Subroute':
            new_path = swap_subroute(path)
        elif random_type == 'Invert':
            new_path = invert_path_between(path)
        return new_path

    
    T = init_temp(X_0)
    alpha = 1 - 1/max_iter**X_0  # geometric scalar to lower temp slowly

    heat = 0
    stagnation = 0
    reheat = 1

    path = init_path()
    current_distance = objective_function(path)
    best_distance = current_distance
    best_path = path
    best_path.append(path[0])

    # begin algorithm
    for t in range(max_iter):
        
        heat += 1

        # find new path and path's distance
        new_path = choose_apply_action(path)
        new_distance = objective_function(new_path)
        E = current_distance - new_distance

        # always track best path
        if new_distance < best_distance:
            best_distance = new_distance
            best_path = new_path

        # new path is shorter
        if E > 0:
            current_distance = new_distance
            path = new_path
            p_array.append(0)

        else:

            # temperature function and probability (most sensitive part of the algorithm)
            temp = T * alpha ** heat
            p = np.exp(E / temp) / (reheat ** 0.8)  # penalize reheating

            # roll whether to keep longer new path anyway
            if np.random.binomial(1, p) == 1:
                current_distance = new_distance
                path = new_path
                stagnation = 0

            # increase stagnation factor: the longer we go with no bad rolls the more stagnated we become
            else:

                stagnation += 1

                # stagnation mechanic: reset temperature but at a reduced probability of acceptance
                if stagnation >= reheat_threshold:
                    heat = 100 * reheat
                    reheat_threshold *= 1.1  # increase threshold such that reheats become rarer each time
                    stagnation = 0
                    reheat += 1

            # if time is over 10 mins, return best dist so far
            if process_time() - t1_start >= 600:
                return best_distance, 600

            distance_array.append(best_distance)
            temp_array.append(temp)
            p_array.append(p)

    best_path.append(best_path[0]) # return to start city

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.tight_layout(pad=5)

    plt.subplot(1, 3, 1)
    plt.plot(distance_array, linewidth=5.5)
    plt.title('Best path distance',fontsize=20)
    plt.ylabel('Distance',fontsize=20)
    plt.xlabel('Iteration',fontsize=20)

    plt.subplot(1, 3, 2)
    plt.scatter(np.arange(len(p_array)), p_array, s=0.009, c=p_array, cmap='viridis_r')
    plt.title('Chance of bad accept',fontsize=20)
    plt.ylabel('Probability',fontsize=20)
    plt.xlabel('Iteration',fontsize=20)

    plt.subplot(1, 3, 3)
    plt.plot(temp_array, color='red', linewidth=5.5)
    plt.title('Temperature over time',fontsize=20)
    plt.ylabel('T',fontsize=20)
    plt.xlabel('Iteration',fontsize=20)
    plt.show()
    
    t1_stop = process_time()
    cpu_time = t1_stop - t1_start
    
    return best_distance, cpu_time