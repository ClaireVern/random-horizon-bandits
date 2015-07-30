#!/usr/bin/env python
#-*-coding: utf-8 -*-

import numpy as np
from bandits import MAB, EstimatedMAB


def play(A_t, tau_t, arms, estimations):
    """
    This function plays the first tau_t arms in A_t. It receives the rewards
    from arms (model) and update the estimations (estimations).

    Parameters
    ----------
    A_t : list (length K)
        List of selected arm indices (ordered).
    tau_t : int
        Number of items rated by the user.
    arms : list (length L)
        List of Arm objects (BernoulliArm, BetaArm, ...) corresponding
        to the hidden distributions.
    estimations : list (length L)
        List of EstimatedArm objects corresponding to the estimations of arms.
    """
    round_reward = 0.
    for index in A_t[:tau_t]:
        reward = arms.play(index)
        estimations.updateEstimatedArm(index, reward)
        round_reward += reward
    return round_reward


## PARAMETERS
T = 10000               # Finite Horizon
K = 3                   # Number of recommended items (among L >= K items)
weights = [1./K] * K    # Multinomial distribution over rank stopping times
# Stopping times for T rounds
taus = map(lambda x: x+1, np.random.choice(K, T, p=weights))

## ARMS
L = 11                  # Number of arms (items)
params = {
    "bernoulli": 0.5,
    "bernoulli": 0.55,
    "bernoulli": 0.45,
    "bernoulli": 0.6,
    "bernoulli": 0.58,
    "bernoulli": 0.2,
    "beta": [0.5, 0.5],     # mean -> 0.5
    "beta": [4., 3.],       # mean -> 0.571
    "beta": [0.45, 0.34],   # mean -> 0.57
    "beta": [2, 2],         # mean -> 0.5
    "exp": 1.85185          # mean -> 0.54
    }

# Create an object of L arms (corresponds to model)
arms = MAB(L, params=params)
rewards = list()    # List of rewards obtained by the policy

"""
Main algorithm
"""

# List of estimations for each arms (see EstimatedArm class)
estimations = EstimatedMAB(L)
# Initialization
for t in xrange(L):
    # Initialization: we present lists A_t = [0, 1, 2], [1, 2, 3], ...
    A_t = [(t+i)%L for i in xrange(K)]
    round_reward = play(A_t, taus[t], arms, estimations)
    rewards.append(round_reward)

for t in xrange(L, T):
    estimations.computeUCBs()
    A_t = estimations.selectArms(K)
    round_reward = play(A_t, taus[t], arms, estimations)
    rewards.append(round_reward)

"""
Plot here
"""

# 1. order the arms in descending order of expectations
# 2. compute the expected round rewards
# 3. plot the regret curve
