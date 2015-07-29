#!/usr/bin/env python
#-*-coding: utf-8 -*-

import numpy as np


## PARAMETERS
T = 10000               # Finite Horizon
K = 3                   # Number of recommended items (among L >= K items)
weights = [1./K] * K    # Multinomial distribution over rank stopping times
taus = np.random.choice(K, T, p=weights)    # Stopping times for T rounds

## ARMS
L = 10                  # Number of arms (items)
arms = MAB(L)           # Create an object of L arms

#for i in xrange(T):
#    
