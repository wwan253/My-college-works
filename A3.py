# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 11:31:29 2021

@author: Weixu Wang
"""

import pandas as pd
import math
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from tqdm import tqdm
import time
import re

np.random.seed(42)
sp.random.seed(42)


# ========== Read in data ==========
new_data = False

data = pd.read_csv('web-Google.txt', skiprows=3, delim_whitespace=True)
data['temp'] = data['#']
data.ToNodeId = data.FromNodeId
data.FromNodeId = data.temp
del data['#'], data['temp']

nodes = 875713

count = 0
old_values = np.unique(data[['ToNodeId', 'FromNodeId']].values)
index = dict()
inverted_index = dict()
for x in old_values:
    if x not in index:
        index[x] = count
        inverted_index[count] = x
        count += 1

data.FromNodeId = [index[x] for x in data.FromNodeId]
data.ToNodeId = [index[x] for x in data.ToNodeId]

value_counts = data.FromNodeId.value_counts().to_dict()
data['edge_weights'] = np.array([1/value_counts[x] for x in data.FromNodeId])


if new_data:
    spm = sp.sparse.dok_matrix((nodes, nodes))
    for _, row in tqdm(data.iterrows()):
        spm[row.ToNodeId, row.FromNodeId] = row.edge_weights

    spm = spm.tocsr()
    sp.sparse.save_npz('sparse_matrix.npz', spm)

else:
    spm = sp.sparse.load_npz('sparse_matrix.npz')



# ========== Question 1 ==========
M = spm
epsilon = 0.02
r = np.ones(nodes)*(1/nodes)

def pagerank_1(M, r, epsilon):
    i = 0
    while True:
        i += 1
        r_prev = r
        r = M.dot(r)
        dist = sum(abs(r - r_prev))
        if dist < epsilon:
            return r, dist, i

start = time.time()
r, dist, i = pagerank_1(M,r,epsilon)
time_taken = time.time() - start

print(f'Time Taken: {time_taken}, n_iterations: {i}')
print()

results = zip(r, range(nodes + 1))
r_top = sorted(results, reverse=True, key=lambda x: x[0])
for j in range(10):
    print(f'Node={inverted_index[r_top[j][1]]}: Value={r_top[j][0]:.7}')


# ========== Question 1 Results ==========

# Time Taken: 12.594843626022339, n_iterations: 62
#
# Node=747106: Value=0.0006177867
# Node=24576: Value=0.0006065543
# Node=370344: Value=0.0006065543
# Node=544138: Value=0.0006065543
# Node=577518: Value=0.0003875678
# Node=587617: Value=0.0003482264
# Node=671168: Value=0.00030959
# Node=791675: Value=0.0002710505
# Node=873996: Value=0.0002702807
# Node=914474: Value=0.0002597883




# ========== Question 2 ==========
M = spm
epsilon = 0.02
r = np.ones(nodes)*(1/nodes)

def pagerank_2(M, r, epsilon, beta=0.9):
    constant = np.ones((M.shape[0])) * 1/M.shape[0]
    i = 0
    while True:
        i += 1
        r_prev = r
        r = beta*M.dot(r) + (1 - beta)*constant 
        dist = sum(abs(r - r_prev))
        if dist < epsilon:
            return r, dist, i


# Run with default value for beta
start = time.time()
r, dist, i = pagerank_2(M,r,epsilon)
time_taken = time.time() - start

print(f'Time Taken: {time_taken}, n_iterations: {i}')
print()

results = zip(r, range(nodes + 1))
r_top = sorted(results, reverse=True, key=lambda x: x[0])
for j in range(10):
    print(f'Node={inverted_index[r_top[j][1]]}: Value={r_top[j][0]:.7}')
print()


# Run with varying beta
for beta in [1, 0.9, 0.8, 0.7, 0.6, 0.5]:
    r = np.ones(nodes)*(1/nodes)
    *_, i = pagerank_2(M, r, epsilon, beta=beta)
    print(f'Beta={beta}, n_iterations={i}')
    


# ========== Question 2 Results ==========

# Time Taken: 2.479174852371216, n_iterations: 11
#
# Node=41909: Value=0.000679151
# Node=597621: Value=0.0006452959
# Node=537039: Value=0.0006219468
# Node=163075: Value=0.0006128389
# Node=384666: Value=0.000571944
# Node=504140: Value=0.0005406181
# Node=486980: Value=0.0005179286
# Node=558791: Value=0.0005011795
# Node=32163: Value=0.0005010396
# Node=605856: Value=0.0004843358
#
# Beta=1, n_iterations=62
# Beta=0.9, n_iterations=11
# Beta=0.8, n_iterations=7
# Beta=0.7, n_iterations=6
# Beta=0.6, n_iterations=5
# Beta=0.5, n_iterations=4




# ========== Question 3 ==========
# Question 3a
for beta in [1, 0.9, 0.8, 0.7, 0.6, 0.5]:
    r = np.ones(nodes)*(1/nodes)
    r, *_ = pagerank_2(M, r, epsilon, beta=beta)
    print(f'Beta={beta}, Leak={1 - sum(r)}')
print()
    
    
# Question 3b
def pagerank_3(M, r, epsilon, beta=0.9):
    i = 0
    constant = np.ones((M.shape[0])) * 1/M.shape[0]
    
    while True:
        i += 1
        r_prev = r
        r = beta*M.dot(r) + (1 - beta)*constant 
        print(f'Iteration={i}, Leak={1 - sum(r):.5}')
        
        dist = sum(abs(r - r_prev))
        if dist < epsilon:
            return r

r = np.ones(nodes)*(1/nodes)
r = pagerank_3(M, r, epsilon)


# ========== Question 3 Results ==========

# Beta=1, Leak=0.8065158568547688
# Beta=0.9, Leak=0.34094781258696216
# Beta=0.8, Leak=0.23887730964298293
# Beta=0.7, Leak=0.1826486642264442
# Beta=0.6, Leak=0.14038973768469554
# Beta=0.5, Leak=0.10664028680911664
#
# Iteration=1, Leak=0.14004
# Iteration=2, Leak=0.20767
# Iteration=3, Leak=0.24136
# Iteration=4, Leak=0.26435
# Iteration=5, Leak=0.28294
# Iteration=6, Leak=0.29767
# Iteration=7, Leak=0.30991
# Iteration=8, Leak=0.31987
# Iteration=9, Leak=0.32822
# Iteration=10, Leak=0.33513
# Iteration=11, Leak=0.34095





# ========== Question 4 ==========
def pagerank_4(M, r, epsilon, beta=0.9):
    constant = np.ones((M.shape[0])) * 1/M.shape[0]
    i = 0
    while True:
        i += 1
        r_prev = r
        r = beta*M.dot(r) + (1 - beta)*constant 
        leak = 1 - sum(r)
        r = r + constant*leak
        
        dist = sum(abs(r - r_prev))
        if dist < epsilon:
            return r, i


r = np.ones(nodes)*(1/nodes)

start = time.time()
r, i = pagerank_4(M, r, epsilon)
time_taken = time.time() - start

print(f'Time Taken: {time_taken}, n_iterations: {i}, Leak: {1 - sum(r)}')
print()


results = zip(r, range(nodes + 1))
r_top = sorted(results, reverse=True, key=lambda x: x[0])
for j in range(10):
    print(f'Node={inverted_index[r_top[j][1]]}: Value={r_top[j][0]:.7}')
    
    


# ========== Question 4 Results ==========

# Time Taken: 4.089877128601074, n_iterations: 11, Leak: -3.079092536495409e-12
#
# Node=41909: Value=0.001008516
# Node=597621: Value=0.0009705537
# Node=537039: Value=0.0009380558
# Node=163075: Value=0.0009314725
# Node=384666: Value=0.0008509553
# Node=504140: Value=0.000811377
# Node=486980: Value=0.0007834381
# Node=558791: Value=0.0007573728
# Node=32163: Value=0.0007537884
# Node=605856: Value=0.0007401293

