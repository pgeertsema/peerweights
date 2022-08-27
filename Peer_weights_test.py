# Python code to represent LightGBM predictions as a linear combination of training data target values
# See section A.3 in "Relative Valuation with Machine Learning", Geertsema & Lu (2022)

# Copyright (C) Paul Geertsema and Helen Lu 2022

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

#---------------------------------------------------------------
# imports
#---------------------------------------------------------------

import numpy as np
from numpy import set_printoptions
import lightgbm as lgb
from sklearn.datasets import make_friedman1
from sklearn.datasets import make_regression
from numba import jit

set_printoptions(precision=4)
np.set_printoptions(linewidth=100)

#---------------------------------------------------------------
# model settings
#---------------------------------------------------------------

# test settings - change as needed
OBSERVATIONS = 100
LEARNING_RATE = 0.1
TREES = 12
LEAVES = 5

# note first LightGMB "tree" is the training sample average
# note python arrays are zero based, so tree[0] is the first tree
# tree[0] is conceptually equivalent to a tree with 1 node 

#---------------------------------------------------------------
# tree extraction utility
#---------------------------------------------------------------

def tree_prediction(model, data, train_sample_average, tree_iteration, learning_rate):
    '''
    extracts individual tree predictions from GBM model
    '''
    if tree_iteration == 0:
        tree_prediction = train_sample_average
    elif tree_iteration == 1:
        # prediction includes effect of sample average, need to reverse this
        gbm1_prediction = model.predict(data, start_iteration = 0, num_iteration = 1)    
        tree_prediction = (gbm1_prediction - train_sample_average)/learning_rate
    else:
        tree_prediction = model.predict(data, start_iteration = tree_iteration-1, num_iteration = 1) / learning_rate
    return tree_prediction

#---------------------------------------------------------------
# membership matrix utility
#---------------------------------------------------------------

@jit(nopython=True)
def getD(tree):
    '''
    utility function to create leaf membership matrix D from leaf membership vector tree
    element d_f,c in D takes value of 1 IFF observations f and c are allocated to the same leaf
    that is, tree[f] == tree[c], otherwise 0
    '''
    # tree contains a list of predicted values
    tree_len = len(tree)
    D = np.full((tree_len, tree_len), np.nan)
    for f in range(0,tree_len):
        for c in range(0,tree_len):
            same = (tree[f] == tree[c])
            #print(f" [{f},{c}] is {same}")
            D[f][c] = same
    return D


#---------------------------------------------------------------
# data
#---------------------------------------------------------------

# non-linear synthetic data
X, y = make_friedman1(n_samples = OBSERVATIONS, n_features = 5, noise = 0, random_state= 42)
print(X, y)

#---------------------------------------------------------------
# model parameters
#---------------------------------------------------------------

params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": LEAVES,
    "verbose": 1,
    "min_data": 2,
    "learning_rate": LEARNING_RATE,    
}

#---------------------------------------------------------------
# train model
#---------------------------------------------------------------

data = lgb.Dataset(X, label=y)
model = lgb.train(params, data, num_boost_round=TREES-1)
y_hat = model.predict(X)
y_hat
y_hat.shape

instance_leaf_membership = model.predict(X, pred_leaf=True)
instance_leaf_membership

#---------------------------------------------------------------
# print trees
#---------------------------------------------------------------

for index in range(1, TREES):
    print(index)
    graph = lgb.create_tree_digraph(booster=model, tree_index = index-1,
                    show_info=['internal_count', 'leaf_count', 
                        'leaf_weight', 'data_percentage', 'internal_value'],
                    name="Tree"+str(index))
    graph.render(view=True)

#---------------------------------------------------------------
# extract tree predictions from GBM model
#---------------------------------------------------------------

train_average = y.mean()
tree = []
tp = np.full(OBSERVATIONS, train_average)
tree.append(tp)
print("tree len after append = ", len(tree))
for index in range(1, TREES):
    print("index = ", index)
    tp = tree_prediction(model=model, data=X, train_sample_average=train_average, tree_iteration=index, learning_rate=LEARNING_RATE)
    tree.append(tp)
    print("tree len after append = ", len(tree))


#---------------------------------------------------------------
# print tree predictions
#---------------------------------------------------------------

for index in range(0, TREES):
    print(index, len(tree[index]))
    print(tree[index])

#---------------------------------------------------------------
# manually create GBM predictions from tree predictions
#---------------------------------------------------------------

ensemble = tree[0]
for index in range(1, TREES):
    ensemble = ensemble + LEARNING_RATE*tree[index]

#---------------------------------------------------------------
# CHECK:
# GBM predictions (y_hat) equals manually created ensemble of trees
#---------------------------------------------------------------

diff = ensemble - y_hat
print(diff)
assert(np.allclose(diff, np.zeros((1,OBSERVATIONS)), atol=1e-6))

#---------------------------------------------------------------
# CHECK:
# Membership matrix based on tree predicted values
# equals membership matrix based on model.predict with get_leaf
#---------------------------------------------------------------

myD1 = getD(tree[1])
myD1

theD1 =getD(instance_leaf_membership[:,0])
theD1

instance_leaf_membership[:,0]

diff = myD1 - theD1
assert(np.allclose(diff, np.zeros((OBSERVATIONS ,OBSERVATIONS)), atol=1e-6))

#---------------------------------------------------------------
# Calculations as per Geertsema & Lu (2022)
# "Relative Valuation with Machine Learning"
# See section A.3 in the Appendix
#---------------------------------------------------------------

# parameters of calculation
#-------------------

N = OBSERVATIONS
M = TREES
_lambda = LEARNING_RATE

ones = np.ones((N,N))
I = np.identity(N)
I

# lists
P = [None for k in range(0, M)]
p = [None for k in range(0, M)]
diff = [None for k in range(0, M)]
G = [None for k in range(0, M)]
g = [None for k in range(0, M)]
D = [None for k in range(0, M)]
W = [None for k in range(0, M)]

W

# iteration i = 0
#-------------------

# equation 11
P[0] = (1/N) * ones
P[0]
P[0].shape
# (N,N)

v = y.reshape(N,1).copy()
v
v.shape
# (N,1)

# equation 12
p[0] = v.T @ P[0]
# (1,N) = (1,N) x (N,N)
p[0]
p[0].shape
# (1,N)


# CHECK: p[0] is same as tree[0]
t = np.reshape(tree[0], (1,N))
t.shape
diff[0] = t - p[0]
diff[0]
diff[0].shape
print(f"difference at level {0}", diff[0])
assert(np.allclose(diff[0], np.zeros((1,N)), atol=1e-6))

# equation 13
G[0] = P[0]
G[0].shape
# (N,N) = (N,N)

# equation 14
g[0] = v.T @ G[0]
# (1,N) = (1,N) x (N,N)
g[0].shape
# (1,N)

# in first tree all obs are allocated to 1 leaf (same as taking mean)
D[0] = np.ones((N,N))

for i in range(1, M):

    print("Doing tree ", i)
    D[i] = getD(tree[i])
    D[i]
    D[i].shape
    # (N,N)

    # equation 15
    W[i] = D[i] / (ones @ D[i])
    W[i]
    W[i].shape
    # (N,N) 

    # equation 16
    P[i] = _lambda * (W[i] @  (I - G[i-1]))
    P[i]
    P[i].shape
    # (N,N)

    # equation 17
    p[i] = v.T @ P[i].T @ W[i]
    p[i]
    p[i].shape
    # (1,N)

    t = np.reshape(tree[i], (1,N))
    diff[i] = (t*_lambda) - (p[i])
    diff[i]
    diff[i].shape
    # (1,N)

    print(f"===CHECK===: difference at level {i}", np.round(diff[i], 4))
    # check that differences are close to zero
    assert(np.allclose(diff[i], np.zeros((1,N)), atol=1e-5))

    # equation 18
    G[i] = G[i-1]+P[i]
    G[i]
    G[i].shape
    # (N,N)

    # equation 19
    g[i] = v.T @ G[i]
    g[i]
    g[i].shape
    # (1,N)


#---------------------------------------------------------------
# now do GBM equivalent predictions
# using calculated weights
#---------------------------------------------------------------


# predict for all training samples
L = D.copy()

# in first tree all obs are allocated to 1 leaf (same as taking mean)
L[0] = np.ones((N, N))
L

# Create GBM predictions for entire training dataset
#--------------------------------------------------------------------

K = np.zeros((N, N))
for i in range(0, M):
    # equation 20
    K = K + P[i].T @ (L[i] / (ones @ L[i]))
    #print("i=",i)
    #print(K)

# equation 21
k = v.T @ K

k
y_hat

diff_pred = k - y_hat
diff_pred
print("===CHECK===: difference in GBM predicted ", np.round(diff_pred,6))

assert(np.allclose(diff_pred, np.zeros((1,N)), atol=1e-5))

# checks complete
print("checks complete")
