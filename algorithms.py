from itertools import combinations
from random import sample

import numpy as np
import statsmodels.api as sm
from sklearn import linear_model


# Helpers

def _intercept(xs, ys):
    return linear_model.LinearRegression().fit(xs, ys).intercept_


def _best_intercept_from_indices(xs, ys, all_indices):
    intercepts = [_intercept(xs.take(indices, axis=0), ys.take(indices)) for indices in all_indices]
    return min(intercepts, key=abs)


# Algorithms

def all_points(xs, ys, m):
    return _intercept(xs, ys)


def opt(xs, ys, m):
    indices = combinations(range(len(ys)), m)
    return _best_intercept_from_indices(xs, ys, indices)


def partition(xs, ys, m):
    k = len(ys) // m
    partition_indices = [np.arange(i * m, (i + 1) * m) for i in range(k)]
    return _best_intercept_from_indices(xs, ys, partition_indices)


def random_subsets(num_times_mk_func):
    def alg(xs, ys, m):
        k = len(ys) // m
        indices = [sample(range(len(ys)), m) for _ in range(num_times_mk_func(m, k))]
        return _best_intercept_from_indices(xs, ys, indices)

    return alg


def min_cooks_distance(xs, ys, m):
    cooks_distances = sm.OLS(xs, ys).fit().get_influence().cooks_distance[0]
    indices = np.argpartition(cooks_distances, m)[:m]
    return _best_intercept_from_indices(xs, ys, [indices])
