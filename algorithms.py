from itertools import combinations
from random import sample

import numpy as np
import statsmodels.api as sm
from math import isclose
from sklearn import linear_model


# Helpers

def _intercept(xs, ys, indices):
    used_xs = xs[indices, :]
    used_ys = ys[np.array(indices)]
    return linear_model.LinearRegression().fit(used_xs, used_ys).intercept_


def _best_intercept_from_indices(xs, ys, all_indices):
    intercepts = [_intercept(xs, ys, indices) for indices in all_indices]
    return min(intercepts, key=abs)


# Algorithms

def all_points(xs, ys, m):
    return _intercept(xs, ys, range(len(ys)))


def opt(xs, ys, m):
    indices = combinations(range(len(ys)), m)
    return _best_intercept_from_indices(xs, ys, indices)


def partition(xs, ys, m):
    k = len(ys) // m
    partition_indices = [np.arange(i * m, (i + 1) * m) for i in range(k)]
    return _best_intercept_from_indices(xs, ys, partition_indices)


def greedy(xs, ys, m):
    remaining = set(range(len(ys)))
    first_point = np.abs(ys).argmin()
    cur_indices = [first_point]
    remaining.remove(first_point)
    for _ in range(m - 1):
        best_index = min(remaining, key=lambda index: abs(_intercept(xs, ys, cur_indices + [index])))
        cur_indices.append(best_index)
        remaining.remove(best_index)
    return _intercept(xs, ys, cur_indices)


def random_subsets(num_times_mk_func):
    def alg(xs, ys, m):
        k = len(ys) // m
        indices = [sample(range(len(ys)), m) for _ in range(num_times_mk_func(m, k))]
        return _best_intercept_from_indices(xs, ys, indices)

    return alg


# def uniform_from_simplex(dims, iterations):
#     s = np.random.exponential(scale=1, size=(iterations, dims))
#     return s / s.sum(axis=1, keepdims=True) * dims
#
#
# def get_best_weight_combo(xs, ys, all_weights, combination_search_steps):
#     intercepts = np.array(
#         [linear_model.LinearRegression().fit(xs, ys, sample_weight=weights).intercept_ for weights in all_weights])
#     if intercepts.min() > 0:
#         return all_weights[intercepts.argmin()]
#     if intercepts.max() < 0:
#         return all_weights[intercepts.argmax()]
#     biggest_negative = all_weights[np.where(intercepts > 0, intercepts, -np.inf).argmax()]
#     smallest_positive = all_weights[np.where(intercepts < 0, intercepts, np.inf).argmin()]
#     lc_steps = np.arange(combination_search_steps + 1) / combination_search_steps
#     combos = lc_steps[:, None] * biggest_negative[None, :] + (1 - lc_steps)[:, None] * smallest_positive[None, :]
#     return min(combos,
#                key=lambda weights: abs(linear_model.LinearRegression().fit(xs, ys, sample_weight=weights).intercept_))
#
#
# def randomized_rounding(weights, num):
#     weights = np.array(weights)
#     weights = weights / weights.sum() * num
#
#     selected_indices = []
#     remaining_indices = list(range(len(weights)))
#
#     for i in range(len(weights)):
#         if isclose(weights[i], 0):
#             remaining_indices.remove(i)
#             weights[i] = 0
#     while weights.max() >= 1:
#         i = weights.argmax()
#         selected_indices.append(i)
#         remaining_indices.remove(i)
#         weights[i] = 0
#         weights = weights / weights.sum() * (num - len(selected_indices))
#
#     while len(remaining_indices) > 1:
#         x, y = sample(remaining_indices, 2)
#         a, b = weights[[x, y]]
#         if isclose(a + b, 1):
#             include_x = np.random.binomial(1, a)
#             if include_x:
#                 included, removed = x, y
#             else:
#                 included, removed = y, x
#             selected_indices.append(included)
#             remaining_indices.remove(included)
#             remaining_indices.remove(removed)
#         elif a + b < 1:
#             remove_y = np.random.binomial(1, a / (a + b))
#             if remove_y:
#                 remain, removed = x, y
#             else:
#                 remain, removed = y, x
#             remaining_indices.remove(removed)
#             weights[remain] = a + b
#         else:
#             include_x = np.random.binomial(1, (1 - b) / (2 - a - b))
#             if include_x:
#                 included, remain = x, y
#             else:
#                 included, remain = y, x
#             selected_indices.append(included)
#             remaining_indices.remove(included)
#             weights[remain] = a + b - 1
#
#     return selected_indices
#
#
# def weighted_dependent_rounding(weight_search_steps, combination_search_steps, random_round_steps):
#     def alg(xs, ys, m):
#         all_weights = uniform_from_simplex(len(ys), weight_search_steps)
#         best_weights = get_best_weight_combo(xs, ys, all_weights, combination_search_steps)
#         possible_indices = [randomized_rounding(best_weights, m) for _ in range(random_round_steps)]
#         return _best_intercept_from_indices(xs, ys, possible_indices)
#
#     return alg


def min_cooks_distance(xs, ys, m):
    cooks_distances = sm.OLS(xs, ys).fit().get_influence().cooks_distance[0]
    indices = np.argpartition(cooks_distances, m)[:m]
    return _best_intercept_from_indices(xs, ys, [indices])
