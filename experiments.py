from itertools import product, repeat, chain
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import bernoulli, beta, norm, powerlaw, uniform
from tqdm import tqdm

from algorithms import all_points, partition, random_subsets, opt, min_cooks_distance, greedy

output_file = f'data/results.csv'

algorithms = {
    'all': all_points,
    'partition': partition,
    'random_1': random_subsets(lambda m, k: 1),
    'random_(10*km)': random_subsets(lambda m, k: 10 * k * m),
    'random_1000': random_subsets(lambda m, k: 1000),
    'cooks': min_cooks_distance,
    'greedy': greedy,
}

m_values = [3, 10, 20]
k_values = [5, 20, 50, 100]
slopes = [0, 1, 5]
iterations = 200


def mirror_distribution(dist):
    def sample(shape):
        return dist.rvs(shape) * (bernoulli(.5).rvs(shape) * 2 - 1)

    return sample


# Note Scipy Uniform takes parameters lower, length
noise_distributions = {
    'uniform(-2,2)': uniform(-2, 4).rvs,
    'normal(0,.5^2)': norm(0, .5).rvs,
    'beta(2,2)-mirrored': mirror_distribution(beta(2, 2)),
    'power-law(2)-mirrored': mirror_distribution(powerlaw(3)),
}

uniform_11 = uniform(-1, 2).rvs


def generate_xs_ys(num_points, noise_distribution, slope, x_distribution):
    xs = x_distribution(num_points)
    noise = noise_distribution(num_points)
    ys = xs * slope + noise
    return xs[:, np.newaxis], ys


def run_experiment(noise_distribution, slope, m, k):
    xs, ys = generate_xs_ys(num_points=m * k,
                            noise_distribution=noise_distribution,
                            slope=slope,
                            x_distribution=uniform_11)

    alg_intercepts = {alg_name: alg(xs, ys, m) for alg_name, alg in algorithms.items()}

    if m <= 3 and k <= 20:
        opt_val = opt(xs, ys, m)
    else:
        opt_val = None

    return alg_intercepts | {'opt': opt_val}


def compute_row(noise_distribution_name, slope, m, k):
    alg_intercepts = run_experiment(
        noise_distribution=noise_distributions[noise_distribution_name],
        slope=slope,
        m=m,
        k=k)

    row = {
              'noise_distribution': noise_distribution_name,
              'slope': slope,
              'm': m,
              'k': k
          } | alg_intercepts

    return row


def compute_row_args_seed(args):
    seed, params = args
    np.random.seed(seed)
    return compute_row(*params)


def main():
    param_combos = list(product(noise_distributions.keys(), slopes, m_values, k_values))
    with Pool(8) as p:
        rows = list(tqdm(p.imap_unordered(compute_row_args_seed,
                                          enumerate(chain.from_iterable(repeat(param_combos, times=iterations)))),
                         total=iterations * len(param_combos)))

    pd.DataFrame(rows).to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
