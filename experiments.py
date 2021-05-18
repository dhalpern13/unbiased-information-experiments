import uuid
from itertools import product
from os import path
from sys import argv

import pandas as pd
from scipy.stats import uniform, norm, bernoulli, beta, powerlaw

from algorithms import all_points, partition, random_subsets, opt, min_cooks_distance

if not argv[-1].endswith('.py'):
    suffix = argv[-1]
else:
    suffix = ''

output_file = f'data/results-cooks-{suffix}.csv'

algorithms = {
    'all': all_points,
    'partition': partition,
    'random_1': random_subsets(lambda m, k: 1),
    'random_(10*km)': random_subsets(lambda m, k: 10 * k * m),
    'random_1000': random_subsets(lambda m, k: 1000),
    'cooks': min_cooks_distance
}

m_values = [3, 10, 20]
k_values = [5, 20, 50, 100]
slopes = [0, 1, 5]

iterations = 100


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

x_distribution = uniform(-1, 2).rvs


def generate_xs_ys(num_points, noise_distribution, slope, x_distribution):
    xs = x_distribution((num_points, 1))
    noise = noise_distribution(num_points)
    ys = xs.dot([slope]) + noise
    return xs, ys


def run_experiment(noise_distribution, slope, m, k, include_opt=True):
    xs, ys = generate_xs_ys(num_points=m * k,
                            noise_distribution=noise_distribution,
                            slope=slope,
                            x_distribution=x_distribution)

    alg_intercepts = {alg_name: alg(xs, ys, m) for alg_name, alg in algorithms.items()}

    if include_opt and m <= 3 and k <= 20:
        opt_val = opt(xs, ys, m)
    else:
        opt_val = None

    return alg_intercepts | {'opt': opt_val}


def compute_row(noise_distribution_name, slope, m, k):
    alg_intercepts = run_experiment(
        noise_distribution=noise_distributions[noise_distribution_name],
        slope=slope,
        m=m,
        k=k,
        include_opt=False)

    row = {
              'row_id': uuid.uuid4(),
              'noise_distribution': noise_distribution_name,
              'slope': slope,
              'm': m,
              'k': k
          } | alg_intercepts

    return row


def main():
    for i in range(iterations):
        results = [compute_row(*params) for params in
                   product(noise_distributions.keys(), slopes, m_values, k_values)]

        if path.exists(output_file):
            extra_kwargs = {
                'mode': 'a',
                'header': False
            }
        else:
            extra_kwargs = {}

        pd.DataFrame(results).to_csv(output_file, index=False, **extra_kwargs)

        print(f'{output_file}: {i} iterations complete')


if __name__ == '__main__':
    main()
